import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# 设置目标归一化参数（基于非 NaN 值计算得到）
target_mean = 33.573902
target_std = 18.321875

####################################
# 1. 定义数据集类，并过滤全为 NaN 的 patch
####################################
class BorneoDataset(Dataset):
    def __init__(self, rep_files, target_files, transform=None):
        """
        :param rep_files: representation .npy 文件列表，形状 (100,100,128)
        :param target_files: target .npy 文件列表，形状 (100,100)，其中部分值可能为 NaN
        :param transform: 可选的 transform
        """
        # 过滤掉那些 target 全为 NaN 的 patch
        valid_rep_files = []
        valid_target_files = []
        for rep_file, target_file in zip(rep_files, target_files):
            target = np.load(target_file)
            if np.all(np.isnan(target)):
                print(f"Skipping {target_file} as all values are NaN.")
                continue
            valid_rep_files.append(rep_file)
            valid_target_files.append(target_file)
        
        self.rep_files = valid_rep_files
        self.target_files = valid_target_files
        self.transform = transform
        
    def __len__(self):
        return len(self.rep_files)
    
    def __getitem__(self, idx):
        # 加载 representation 和 target
        rep = np.load(self.rep_files[idx])
        target = np.load(self.target_files[idx])
        
        # representation: (100,100,128) -> 转为 (128,100,100)
        rep = torch.from_numpy(rep).float().permute(2, 0, 1)
        
        # target: (100,100) -> 转为 (1,100,100)
        target = torch.from_numpy(target).float().unsqueeze(0)
        # 对 target 进行归一化，nan 保持不变
        target = (target - target_mean) / target_std
        
        if self.transform:
            rep = self.transform(rep)
            target = self.transform(target)
            
        return rep, target

####################################
# 2. 定义 UNet 模型
####################################
class DoubleConv(nn.Module):
    """两次 3x3 卷积 + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=128, out_channels=1, features=[128, 256, 512]):
        """
        :param in_channels: 输入通道数（这里为 128）
        :param out_channels: 输出通道数（回归任务设为 1）
        :param features: 编码器中每层的特征数
        """
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 构建编码器（Downsampling）
        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in_channels, feature))
            curr_in_channels = feature
        
        # Bottleneck 层
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # 构建解码器（Upsampling）
        for feature in reversed(features):
            # 使用转置卷积进行上采样
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            # 拼接后通道数为 2*feature，再经过双卷积降回 feature
            self.ups.append(DoubleConv(feature*2, feature))
        
        # 最终的 1x1 卷积，将通道数映射为 1
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        # 编码器部分：保存每个阶段的特征用于跳跃连接
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 反转以便与解码器对应
        
        # 解码器部分：上采样、拼接、双卷积
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            # 如果尺寸不匹配，则进行插值调整
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
            
        return self.final_conv(x)

####################################
# 新增：最简单的卷积模型，用于回归任务（带下采样与上采样）
####################################
class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=128, out_channels=1, dropout_prob=0.2):
        super(SimpleConvNet, self).__init__()
        # Encoder: 下采样
        self.encoder = nn.Sequential(
            # Layer 1:
            # Conv2d: 输入 (B, 128, H, W) -> 输出 (B, 256, H/2, W/2)
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            # Layer 2:
            # Conv2d: 输入 (B, 256, H/2, W/2) -> 输出 (B, 512, H/4, W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )
        
        # Decoder: 上采样恢复到原始尺寸
        self.decoder = nn.Sequential(
            # Layer 3:
            # ConvTranspose2d: 输入 (B, 512, H/4, W/4) -> 输出 (B, 64, H/2, W/2)
            nn.ConvTranspose2d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            # Layer 4:
            # ConvTranspose2d: 输入 (B, 64, H/2, W/2) -> 输出 (B, out_channels, H, W)
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        # 输入 x: (B, in_channels, H, W)
        x = self.encoder(x)  # 经过 encoder 后: (B, 128, H/4, W/4)
        x = self.decoder(x)  # 经过 decoder 后: (B, out_channels, H, W)
        return x

####################################
# MobileNetV3 部分：倒置残差块与特征提取器（修改后用于回归任务）
####################################
class InvertedResidual(nn.Module):
    """
    采用 MobileNetV3 风格的倒置残差块，包含 1x1 扩张、3x3 深度可分离卷积和 1x1 投影，
    并在扩张和深度卷积后各加入 BatchNorm、ReLU 和 Dropout2d。
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dropout_prob=0.1):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        self.conv = nn.Sequential(
            # 1x1 扩张卷积：将通道数从 in_channels 提升到 hidden_dim
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            # 3x3 深度卷积：stride 控制下采样，padding=1 保持尺寸
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            # 1x1 投影卷积：将通道数压缩到 out_channels
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class FeatureExtractorMobileNetV3(nn.Module):
    """
    特征提取器：输入形状 (B,128,64,64)
    Encoder 部分逐步下采样并提高通道数：
      1. initial_conv：使用 3x3 卷积将通道从 128 提升到 160，同时下采样 64x64->32x32
      2. block1：倒置残差块，将 160 -> 192，stride=2，实现 32x32->16x16
      3. block2：倒置残差块，将 192 -> 256，stride=2，实现 16x16->8x8
      4. block3：倒置残差块，将 256 -> 320，stride=2，实现 8x8->4x4
      5. final_conv：1x1 卷积将通道提升到 512，保持 4x4 尺寸
    接下来通过上采样（scale_factor=16，将 4x4 放大到 64x64），再通过 decoder 将通道调整为 1。
    """
    def __init__(self, dropout_prob=0.1):
        super(FeatureExtractorMobileNetV3, self).__init__()
        # 初始卷积：保持信息同时提升通道
        # 输入: (B,128,64,64) -> 输出: (B,160,32,32)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )
        # Block1：160 -> 192，下采样: (32,32) -> (16,16)
        self.block1 = InvertedResidual(160, 192, stride=2, expand_ratio=2, dropout_prob=dropout_prob)
        # Block2：192 -> 256，下采样: (16,16) -> (8,8)
        # self.block2 = InvertedResidual(192, 256, stride=2, expand_ratio=2, dropout_prob=dropout_prob)
        # # Block3：256 -> 320，下采样: (8,8) -> (4,4)
        # self.block3 = InvertedResidual(256, 320, stride=2, expand_ratio=2, dropout_prob=dropout_prob)
        # 最后 1x1 卷积：将通道提升到 512，保持尺寸 4x4
        self.final_conv = nn.Sequential(
            nn.Conv2d(192, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )
        # Decoder：先上采样到原始尺寸，再将通道调整为 1
        # 上采样：将 (B,512,4,4) 通过双线性插值放大到 (B,512,64,64)
        # Decoder: 使用 3x3 卷积（BN+ReLU+Dropout）再 1x1 卷积映射到 1 通道
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(128, 1, kernel_size=1)
        )
    
    def forward(self, x):
        # 输入 x: (B,128,64,64)
        x = self.initial_conv(x)   # -> (B,160,32,32)
        x = self.block1(x)         # -> (B,192,16,16)
        # x = self.block2(x)         # -> (B,256,8,8)
        # x = self.block3(x)         # -> (B,320,4,4)
        x = self.final_conv(x)     # -> (B,512,4,4)
        # 上采样到 (B,512,64,64)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.decoder(x)        # -> (B,1,64,64)
        return x

####################################
# 3. 数据集划分与 DataLoader
####################################
def get_file_lists(root_dir):
    rep_dir = os.path.join(root_dir, "representation")
    target_dir = os.path.join(root_dir, "target")
    
    rep_files = sorted(glob.glob(os.path.join(rep_dir, "*.npy")))
    target_files = sorted(glob.glob(os.path.join(target_dir, "*.npy")))
    
    return rep_files, target_files

def split_data(rep_files, target_files, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8):
    total = len(rep_files)
    indices = np.arange(total)
    np.random.shuffle(indices)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_rep = [rep_files[i] for i in train_idx]
    train_target = [target_files[i] for i in train_idx]
    
    val_rep = [rep_files[i] for i in val_idx]
    val_target = [target_files[i] for i in val_idx]
    
    test_rep = [rep_files[i] for i in test_idx]
    test_target = [target_files[i] for i in test_idx]
    
    return (train_rep, train_target), (val_rep, val_target), (test_rep, test_target)

####################################
# 4. 定义指标计算与 Masked Loss
####################################
def masked_mse_loss(pred, target):
    """
    仅对 target 中非 NaN 部分计算 MSE Loss；
    如果全部为 NaN，则返回一个与计算图关联的零损失。
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        # 返回一个与 pred 计算图相关联的零损失
        return (pred * 0).sum()
    loss = torch.mean((pred[mask] - target[mask]) ** 2)
    return loss

def compute_metrics(pred, target):
    """
    计算 MAE, RMSE, R2 指标，排除 target 中的 NaN 值
    先反归一化，再计算指标
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    
    # 反归一化
    pred_denorm = pred * target_std + target_mean
    target_denorm = target * target_std + target_mean

    # 仅计算有效像素
    pred_valid = pred_denorm[mask]
    target_valid = target_denorm[mask]
    
    mae = torch.mean(torch.abs(pred_valid - target_valid))
    rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))
    
    # 计算 R2：先 flatten 有效像素
    pred_flat = pred_valid.view(-1)
    target_flat = target_valid.view(-1)
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else torch.tensor(0.0)
    
    return mae.item(), rmse.item(), r2.item()

####################################
# 5. 训练、验证与测试过程（含checkpoint保存与加载）
####################################
def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    # 用于保存最佳验证loss
    best_val_loss = float('inf')
    checkpoint_dir = os.path.join("checkpoints", "downstream")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_borneo_patch_ckpt.pth")
    
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for batch_idx, (rep, target) in enumerate(progress_bar):
            rep = rep.to(device)       # (B, 128, 64, 64)
            target = target.to(device) # (B, 1, 64, 64) 其中部分像素可能为 NaN
            
            optimizer.zero_grad()
            output = model(rep)        # 输出应为 (B, 1, 64, 64)
            loss = masked_mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            mae, rmse, r2 = compute_metrics(output, target)
            
            # 实时更新进度条显示指标
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MAE": f"{mae:.4f}",
                "RMSE": f"{rmse:.4f}",
                "R2": f"{r2:.4f}"
            })
        
        # 验证集评估
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0
        val_r2 = 0.0
        with torch.no_grad():
            for rep, target in val_loader:
                rep = rep.to(device)
                target = target.to(device)
                output = model(rep)
                loss = masked_mse_loss(output, target)
                val_loss += loss.item()
                mae, rmse, r2 = compute_metrics(output, target)
                val_mae += mae
                val_rmse += rmse
                val_r2 += r2
        
        num_val = len(val_loader)
        avg_val_loss = val_loss / num_val
        print(f"Epoch {epoch}/{epochs} - Train Loss: {np.mean(train_losses):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae/num_val:.4f} | "
              f"Val RMSE: {val_rmse/num_val:.4f} | Val R2: {val_r2/num_val:.4f}")
        
        # 保存验证集loss最好的checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved new best checkpoint at epoch {epoch} with Val Loss: {avg_val_loss:.4f}")
            
def main():
    # 数据所在的根目录
    root_dir = "/mnt/e/Codes/btfm4rs/data/downstream/borneo_patch"
    rep_files, target_files = get_file_lists(root_dir)
    
    (train_rep, train_target), (val_rep, val_target), (test_rep, test_target) = split_data(
        rep_files, target_files, train_ratio=0.3, val_ratio=0.1, test_ratio=0.6)
    
    # 创建数据集与 DataLoader
    train_dataset = BorneoDataset(train_rep, train_target)
    val_dataset = BorneoDataset(val_rep, val_target)
    test_dataset = BorneoDataset(test_rep, test_target)
    
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 模型与设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用 UNet 模型
    model = UNet(in_channels=128, out_channels=1)
    # 使用新的最简单的卷积模型
    # model = SimpleConvNet(in_channels=128, out_channels=1)
    # model = FeatureExtractorMobileNetV3(dropout_prob=0.1)
    
    # 打印模型参数个数
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters.")
    
    # 训练模型
    train_model(model, train_loader, val_loader, device, epochs=200, lr=1e-3)
    
    # 加载最佳的checkpoint再进行测试
    checkpoint_path = os.path.join("checkpoints", "downstream", "best_borneo_patch_ckpt.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}")
    else:
        print("No checkpoint found. Using current model for testing.")
    
    # 测试集评估
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_rmse = 0.0
    test_r2 = 0.0
    with torch.no_grad():
        for rep, target in test_loader:
            rep = rep.to(device)
            target = target.to(device)
            output = model(rep)
            loss = masked_mse_loss(output, target)
            test_loss += loss.item()
            mae, rmse, r2 = compute_metrics(output, target)
            test_mae += mae
            test_rmse += rmse
            test_r2 += r2
            
    num_test = len(test_loader)
    print(f"Test Loss: {test_loss/num_test:.4f} | "
          f"Test MAE: {test_mae/num_test:.4f} | Test RMSE: {test_rmse/num_test:.4f} | "
          f"Test R2: {test_r2/num_test:.4f}")

if __name__ == "__main__":
    main()
