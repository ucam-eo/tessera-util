import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import logging
from einops import rearrange
import pandas as pd
from datetime import datetime

# -----------------------------------------------------------------------------
# FocalLoss损失函数定义
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        """
        Args:
            gamma (float): 聚焦参数γ，默认为2。
            weight (Tensor, optional): 类别权重。
            reduction (str): 损失计算方式：'mean', 'sum' 或 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        logpt = nn.functional.log_softmax(input, dim=1)
        ce_loss = nn.functional.nll_loss(logpt, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------------------------------------------------------------------
# Dataset类：支持训练、验证和测试划分
# -----------------------------------------------------------------------------
class LandClassificationDataset(Dataset):
    def __init__(self, representations, labels, field_ids, split='train',
                 train_field_ids=None, val_field_ids=None, test_field_ids=None):
        """
        Args:
            representations (ndarray): 特征表示，形状 (H, W, D)
            labels (ndarray): 标签，形状 (H, W)
            field_ids (ndarray): 字段ID，形状 (H, W)
            split (str): 'train', 'val' 或 'test'
            train_field_ids (list): 训练集使用的字段ID列表
            val_field_ids (list): 验证集使用的字段ID列表
            test_field_ids (list): 测试集使用的字段ID列表
        """
        valid_mask = labels != 0  # 过滤掉标签为0的无效像素
        if split == 'train':
            assert train_field_ids is not None, "训练集需要传入train_field_ids"
            mask = np.isin(field_ids, train_field_ids) & valid_mask
        elif split == 'val':
            assert val_field_ids is not None, "验证集需要传入val_field_ids"
            mask = np.isin(field_ids, val_field_ids) & valid_mask
        elif split == 'test':
            assert test_field_ids is not None, "测试集需要传入test_field_ids"
            mask = np.isin(field_ids, test_field_ids) & valid_mask
        else:
            raise ValueError("split 必须为 'train', 'val' 或 'test'")
        
        # 将符合条件的像素抽取出来，形状为 (N, D)
        self.representations = rearrange(representations[mask], 'n d -> n d')
        self.labels = labels[mask]

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.representations[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# -----------------------------------------------------------------------------
# 分类头网络定义
# -----------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.ln1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.ln2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu1(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def load_and_dequantize_representation(representation_file_path, scales_file_path):
    """
    Load and dequantize int8 representations back to float32.
    
    Args:
        representation_file_path: Path to the int8 representation file (H,W,C)
        scales_file_path: Path to the float32 scales file (H,W)
    
    Returns:
        representation_f32: float32 ndarray of shape (H,W,C)
    """
    # Load the files
    representation_int8 = np.load(representation_file_path)  # (H, W, C), dtype=int8
    scales = np.load(scales_file_path)  # (H, W), dtype=float32
    
    # Convert int8 to float32 for computation
    representation_f32 = representation_int8.astype(np.float32)
    
    # Expand scales to match representation shape
    # scales shape: (H, W) -> (H, W, 1)
    scales_expanded = scales[..., np.newaxis]
    
    # Dequantize by multiplying with scales
    representation_f32 = representation_f32 * scales_expanded
    
    return representation_f32

# -----------------------------------------------------------------------------
# 主函数：数据加载、数据集划分、训练、验证和最终测试
# -----------------------------------------------------------------------------
def main():
    # 固定随机种子，便于结果复现
    np.random.seed(42)
    torch.manual_seed(42)
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    
    # -------------------------
    # 定义文件路径
    # -------------------------
    # representation_file_path = "data/representation/austrian_crop_downsample_100_fsdp_20250407_195912.npy"
    # representation_file_path = "data/representation/austrian_crop_whole_year_downsample_100_fsdp_20250427_084307.npy"
    # representation_file_path = "/mnt/e/Codes/btfm4rs/data/representation/Austria_EFM_Embeddings_2021_method1_100m.npy"
    representation_file_path = "data/representation/mpc_pipeline_fsdp_20250605_221257_downsample_100.npy"
    # representation_file_path = "data/representation/austria_Presto_embeddings_100m.npy"
    # representation_file_path = "data/representation/mpc_pipeline_fsdp_20250604_100313_downsample_100.npy"
    # representation_file_path = "data/representation/mpc_pipeline_fsdp_20250408_101211_downsample_100.npy"
    # representation_file_path = "data/representation/austrian_crop_EFM_Embeddings_2022_downsample_100.npy"
    label_file_path = "data/downstream/austrian_crop/fieldtype_17classes_downsample_100.npy"
    field_id_file_path = "data/downstream/austrian_crop/fieldid_downsample_100.npy"
    updated_fielddata_path = "data/downstream/austrian_crop/updated_fielddata.csv"
    
    training_ratio = 0.3

    # -------------------------d
    # 加载representation、标签和字段ID
    # -------------------------
    representations = np.load(representation_file_path)  # (H, W, D)
    
    # representation_file_path = "data/representation/austrian_crop_fsdp_20250606_162720_QAT_downsample_100.npy"
    # scales_file_path = "data/representation/austrian_crop_fsdp_20250606_162720_QAT_scales_downsample_100.npy"
    # representations = load_and_dequantize_representation(representation_file_path, scales_file_path)  # (H, W, D)
    
    labels = np.load(label_file_path).astype(np.int64)     # (H, W)
    field_ids = np.load(field_id_file_path).astype(np.int64) # (H, W)

    # 检查并调整representation的H和W以匹配labels
    if representations.shape[0] != labels.shape[0] or representations.shape[1] != labels.shape[1]:
        logging.info(f"Representation shape {representations.shape[:2]} does not match labels shape {labels.shape}. Resizing representation...")
        # 将 (H, W, D) 转换为 (D, H, W) 以便插值
        representations_tensor = torch.tensor(representations).permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
        # 使用双线性插值进行resize
        resized_representations_tensor = nn.functional.interpolate(
            representations_tensor,
            size=(labels.shape[0], labels.shape[1]),
            mode='bilinear',
            align_corners=False
        )
        # 将 (1, D, H_new, W_new) 转换回 (H_new, W_new, D) 并转为numpy
        representations = resized_representations_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        logging.info(f"Resized representation shape: {representations.shape}")
    
    # 重映射标签：确保标签连续（旧代码中将标签0视为无效）
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")
    num_classes = len(unique_labels)
    print(f"Number of valid classes: {num_classes}")
    label_map = {label: i for i, label in enumerate(unique_labels)}
    labels = np.vectorize(label_map.get)(labels)
    
    # -------------------------
    # 利用updated_fielddata.csv划分训练、验证和测试集
    # 旧代码中按SNAR_CODE分组，每个组取30%的面积作为训练集，其余部分随机划分验证和测试集
    # -------------------------
    fielddata_df = pd.read_csv(updated_fielddata_path)
    area_summary = fielddata_df.groupby('SNAR_CODE')['area_m2'].sum().reset_index()
    area_summary.rename(columns={'area_m2': 'total_area'}, inplace=True)
    
    all_selected_fids = []
    for _, row in area_summary.iterrows():
        sn_code = row['SNAR_CODE']
        total_area = row['total_area']
        target_area = total_area * training_ratio
        selected_rows = fielddata_df[fielddata_df['SNAR_CODE'] == sn_code].sort_values(by='area_m2')
        
        selected_fids = []
        selected_area_sum = 0
        for _, selected_row in selected_rows.iterrows():
            if selected_area_sum < target_area:
                selected_fids.append(int(selected_row['fid_1']))
                selected_area_sum += selected_row['area_m2']
            else:
                break
        all_selected_fids.extend(selected_fids)
    
    all_selected_fids = [int(fid) for fid in all_selected_fids]
    logging.info(f"Selected field IDs for training (last 20): {all_selected_fids[-20:]}")
    
    # 划分验证集和测试集：其余字段中随机分出1/7作为验证集，其余为测试集
    all_fields = fielddata_df['fid_1'].unique()
    set_all = set(all_fields)
    set_train = set(all_selected_fids)
    remaining = list(set_all - set_train)
    np.random.shuffle(remaining)
    val_test_split_ratio = 1/7.0
    val_count = int(len(remaining) * val_test_split_ratio)
    val_fids = remaining[:val_count]
    test_fids = remaining[val_count:]
    
    # -------------------------
    # 构造数据集与DataLoader
    # -------------------------
    train_dataset = LandClassificationDataset(representations, labels, field_ids, split='train', train_field_ids=all_selected_fids)
    val_dataset = LandClassificationDataset(representations, labels, field_ids, split='val', val_field_ids=val_fids)
    test_dataset = LandClassificationDataset(representations, labels, field_ids, split='test', test_field_ids=test_fids)
    
    logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    batch_size = 8192
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # -------------------------
    # 模型、损失函数、优化器
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = representations.shape[-1]
    model = ClassificationHead(input_dim, num_classes).to(device)
    # 使用FocalLoss替换原来的交叉熵损失函数
    # criterion = FocalLoss(gamma=2).to(device)
    # 使用交叉熵
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # -------------------------
    # 训练配置与checkpoint设置
    # -------------------------
    num_epochs = 200
    log_interval = 32  # 每隔一定步数打印日志
    best_val_f1 = 0.0
    best_val_accuracy = 0.0
    best_epoch = 0
    checkpoint_save_folder = "checkpoints/downstream/"
    os.makedirs(checkpoint_save_folder, exist_ok=True)
    best_checkpoint_name = f"austrian_crop_representation_checkpoint_{timestamp}_best.pt"
    best_checkpoint_path = os.path.join(checkpoint_save_folder, best_checkpoint_name)
    
    # -------------------------
    # 训练和验证循环
    # -------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        for step, (reps, labels_batch) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            reps, labels_batch = reps.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(reps)
            # 将输出reshape为 [N, num_classes]，labels同理
            outputs = outputs.view(-1, num_classes)
            labels_batch = labels_batch.view(-1)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(labels_batch.cpu().numpy())
            
            if (step + 1) % log_interval == 0:
                if len(train_preds) > 0:
                    flat_preds = np.array(train_preds)
                    flat_targets = np.array(train_targets)
                    # 忽略标签为0的无效样本
                    valid_mask = flat_targets != 0
                    valid_preds = flat_preds[valid_mask]
                    valid_targets = flat_targets[valid_mask]
                    
                    train_loss = running_loss / log_interval
                    train_accuracy = accuracy_score(valid_targets, valid_preds)
                    train_f1 = f1_score(valid_targets, valid_preds, average='weighted')
                    print(f"Step [{step+1}/{len(train_loader)}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
                    logging.info(f"Step [{step+1}/{len(train_loader)}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
                else:
                    logging.info(f"Step [{step+1}/{len(train_loader)}] - No valid predictions to log.")
                running_loss = 0.0
                train_preds, train_targets = [], []
                
        # 每个epoch结束后进行验证
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for reps, labels_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                reps, labels_batch = reps.to(device), labels_batch.to(device)
                outputs = model(reps)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(labels_batch.cpu().numpy())
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")
        
        # 如果当前epoch的验证表现更好，则保存checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_checkpoint_path)
            logging.info(f"Best checkpoint saved at epoch {best_epoch} with val_accuracy: {best_val_accuracy:.4f}, val_f1: {best_val_f1:.4f}")
            # 打印验证集分类报告
            class_report = classification_report(val_targets, val_preds, digits=4)
            print("\nValidation Classification Report:\n")
            print(class_report)
            logging.info("\nValidation Classification Report:\n" + class_report)
    
    print(f"Training completed. Best Validation Accuracy: {best_val_accuracy:.4f}, Best Validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    logging.info(f"Training completed. Best Validation Accuracy: {best_val_accuracy:.4f}, Best Validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # -------------------------
    # 最终在测试集上进行评估：加载最佳checkpoint
    # -------------------------
    logging.info(f"Loading best checkpoint from epoch {best_epoch} for final test evaluation.")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for reps, labels_batch in tqdm(test_loader, desc="Final Test Evaluation"):
            reps, labels_batch = reps.to(device), labels_batch.to(device)
            outputs = model(reps)
            test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            test_targets.extend(labels_batch.cpu().numpy())
    test_accuracy = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='weighted')
    test_class_report = classification_report(test_targets, test_preds, digits=4)
    
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test F1: {test_f1:.4f}")
    print("\nTest Classification Report:\n")
    print(test_class_report)
    logging.info(f"Final Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Final Test F1: {test_f1:.4f}")
    logging.info("\nTest Classification Report:\n" + test_class_report)

if __name__ == "__main__":
    main()





