#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

####################################
# 1. 定义数据集类（分割任务，无需过滤全为 NaN 的 patch）
#    增加数据增强（随机水平/垂直翻转）以缓解过拟合
####################################
class AustrianCropPatchDataset(Dataset):
    def __init__(self, rep_files, target_files, transform=None, augment=False):
        """
        :param rep_files: representation .npy 文件列表，形状 (patch_size, patch_size, 128)
        :param target_files: label .npy 文件列表，形状 (patch_size, patch_size)，类别为整数，其中背景为0，不参与loss和metric计算
        :param transform: 可选的 transform
        :param augment: 是否对数据进行数据增强（训练时可设置为True）
        """
        self.rep_files = rep_files
        self.target_files = target_files
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.rep_files)
    
    def __getitem__(self, idx):
        rep = np.load(self.rep_files[idx])
        # representation: (patch_size, patch_size, 128) -> (128, patch_size, patch_size)
        rep = torch.from_numpy(rep).float().permute(2, 0, 1)
        
        target = np.load(self.target_files[idx])
        # target: (patch_size, patch_size) -> 转为 long 类型
        target = torch.from_numpy(target).long()
        
        # 数据增强：随机水平和垂直翻转
        if self.augment:
            if random.random() > 0.5:
                rep = torch.flip(rep, dims=[2])  # 水平翻转：对W维度翻转
                target = torch.flip(target, dims=[1])
            if random.random() > 0.5:
                rep = torch.flip(rep, dims=[1])  # 垂直翻转：对H维度翻转
                target = torch.flip(target, dims=[0])
        
        if self.transform:
            rep = self.transform(rep)
            target = self.transform(target)
            
        return rep, target

####################################
# 2. 定义 UNet 模型（仅修改最后一层输出为 num_classes）
#    在 DoubleConv 中增加 Dropout 来缓解过拟合
####################################
class DoubleConv(nn.Module):
    """两次 3x3 卷积 + BatchNorm + ReLU，并在末尾加入 Dropout（若 dropout_rate>0）"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.double_conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=128, num_classes=18, features=[128, 256, 512], dropout=0.1):
        """
        :param in_channels: 输入通道数（这里为 128）
        :param num_classes: 输出类别数（分割任务，背景为0，不参与loss和metric计算）
        :param features: 编码器中每层的特征数
        :param dropout: Dropout 概率
        """
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in_channels, feature, dropout_rate=dropout))
            curr_in_channels = feature
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_rate=dropout)
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature, dropout_rate=dropout))
        
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
            
        return self.final_conv(x)

####################################
# 3. 数据集划分与 DataLoader
####################################
def get_file_lists(root_dir):
    """
    从指定的根目录中查找 representation 与 label 文件
    representation 文件命名格式为 patch_{i}.npy
    label 文件命名格式为 label_{i}.npy
    """
    rep_files = sorted(glob.glob(os.path.join(root_dir, "patch_*.npy")))
    target_files = sorted(glob.glob(os.path.join(root_dir, "label_*.npy")))
    return rep_files, target_files

def split_data(rep_files, target_files, train_ratio=0.3, val_ratio=0.1, test_ratio=0.6):
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
# 4. 定义指标计算与损失函数（忽略背景像素，背景标签为0）
####################################
def compute_segmentation_metrics(pred, target, num_classes):
    """
    计算整体分割指标（忽略背景像素）：accuracy, F1, mIoU
    :param pred: 预测标签，形状 (B, H, W)
    :param target: 真实标签，形状 (B, H, W)
    :param num_classes: 输出类别数
    :return: accuracy, mean F1, mean IoU
    """
    # 忽略背景（标签为0）
    mask = target != 0
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    pred_masked = pred[mask]
    target_masked = target[mask]
    
    acc = (pred_masked == target_masked).float().mean().item()
    
    f1s = []
    ious = []
    for cls in range(1, num_classes):
        tp = ((pred_masked == cls) & (target_masked == cls)).sum().item()
        fp = ((pred_masked == cls) & (target_masked != cls)).sum().item()
        fn = ((pred_masked != cls) & (target_masked == cls)).sum().item()
        
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        ious.append(iou)
        
    mean_f1 = np.mean(f1s) if f1s else 0.0
    mean_iou = np.mean(ious) if ious else 0.0
    
    return acc, mean_f1, mean_iou

def compute_per_class_metrics(all_preds, all_targets, num_classes):
    """
    计算每个类别（排除背景0）的指标：accuracy, F1, IoU
    :param all_preds: tensor, 形状 (N, H, W)
    :param all_targets: tensor, 形状 (N, H, W)
    :param num_classes: 类别数
    :return: dict, key为类别编号，value为 {'acc':..., 'f1':..., 'iou':...}
    """
    metrics = {}
    # 将预测和真实标签展平
    preds = all_preds.view(-1)
    targets = all_targets.view(-1)
    for cls in range(1, num_classes):
        # 针对类别 cls，二值化
        pred_cls = (preds == cls).long()
        target_cls = (targets == cls).long()
        # 计算各项指标
        tp = ((pred_cls == 1) & (target_cls == 1)).sum().item()
        fp = ((pred_cls == 1) & (target_cls == 0)).sum().item()
        tn = ((pred_cls == 0) & (target_cls == 0)).sum().item()
        fn = ((pred_cls == 0) & (target_cls == 1)).sum().item()
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        metrics[cls] = {"acc": acc, "f1": f1, "iou": iou}
    return metrics

####################################
# 5. 训练、验证与测试过程（含 checkpoint 保存与加载）
####################################
def train_model(model, train_loader, val_loader, device, num_classes, epochs=500, lr=1e-3):
    # 使用 weight_decay 来增加 L2 正则化
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.to(device)
    
    # 学习率调度器：当验证 loss 不下降时降低 lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    checkpoint_dir = os.path.join("checkpoints", "downstream")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_austriancrop_patch_seg_ckpt.pth")
    
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for batch_idx, (rep, target) in enumerate(progress_bar):
            rep = rep.to(device)       # (B, 128, H, W)
            target = target.to(device) # (B, H, W)
            
            optimizer.zero_grad()
            output = model(rep)        # (B, num_classes, H, W)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            pred = output.argmax(dim=1)  # (B, H, W)
            acc, f1, miou = compute_segmentation_metrics(pred, target, num_classes)
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{acc:.4f}",
                "F1": f"{f1:.4f}",
                "mIoU": f"{miou:.4f}"
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        val_miou = 0.0
        with torch.no_grad():
            for rep, target in val_loader:
                rep = rep.to(device)
                target = target.to(device)
                output = model(rep)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                acc, f1, miou = compute_segmentation_metrics(pred, target, num_classes)
                val_acc += acc
                val_f1 += f1
                val_miou += miou
                
        num_val = len(val_loader)
        avg_val_loss = val_loss / num_val
        avg_val_acc = val_acc / num_val
        avg_val_f1 = val_f1 / num_val
        avg_val_miou = val_miou / num_val
        print(f"Epoch {epoch}/{epochs} - Train Loss: {np.mean(train_losses):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | "
              f"Val F1: {avg_val_f1:.4f} | Val mIoU: {avg_val_miou:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # 保存验证 loss 最好的 checkpoint
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
    # 数据所在的根目录（请根据实际情况修改路径）
    root_dir = "/mnt/e/Codes/btfm4rs/data/downstream/austrian_crop_patch"
    rep_files, target_files = get_file_lists(root_dir)
    
    (train_rep, train_target), (val_rep, val_target), (test_rep, test_target) = split_data(
        rep_files, target_files, train_ratio=0.2, val_ratio=0.1, test_ratio=0.7)
    
    # 训练集可开启数据增强
    train_dataset = AustrianCropPatchDataset(train_rep, train_target, augment=True)
    # 验证、测试集不做数据增强
    val_dataset = AustrianCropPatchDataset(val_rep, val_target, augment=False)
    test_dataset = AustrianCropPatchDataset(test_rep, test_target, augment=False)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 18  # 假设总共有 18 个类别，其中背景为 0（不参与loss和metric计算）
    model = UNet(in_channels=128, num_classes=num_classes, dropout=0.1)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters.")
    
    # 训练模型
    train_model(model, train_loader, val_loader, device, num_classes, epochs=200, lr=1e-3)
    
    # 加载最佳 checkpoint 进行测试
    checkpoint_path = os.path.join("checkpoints", "downstream", "best_austriancrop_patch_seg_ckpt.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}")
    else:
        print("No checkpoint found. Using current model for testing.")
    
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for rep, target in test_loader:
            rep = rep.to(device)
            target = target.to(device)
            output = model(rep)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            
    num_test = len(test_loader)
    avg_test_loss = test_loss / num_test
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    overall_acc, overall_f1, overall_iou = compute_segmentation_metrics(all_preds, all_targets, num_classes)
    print(f"Overall Test - Acc: {overall_acc:.4f}, F1: {overall_f1:.4f}, mIoU: {overall_iou:.4f}")
    
    per_class = compute_per_class_metrics(all_preds, all_targets, num_classes)
    print("Per-class metrics (excluding background, class 0):")
    for cls in sorted(per_class.keys()):
        m = per_class[cls]
        print(f"  Class {cls}: Acc: {m['acc']:.4f}, F1: {m['f1']:.4f}, IoU: {m['iou']:.4f}")

if __name__ == "__main__":
    main()
