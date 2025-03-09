#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
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
class PastisPatchDataset(Dataset):
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
        
        target = np.load(self.target_files[idx])[0]
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
    def __init__(self, in_channels=128, num_classes=20, features=[128, 256, 512], dropout=0.1):
        """
        :param in_channels: 输入通道数（这里为 128）
        :param num_classes: 输出类别数（本例为20，其中标签19为空标签，将被ignore；标签0为背景）
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
def get_file_lists(repr_root_dir, label_root_dir):
    """
    从指定的根目录中查找 representation 与 label 文件
    representation 文件命名格式为 representation_*.npy
    label 文件命名格式为 TARGET_*.npy
    """
    rep_files = sorted(glob.glob(os.path.join(repr_root_dir, "representation_*.npy")))
    target_files = sorted(glob.glob(os.path.join(label_root_dir, "TARGET_*.npy")))
    return rep_files, target_files

def split_data_by_metadata(rep_files, target_files, metadata_path):
    """
    根据 metadata.geojson 文件中每个 patch 的 Fold 字段进行数据集划分：
      - Fold 1,2,3 作为训练集
      - Fold 4 作为验证集
      - Fold 5 作为测试集
    文件名中的 patch id 与 metadata 中的 ID_PATCH 一致，如 representation_10000.npy 对应 ID_PATCH=10000
    """
    # 加载 metadata 文件
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    # 构建 patch id 到 Fold 的映射
    patch_fold = {}
    for feature in metadata["features"]:
        patch_id = str(feature["properties"]["ID_PATCH"])
        fold = feature["properties"]["Fold"]
        patch_fold[patch_id] = fold

    # 构建从 patch id 到文件路径的映射（对于 representation 和 TARGET 文件）
    rep_dict = {}
    for rep_file in rep_files:
        basename = os.path.basename(rep_file)  # e.g. representation_10000.npy
        patch_id = basename.replace("representation_", "").replace(".npy", "")
        rep_dict[patch_id] = rep_file

    target_dict = {}
    for target_file in target_files:
        basename = os.path.basename(target_file)  # e.g. TARGET_10000.npy
        patch_id = basename.replace("TARGET_", "").replace(".npy", "")
        target_dict[patch_id] = target_file

    # 根据 Fold 字段划分数据集
    train_rep, train_target = [], []
    val_rep, val_target = [], []
    test_rep, test_target = [], []

    for patch_id, fold in patch_fold.items():
        if patch_id not in rep_dict or patch_id not in target_dict:
            continue
        if fold in [1, 2, 3]:
            train_rep.append(rep_dict[patch_id])
            train_target.append(target_dict[patch_id])
        elif fold == 4:
            val_rep.append(rep_dict[patch_id])
            val_target.append(target_dict[patch_id])
        elif fold == 5:
            test_rep.append(rep_dict[patch_id])
            test_target.append(target_dict[patch_id])

    # 为了确保顺序一致，可按 patch id 排序
    train_rep = sorted(train_rep)
    train_target = sorted(train_target)
    val_rep = sorted(val_rep)
    val_target = sorted(val_target)
    test_rep = sorted(test_rep)
    test_target = sorted(test_target)

    return (train_rep, train_target), (val_rep, val_target), (test_rep, test_target)

####################################
# 4. 定义指标计算与损失函数（忽略空标签，空标签为19；背景标签0按加权处理）
####################################
def compute_segmentation_metrics(pred, target, num_classes):
    """
    计算整体分割指标（忽略空标签）：accuracy, F1, mIoU
    :param pred: 预测标签，形状 (B, H, W)
    :param target: 真实标签，形状 (B, H, W)
    :param num_classes: 输出类别数
    :return: accuracy, mean F1, mean IoU
    """
    # 忽略空标签（标签为19）
    mask = target != 19
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    pred_masked = pred[mask]
    target_masked = target[mask]
    
    acc = (pred_masked == target_masked).float().mean().item()
    
    f1s = []
    ious = []
    for cls in range(num_classes):
        if cls == 19:  # 空标签忽略
            continue
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
    计算每个类别（排除空标签 19）的指标：accuracy, F1, IoU
    :param all_preds: tensor, 形状 (N, H, W)
    :param all_targets: tensor, 形状 (N, H, W)
    :param num_classes: 类别数
    :return: dict, key为类别编号，value为 {'acc':..., 'f1':..., 'iou':...}
    """
    metrics = {}
    # 将预测和真实标签展平
    preds = all_preds.view(-1)
    targets = all_targets.view(-1)
    for cls in range(num_classes):
        if cls == 19:  # 忽略空标签
            continue
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
# 新增：WeightedCrossEntropy 损失函数
# 对于空标签（19）直接 ignore，对于背景标签（0）及其他类别使用权重：权重根据 distribution = [0.00000, 0.25675, 0.06733, 0.10767, 0.02269,
# 0.01451, 0.00745, 0.01111, 0.08730, 0.00715, 0.00991, 0.01398, 0.02149, 0.00452, 0.02604, 0.00994, 0.02460, 0.00696, 0.00580, 0.29476]
# 注意：若 distribution 中某一项为 0（如背景标签 0），则将其权重设为1，避免除0异常
####################################
class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list):
        """
        :param ignore_index: 要忽略的标签（空标签，本例为19）
        :param distribution: 各类别的分布概率列表，长度应为 num_classes
        """
        super(WeightedCrossEntropy, self).__init__()
        weights = []
        for i, w in enumerate(distribution):
            if i == ignore_index:
                # 对空标签，权重任意（不会参与loss计算）
                weights.append(0.0)
            else:
                # 如果 distribution 为0，则设置为1以避免除零
                weights.append(1.0 / w if w > 0 else 1.0)
        loss_weights = torch.Tensor(weights).to("cuda")
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=loss_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the weighted cross-entropy loss
        return self.loss(logits, target)

####################################
# 5. 训练、验证与测试过程（含 checkpoint 保存与加载）
####################################
def train_model(model, train_loader, val_loader, device, num_classes, epochs=500, lr=1e-3):
    # 定义 distribution（长度20，对应类别 0~19，其中标签19为空标签，将被忽略）
    distribution = [
        0.00000,
        0.25675,
        0.06733,
        0.10767,
        0.02269,
        0.01451,
        0.00745,
        0.01111,
        0.08730,
        0.00715,
        0.00991,
        0.01398,
        0.02149,
        0.00452,
        0.02604,
        0.00994,
        0.02460,
        0.00696,
        0.00580,
        0.29476
    ]
    # 使用自定义的 weighted cross entropy loss，忽略标签 19（空标签）
    criterion = WeightedCrossEntropy(ignore_index=19, distribution=distribution)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)
    
    # 学习率调度器：当验证 loss 不下降时降低 lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    checkpoint_dir = os.path.join("checkpoints", "downstream")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_pastis_patch_seg_ckpt.pth")
    
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
    rep_root_dir = "/scratch/zf281/pastis/representation/dual_transformer_120tiles_20epochs"
    label_root_dir = "/scratch/zf281/pastis/data/ANNOTATIONS"
    metadata_path = "/scratch/zf281/pastis/data/metadata.geojson"
    rep_files, target_files = get_file_lists(rep_root_dir, label_root_dir)
    
    # 根据 metadata 中的 Fold 字段进行数据集划分：
    # Fold 1,2,3 作为训练集；Fold 4 作为验证集；Fold 5 作为测试集
    (train_rep, train_target), (val_rep, val_target), (test_rep, test_target) = split_data_by_metadata(rep_files, target_files, metadata_path)
    
    # 训练集可开启数据增强
    train_dataset = PastisPatchDataset(train_rep, train_target, augment=True)
    # 验证、测试集不做数据增强
    val_dataset = PastisPatchDataset(val_rep, val_target, augment=False)
    test_dataset = PastisPatchDataset(test_rep, test_target, augment=False)
    # 打印数据集数目
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 20  # 总共20个类别，其中标签19为空标签（ignore），标签0为背景（使用加权处理）
    model = UNet(in_channels=128, num_classes=num_classes, dropout=0.1)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters.")
    
    # 训练模型
    train_model(model, train_loader, val_loader, device, num_classes, epochs=100, lr=1e-3)
    
    # 加载最佳 checkpoint 进行测试
    checkpoint_path = os.path.join("checkpoints", "downstream", "best_pastis_patch_seg_ckpt.pth")
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
    # 测试时同样使用自定义的 weighted loss（ignore_index=19）
    distribution = [
        0.00000,
        0.25675,
        0.06733,
        0.10767,
        0.02269,
        0.01451,
        0.00745,
        0.01111,
        0.08730,
        0.00715,
        0.00991,
        0.01398,
        0.02149,
        0.00452,
        0.02604,
        0.00994,
        0.02460,
        0.00696,
        0.00580,
        0.29476
    ]
    criterion = WeightedCrossEntropy(ignore_index=19, distribution=distribution)
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
    print("Per-class metrics (excluding empty label, class 19):")
    for cls in sorted(per_class.keys()):
        m = per_class[cls]
        print(f"  Class {cls}: Acc: {m['acc']:.4f}, F1: {m['f1']:.4f}, IoU: {m['iou']:.4f}")

if __name__ == "__main__":
    main()
