#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import re
import argparse

####################################
# 1. 数据集类（基于代码A的路径结构和标签映射逻辑，使用代码B的增强逻辑）
####################################
class PatchSegmentationDataset(Dataset):
    def __init__(self, patch_rep_files, patch_label_files, label_map, ignore_label=0, ignore_index=-100, augment=False):
        """
        Args:
            patch_rep_files (list): List of file paths to representation patches (P, P, C)
            patch_label_files (list): List of file paths to label patches (P, P)
            label_map (dict): Dictionary to map original label values to 0-N indices
            ignore_label (int): Label value to ignore (default: 0 for background)
            ignore_index (int): PyTorch standard ignore index for loss calculation
            augment (bool): Whether to apply data augmentation
        """
        self.patch_rep_files = []
        self.patch_label_files = []
        self.label_map = label_map
        self.ignore_label = ignore_label
        self.ignore_index = ignore_index
        self.augment = augment

        logging.info(f"Processing {len(patch_rep_files)} patch files for segmentation...")
        
        for rep_file, label_file in tqdm(zip(patch_rep_files, patch_label_files), total=len(patch_rep_files)):
            try:
                label_patch_original = np.load(label_file)  # Shape (P, P)
                
                # 处理 NaN 值：将 NaN 替换为 ignore_label (0)
                if np.any(np.isnan(label_patch_original)):
                    logging.warning(f"Found NaN values in label file {label_file}, replacing with {ignore_label}")
                    label_patch_original = np.nan_to_num(label_patch_original, nan=ignore_label)
                
                # Check if patch has any valid labels (non-ignore_label)
                valid_labels_in_patch = label_patch_original[label_patch_original != ignore_label]
                
                if valid_labels_in_patch.size == 0:
                    continue  # Skip patches with no valid labels
                
                self.patch_rep_files.append(rep_file)
                self.patch_label_files.append(label_file)
                
            except Exception as e:
                logging.error(f"Error processing patch {rep_file} or {label_file}: {e}")
                continue
                
        logging.info(f"Kept {len(self.patch_rep_files)} patches with valid labels for segmentation.")

    def __len__(self):
        return len(self.patch_rep_files)
    
    def get_patch_id_from_filename(self, filename):
        """从文件名中提取patch ID"""
        basename = os.path.basename(filename)
        match = re.search(r'patch_(\d+)\.npy', basename)
        if match:
            return int(match.group(1))
        return None

    def __getitem__(self, idx):
        rep_patch = np.load(self.patch_rep_files[idx])  # Shape (P, P, C)
        label_patch = np.load(self.patch_label_files[idx])  # Shape (P, P)
        
        # 处理 representation 中的 NaN 值：替换为 0
        if np.any(np.isnan(rep_patch)):
            logging.debug(f"Found NaN values in representation patch {self.patch_rep_files[idx]}, replacing with 0")
            rep_patch = np.nan_to_num(rep_patch, nan=0.0)
        
        # 处理 label 中的 NaN 值：替换为 ignore_label (0)
        if np.any(np.isnan(label_patch)):
            logging.debug(f"Found NaN values in label patch {self.patch_label_files[idx]}, replacing with {self.ignore_label}")
            label_patch = np.nan_to_num(label_patch, nan=self.ignore_label)
        
        # Convert representation to (C, P, P) for PyTorch CNNs
        rep_patch_tensor = torch.tensor(rep_patch, dtype=torch.float32).permute(2, 0, 1)
        
        # Map labels and handle ignore_label
        label_patch_mapped = np.full_like(label_patch, self.ignore_index, dtype=np.int64)
        
        for original_label, mapped_label in self.label_map.items():
            mask = (label_patch == original_label)
            label_patch_mapped[mask] = mapped_label
        
        # Background (ignore_label) remains as ignore_index for loss calculation
        label_patch_tensor = torch.tensor(label_patch_mapped, dtype=torch.long)
        
        # Data augmentation: random horizontal and vertical flips
        if self.augment:
            if random.random() > 0.5:
                rep_patch_tensor = torch.flip(rep_patch_tensor, dims=[2])  # 水平翻转：对W维度翻转
                label_patch_tensor = torch.flip(label_patch_tensor, dims=[1])
            if random.random() > 0.5:
                rep_patch_tensor = torch.flip(rep_patch_tensor, dims=[1])  # 垂直翻转：对H维度翻转
                label_patch_tensor = torch.flip(label_patch_tensor, dims=[0])
        
        return rep_patch_tensor, label_patch_tensor

####################################
# 2. UNet模型（基于代码B，添加Dropout防止过拟合）
####################################
class DoubleConv(nn.Module):
    """Two 3x3 convolutions + BatchNorm + ReLU + Dropout"""
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
        :param in_channels: Input channels
        :param num_classes: Output channels (num_classes for segmentation)
        :param features: Number of features in each layer of the encoder
        :param dropout: Dropout rate
        """
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Build encoder (Downsampling)
        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in_channels, feature, dropout_rate=dropout))
            curr_in_channels = feature
        
        # Bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_rate=dropout)
        
        # Build decoder (Upsampling)
        for feature in reversed(features):
            # Use transposed convolution for upsampling
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            # Channel number after concatenation is 2*feature, then reduced back to feature through double convolution
            self.ups.append(DoubleConv(feature*2, feature, dropout_rate=dropout))
        
        # Final 1x1 convolution for segmentation
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        # Encoder part: save features of each stage for skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse to correspond with the decoder
        
        # Decoder part: upsampling, concatenation, double convolution
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            # If dimensions do not match, adjust using interpolation
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
            
        # Return segmentation map: (B, num_classes, H, W)
        return self.final_conv(x)

####################################
# 3. 指标计算函数（基于代码B，但适配ignore_index）
####################################
def compute_segmentation_metrics(pred, target, num_classes, ignore_index=-100):
    """
    计算整体分割指标（忽略背景像素）：accuracy, balanced F1 (macro F1), mIoU
    :param pred: 预测标签，形状 (B, H, W)
    :param target: 真实标签，形状 (B, H, W)
    :param num_classes: 输出类别数
    :param ignore_index: 要忽略的索引值
    :return: accuracy, balanced F1 (macro F1), mean IoU
    """
    # 忽略背景（标签为ignore_index）
    mask = target != ignore_index
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    pred_masked = pred[mask]
    target_masked = target[mask]
    
    acc = (pred_masked == target_masked).float().mean().item()
    
    # 计算每个类别的F1和IoU，包括那些在当前batch中可能不存在的类别
    f1_scores = np.zeros(num_classes)
    iou_scores = np.zeros(num_classes)
    class_present = np.zeros(num_classes, dtype=bool)
    
    for cls in range(num_classes):
        tp = ((pred_masked == cls) & (target_masked == cls)).sum().item()
        fp = ((pred_masked == cls) & (target_masked != cls)).sum().item()
        fn = ((pred_masked != cls) & (target_masked == cls)).sum().item()
        
        # 检查这个类别是否在真实标签中出现
        if (tp + fn) > 0:  # 类别在真实标签中存在
            class_present[cls] = True
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_scores[cls] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            iou_scores[cls] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # 计算balanced F1 (macro F1): 只对真实标签中存在的类别求平均
    if class_present.sum() > 0:
        balanced_f1 = f1_scores[class_present].mean()
        mean_iou = iou_scores[class_present].mean()
    else:
        balanced_f1 = 0.0
        mean_iou = 0.0
    
    return acc, balanced_f1, mean_iou

def compute_per_class_metrics(all_preds, all_targets, num_classes, label_map, ignore_index=-100):
    """
    计算每个类别的指标：accuracy, F1, IoU
    :param all_preds: tensor, 形状 (N, H, W)
    :param all_targets: tensor, 形状 (N, H, W)
    :param num_classes: 类别数
    :param label_map: 原始标签到索引的映射
    :param ignore_index: 要忽略的索引值
    :return: dict, key为原始类别编号，value为 {'acc':..., 'f1':..., 'iou':...}
    """
    metrics = {}
    # 将预测和真实标签展平
    preds = all_preds.view(-1)
    targets = all_targets.view(-1)
    
    # 创建反向映射：索引到原始标签
    idx_to_original = {v: k for k, v in label_map.items()}
    
    # 过滤掉ignore_index
    valid_mask = targets != ignore_index
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    for mapped_idx in range(num_classes):
        original_label = idx_to_original.get(mapped_idx, mapped_idx)
        
        # 针对类别 mapped_idx，二值化
        pred_cls = (preds == mapped_idx).long()
        target_cls = (targets == mapped_idx).long()
        
        # 计算各项指标
        tp = ((pred_cls == 1) & (target_cls == 1)).sum().item()
        fp = ((pred_cls == 1) & (target_cls == 0)).sum().item()
        tn = ((pred_cls == 0) & (target_cls == 0)).sum().item()
        fn = ((pred_cls == 0) & (target_cls == 1)).sum().item()
        total = tp + tn + fp + fn
        
        if total == 0:
            continue
            
        acc = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        metrics[original_label] = {"acc": acc, "f1": f1, "iou": iou}
    
    return metrics

####################################
# 4. 颜色映射函数（与clip_austriancrop_representation_map.py保持一致）
####################################
def create_label_to_color_mapping(unique_labels):
    """
    创建标签到颜色的映射，与clip_austriancrop_representation_map.py保持一致
    """
    unique_labels = np.sort(unique_labels)  # 确保一致的颜色映射
    num_distinct_colors_needed = len(unique_labels)
    
    if num_distinct_colors_needed > 20:
        logging.warning(f"Number of unique labels ({num_distinct_colors_needed}) exceeds tab20 colormap size (20). Colors may repeat.")
    
    cmap = plt.get_cmap("tab20", max(20, num_distinct_colors_needed))
    label_to_color = {}
    
    for idx, label_val in enumerate(unique_labels):
        # 归一化索引
        norm_idx = idx / (num_distinct_colors_needed - 1) if num_distinct_colors_needed > 1 else 0.0
        rgba = cmap(norm_idx)
        rgb = tuple(int(255 * x) for x in rgba[:3])
        label_to_color[label_val] = rgb
    
    return label_to_color

def visualize_inference_result(pred_patch, label_map, output_path, patch_size):
    """
    可视化推理结果
    :param pred_patch: 预测的标签patch，形状 (P, P)，包含mapped indices
    :param label_map: 原始标签到索引的映射
    :param output_path: 输出路径
    :param patch_size: patch大小
    """
    # 创建反向映射：索引到原始标签
    idx_to_original = {v: k for k, v in label_map.items()}
    
    # 将预测的indices转换回原始标签值
    pred_original = np.zeros_like(pred_patch)
    for mapped_idx, original_label in idx_to_original.items():
        mask = (pred_patch == mapped_idx)
        pred_original[mask] = original_label
    
    # 获取所有可能的标签（包括0）
    all_possible_labels = list(idx_to_original.values()) + [0]  # 添加背景标签0
    unique_labels = sorted(list(set(all_possible_labels)))
    
    # 创建颜色映射
    label_to_color = create_label_to_color_mapping(unique_labels)
    
    # 创建可视化图像
    vis_image = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    for label_val, color in label_to_color.items():
        mask = (pred_original == label_val)
        vis_image[mask] = color
    
    # 保存图像
    plt.imsave(output_path, vis_image)

####################################
# 5. 主训练函数
####################################
def train_model(model, train_loader, val_loader, device, num_classes, epochs=200, lr=1e-3, timestamp="", label_map=None):
    """训练模型，包含学习率调度和最佳模型保存"""
    # 使用 weight_decay 来增加 L2 正则化
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 使用标准的ignore_index
    model.to(device)
    
    # 学习率调度器：当验证 loss 不下降时降低 lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    best_val_f1 = 0.0
    best_val_miou = 0.0
    best_epoch = 0
    checkpoint_dir = os.path.join("checkpoints", f"combined_segmentation_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        train_acc_sum = 0.0
        train_f1_sum = 0.0
        train_miou_sum = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for batch_idx, (rep, target) in enumerate(progress_bar):
            rep = rep.to(device)       # (B, C, H, W)
            target = target.to(device) # (B, H, W)
            
            optimizer.zero_grad()
            output = model(rep)        # (B, num_classes, H, W)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            pred = output.argmax(dim=1)  # (B, H, W)
            acc, f1, miou = compute_segmentation_metrics(pred, target, num_classes, ignore_index=-100)
            train_acc_sum += acc
            train_f1_sum += f1
            train_miou_sum += miou
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{acc:.4f}",
                "Bal F1": f"{f1:.4f}",
                "mIoU": f"{miou:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
            })
        
        # 计算训练平均指标
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = train_acc_sum / len(train_loader)
        avg_train_f1 = train_f1_sum / len(train_loader)
        avg_train_miou = train_miou_sum / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        val_miou = 0.0
        with torch.no_grad():
            for rep, target in tqdm(val_loader, desc=f"Validating Epoch {epoch}", leave=False):
                rep = rep.to(device)
                target = target.to(device)
                output = model(rep)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                acc, f1, miou = compute_segmentation_metrics(pred, target, num_classes, ignore_index=-100)
                val_acc += acc
                val_f1 += f1
                val_miou += miou
                
        num_val = len(val_loader)
        avg_val_loss = val_loss / num_val
        avg_val_acc = val_acc / num_val
        avg_val_f1 = val_f1 / num_val
        avg_val_miou = val_miou / num_val
        
        logging.info(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
                    f"Train Balanced F1: {avg_train_f1:.4f} | Train mIoU: {avg_train_miou:.4f}")
        logging.info(f"Epoch {epoch}/{epochs} - Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | "
                    f"Val Balanced F1: {avg_val_f1:.4f} | Val mIoU: {avg_val_miou:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # 保存验证 F1 最好的 checkpoint
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_val_miou = avg_val_miou
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_f1': avg_val_f1,
                'val_miou': avg_val_miou,
                'num_classes': num_classes,
                'label_map': label_map
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"🎉 Saved new best checkpoint at epoch {epoch} with Val Balanced F1: {avg_val_f1:.4f}, mIoU: {avg_val_miou:.4f}")
    
    logging.info(f"Training completed. Best Val Balanced F1: {best_val_f1:.4f}, mIoU: {best_val_miou:.4f} at Epoch {best_epoch}")
    return checkpoint_path

####################################
# 6. 主函数
####################################
def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Train downstream model for Austrian crop segmentation with different patch sizes')
    parser.add_argument('--patchsize', type=int, choices=[32, 64, 128], required=True,
                        help='Patch size for training (32, 64, or 128)')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Ratio of training data to use (default: 1.0, use all training data)')
    args = parser.parse_args()
    
    patchsize = args.patchsize
    train_ratio = args.train_ratio
    
    # 验证train_ratio参数
    if not (0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    logging.info(f"Training with patch size: {patchsize}x{patchsize}")
    logging.info(f"Using {train_ratio*100:.1f}% of training data")

    # -------------------------
    # 根据patchsize参数设置数据路径
    # -------------------------
    base_patch_dir = f"data/downstream/austrian_crop/patch_{patchsize}_{patchsize}"
    # base_patch_dir = f"data/downstream/austrian_crop/alphaearth/patch_{patchsize}_{patchsize}"
    # base_patch_dir = f"data/downstream/austrian_crop/presto/patch_{patchsize}_{patchsize}"
    
    if not os.path.exists(base_patch_dir):
        raise ValueError(f"Data directory {base_patch_dir} does not exist. Please check the path.")
    
    train_rep_files = sorted([os.path.join(base_patch_dir, "train", f) for f in os.listdir(os.path.join(base_patch_dir, "train")) if f.startswith("patch_") and f.endswith(".npy")])
    train_label_files = sorted([os.path.join(base_patch_dir, "train", f) for f in os.listdir(os.path.join(base_patch_dir, "train")) if f.startswith("label_") and f.endswith(".npy")])

    # 根据train_ratio采样训练数据
    if train_ratio < 1.0:
        total_train_samples = len(train_rep_files)
        num_samples = int(total_train_samples * train_ratio)
        
        # 创建索引列表并随机采样
        indices = list(range(total_train_samples))
        np.random.shuffle(indices)
        selected_indices = sorted(indices[:num_samples])
        
        # 根据选中的索引筛选文件
        train_rep_files = [train_rep_files[i] for i in selected_indices]
        train_label_files = [train_label_files[i] for i in selected_indices]
        
        logging.info(f"Sampled {num_samples} training samples from {total_train_samples} total samples")
    else:
        logging.info(f"Using all {len(train_rep_files)} training samples")

    test_rep_files = sorted([os.path.join(base_patch_dir, "test", f) for f in os.listdir(os.path.join(base_patch_dir, "test")) if f.startswith("patch_") and f.endswith(".npy")])
    test_label_files = sorted([os.path.join(base_patch_dir, "test", f) for f in os.listdir(os.path.join(base_patch_dir, "test")) if f.startswith("label_") and f.endswith(".npy")])

    # 检查是否有val目录，如果有则使用，否则从test中分割
    val_dir = os.path.join(base_patch_dir, "val")
    if os.path.exists(val_dir):
        val_rep_files = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.startswith("patch_") and f.endswith(".npy")])
        val_label_files = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.startswith("label_") and f.endswith(".npy")])
        logging.info("Using separate validation directory")
    else:
        # 从测试集中分出验证集
        if len(test_rep_files) > 100:
            val_split_idx = int(len(test_rep_files) * 1 / 7)
            val_rep_files = test_rep_files[:val_split_idx]
            val_label_files = test_label_files[:val_split_idx]
            test_rep_files = test_rep_files[val_split_idx:]
            test_label_files = test_label_files[val_split_idx:]
            logging.info("Split validation set from test set")
        else:
            logging.warning("Test set is too small for a validation split, using test set for validation metrics.")
            val_rep_files = test_rep_files
            val_label_files = test_label_files

    # 打印前几个test文件名以验证
    logging.info("\nFirst 10 test representation files:")
    for i, f in enumerate(test_rep_files[:10]):
        logging.info(f"  {i}: {os.path.basename(f)}")
    
    logging.info("\nFirst 10 test label files:")
    for i, f in enumerate(test_label_files[:10]):
        logging.info(f"  {i}: {os.path.basename(f)}")

    # -------------------------
    # 确定类别数和标签映射（使用代码A的逻辑）
    # -------------------------
    all_original_labels_in_patches = set()
    sample_label_files_for_meta = train_label_files[:min(100, len(train_label_files))]
    for f in sample_label_files_for_meta:
        patch_l = np.load(f)
        # 处理 NaN 值
        if np.any(np.isnan(patch_l)):
            patch_l = np.nan_to_num(patch_l, nan=0)
        all_original_labels_in_patches.update(np.unique(patch_l))

    unique_original_nonzero_labels = sorted([l for l in all_original_labels_in_patches if l != 0])
    if not unique_original_nonzero_labels:
        raise ValueError("No non-zero labels found in sample patches. Cannot determine classes.")

    # 创建标签映射：原始非零标签到0-N索引
    label_map_for_dataset = {label: i for i, label in enumerate(unique_original_nonzero_labels)}
    num_classes = len(unique_original_nonzero_labels)

    logging.info(f"Original non-zero labels found: {unique_original_nonzero_labels}")
    logging.info(f"Label map for segmentation: {label_map_for_dataset}")
    logging.info(f"Number of classes for pixel-wise segmentation: {num_classes}")
    
    # -------------------------
    # 检查数据维度并动态获取通道数
    # -------------------------
    if not train_rep_files:
        raise ValueError("No training representation files found. Check paths and patch generation.")
    
    sample_rep_patch = np.load(train_rep_files[0])
    # 处理样本中的 NaN 值
    if np.any(np.isnan(sample_rep_patch)):
        sample_rep_patch = np.nan_to_num(sample_rep_patch, nan=0.0)
    patch_p, _, patch_c = sample_rep_patch.shape  # Assuming (P,P,C)
    logging.info(f"Detected patch properties: Size P={patch_p}, Channels C={patch_c}")
    
    # 验证patch size与参数一致
    if patch_p != patchsize:
        raise ValueError(f"Detected patch size {patch_p} does not match specified patchsize {patchsize}")

    # -------------------------
    # 创建数据集和数据加载器
    # -------------------------
    # 训练集开启数据增强
    train_dataset = PatchSegmentationDataset(train_rep_files, train_label_files, 
                                           label_map=label_map_for_dataset, 
                                           ignore_label=0, 
                                           ignore_index=-100,
                                           augment=True)
    # 验证和测试集不开启数据增强
    val_dataset = PatchSegmentationDataset(val_rep_files, val_label_files, 
                                         label_map=label_map_for_dataset,
                                         ignore_label=0,
                                         ignore_index=-100,
                                         augment=False)
    test_dataset = PatchSegmentationDataset(test_rep_files, test_label_files, 
                                          label_map=label_map_for_dataset,
                                          ignore_label=0,
                                          ignore_index=-100,
                                          augment=False)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check patch processing and label mapping.")

    batch_size = 32  # 可根据显存调整
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # 打印test_dataset的前几个文件以验证
    logging.info("\nFirst 10 files in test_dataset after filtering:")
    for i in range(min(10, len(test_dataset))):
        logging.info(f"  {i}: {os.path.basename(test_dataset.patch_rep_files[i])}")

    # -------------------------
    # 创建模型
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    model = UNet(in_channels=patch_c, num_classes=num_classes, features=[128, 256, 512], dropout=0.1)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model has {num_params:,} parameters")

    # -------------------------
    # 训练模型
    # -------------------------
    checkpoint_path = train_model(model, train_loader, val_loader, device, num_classes, 
                                 epochs=100, lr=1e-3, timestamp=timestamp, label_map=label_map_for_dataset)
    
    # -------------------------
    # 加载最佳checkpoint进行测试
    # -------------------------
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with Val Balanced F1: {checkpoint.get('val_f1', 0):.4f}")
    else:
        logging.warning("No checkpoint found. Using current model for testing.")
    
    # -------------------------
    # 最终测试评估
    # -------------------------
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        for rep, target in tqdm(test_loader, desc="Final Test Evaluation", leave=False):
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
    logging.info(f"Final Test Loss: {avg_test_loss:.4f}")
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    overall_acc, overall_f1, overall_iou = compute_segmentation_metrics(all_preds, all_targets, num_classes, ignore_index=-100)
    logging.info(f"FINAL TEST RESULTS:")
    logging.info(f"Overall Test - Accuracy: {overall_acc:.4f}, Balanced F1: {overall_f1:.4f}, mIoU: {overall_iou:.4f}")
    print(f"\nFINAL TEST RESULTS:")
    print(f"Overall Test - Accuracy: {overall_acc:.4f}, Balanced F1: {overall_f1:.4f}, mIoU: {overall_iou:.4f}")
    
    # 计算每个类别的详细指标
    per_class = compute_per_class_metrics(all_preds, all_targets, num_classes, label_map_for_dataset, ignore_index=-100)
    logging.info("\nPer-class metrics (excluding background, class 0):")
    print("\nPer-class metrics (excluding background, class 0):")
    for original_label in sorted(per_class.keys()):
        m = per_class[original_label]
        logging.info(f"  Class {original_label}: Acc: {m['acc']:.4f}, F1: {m['f1']:.4f}, IoU: {m['iou']:.4f}")
        print(f"  Class {original_label}: Acc: {m['acc']:.4f}, F1: {m['f1']:.4f}, IoU: {m['iou']:.4f}")
    
    # -------------------------
    # 对前5个test patch进行推理并可视化（直接使用文件名中的ID）
    # -------------------------
    logging.info("\n" + "="*80)
    logging.info("INFERENCE VISUALIZATION FOR FIRST 5 TEST PATCHES")
    logging.info("="*80)
    
    # 获取test目录中的所有patch文件并按文件名中的数字排序
    test_dir = os.path.join(base_patch_dir, "test")
    all_patch_files = [f for f in os.listdir(test_dir) if f.startswith("patch_") and f.endswith(".npy")]
    
    # 提取文件名中的数字并排序
    def extract_patch_number(filename):
        match = re.search(r'patch_(\d+)\.npy', filename)
        return int(match.group(1)) if match else float('inf')
    
    all_patch_files.sort(key=extract_patch_number)
    
    # 选择前5个patch进行可视化
    patches_to_visualize = all_patch_files[:5]
    logging.info(f"\nPatches selected for visualization: {patches_to_visualize}")
    
    for patch_filename in patches_to_visualize:
        # 获取patch ID
        patch_id = extract_patch_number(patch_filename)
        
        # 构建完整路径
        rep_path = os.path.join(test_dir, patch_filename)
        label_path = os.path.join(test_dir, f"label_{patch_id}.npy")
        
        if not os.path.exists(rep_path) or not os.path.exists(label_path):
            logging.warning(f"Cannot find files for patch {patch_id}: {rep_path} or {label_path}")
            continue
        
        logging.info(f"\nProcessing patch {patch_id}:")
        logging.info(f"  Representation file: {rep_path}")
        logging.info(f"  Label file: {label_path}")
        
        # 查找这个文件在test_dataset中的索引
        dataset_idx = None
        for idx, file_path in enumerate(test_dataset.patch_rep_files):
            if os.path.basename(file_path) == patch_filename:
                dataset_idx = idx
                break
        
        if dataset_idx is None:
            logging.warning(f"  Patch {patch_id} not found in test_dataset (might have been filtered out due to no valid labels)")
            continue
        
        logging.info(f"  Found at dataset index: {dataset_idx}")
        
        # 获取数据
        rep_tensor, label_tensor = test_dataset[dataset_idx]
        
        # 准备输入
        rep_tensor = rep_tensor.unsqueeze(0).to(device)  # 添加batch维度
        
        # 推理
        with torch.no_grad():
            output = model(rep_tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # 移除batch维度，转为numpy
        
        # 生成输出路径
        output_path = os.path.join(test_dir, f"patch_{patch_id}_inference.png")
        
        # 可视化并保存
        visualize_inference_result(pred, label_map_for_dataset, output_path, patch_p)
        
        logging.info(f"  ✓ Saved inference visualization to {output_path}")
        
        # 加载原始标签进行对比（用于调试）
        original_label = np.load(label_path)
        if np.any(np.isnan(original_label)):
            original_label = np.nan_to_num(original_label, nan=0)
        unique_labels_in_original = np.unique(original_label)
        logging.info(f"  Original label unique values: {unique_labels_in_original}")
        
        # 反向映射预测结果
        idx_to_original = {v: k for k, v in label_map_for_dataset.items()}
        pred_original = np.zeros_like(pred)
        for mapped_idx, original_label_val in idx_to_original.items():
            mask = (pred == mapped_idx)
            pred_original[mask] = original_label_val
        unique_labels_in_pred = np.unique(pred_original)
        logging.info(f"  Predicted label unique values: {unique_labels_in_pred}")
    
    logging.info("\n" + "="*80)
    logging.info("✅ All processing complete!")
    logging.info("="*80)

if __name__ == "__main__":
    main()