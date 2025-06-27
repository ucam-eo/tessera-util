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
import torch.nn.functional as F
import math
import argparse
from typing import List, Optional, Tuple, Union, Dict
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

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
        
        # 增强的数据增强策略
        if self.augment:
            # 水平和垂直翻转
            if random.random() > 0.5:
                rep = torch.flip(rep, dims=[2])  # 水平翻转：对W维度翻转
                target = torch.flip(target, dims=[1])
            if random.random() > 0.5:
                rep = torch.flip(rep, dims=[1])  # 垂直翻转：对H维度翻转
                target = torch.flip(target, dims=[0])
                
            # 随机旋转（90度的倍数）
            k = random.randint(0, 3)  # 0: 0°, 1: 90°, 2: 180°, 3: 270°
            if k > 0:
                rep = torch.rot90(rep, k, dims=[1, 2])
                target = torch.rot90(target, k, dims=[0, 1])
        
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
            
        x = self.final_conv(x)
        return x, None  # 返回主输出和None作为辅助输出，以匹配新的模型接口

####################################
# 3. 深度可分离卷积模块
####################################
class DepthwiseSeparableConv(nn.Module):
    """轻量级深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, activation=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""
    def __init__(self, channels, reduction_ratio=8):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        identity = x
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return identity * x

class SpatialAttention(nn.Module):
    """空间注意力机制：学习空间位置的重要性"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(out)
        return attention * x

class EfficientChannelAttention(nn.Module):
    """高效通道注意力模块，使用动态卷积核大小，降低计算量"""
    def __init__(self, in_channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        # 根据通道数自适应计算卷积核大小
        t = int(abs((math.log2(in_channels) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 输入: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """通道-空间注意力模块，整合EfficientChannelAttention和SpatialAttention"""
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_att = EfficientChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_att(x)  # 先应用通道注意力
        x = self.spatial_att(x)  # 再应用空间注意力
        return x

####################################
# 4. DepthwiseUNet模型 
####################################
class DepthwiseUNet(nn.Module):
    """使用深度可分离卷积的UNet"""
    def __init__(self, in_channels=128, num_classes=20, 
                 features=[64, 128, 256, 512], 
                 use_channel_attention=True,
                 channel_reduction=16,
                 use_residual=True,
                 dropout_rate=0.1):
        """
        :param in_channels: 输入通道数
        :param num_classes: 输出类别数
        :param features: 每层的特征数列表
        :param use_channel_attention: 是否使用通道注意力
        :param channel_reduction: 通道注意力中的通道减少比例
        :param use_residual: 是否使用残差连接
        :param dropout_rate: Dropout比率
        """
        super(DepthwiseUNet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 编码器部分
        in_channels_temp = in_channels
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    DepthwiseSeparableConv(in_channels_temp, feature),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    DepthwiseSeparableConv(feature, feature),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
                )
            )
            in_channels_temp = feature
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(features[-1], features[-1]*2),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(features[-1]*2, features[-1]*2),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
        # 解码器部分
        for idx, feature in enumerate(reversed(features)):
            # 上采样
            self.ups.append(
                nn.ConvTranspose2d(
                    features[-idx-1]*2 if idx == 0 else features[-idx], 
                    feature, 
                    kernel_size=2, 
                    stride=2
                )
            )
            
            # 卷积块
            self.ups.append(
                nn.Sequential(
                    DepthwiseSeparableConv(feature*2, feature),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    DepthwiseSeparableConv(feature, feature),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 and idx < len(features)-1 else nn.Identity()
                )
            )
        
        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
        # 辅助分类头（从第三个编码器阶段）
        self.aux_head = nn.Sequential(
            nn.Conv2d(features[2], num_classes, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[2:]  # 保存输入尺寸
        
        # 存储跳跃连接
        skip_connections = []
        
        # 编码器路径
        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
            # 在第三个编码器阶段获取辅助输出
            if i == 2:
                aux_features = skip_connections[-1]
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # 解码器路径
        skip_connections = skip_connections[::-1]  # 反转跳跃连接列表
        
        for idx in range(0, len(self.ups), 2):
            # 上采样
            x = self.ups[idx](x)
            
            # 跳跃连接
            skip = skip_connections[idx // 2]
            
            # 尺寸不匹配时进行处理
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            # 连接
            x = torch.cat((skip, x), dim=1)
            
            # 卷积块
            x = self.ups[idx + 1](x)
        
        # 最终输出
        output = self.final_conv(x)
        
        # 确保输出与输入尺寸相同
        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        # 辅助输出
        aux_output = self.aux_head(aux_features)
        
        # 确保辅助输出与输入尺寸相同
        if aux_output.shape[2:] != input_size:
            aux_output = F.interpolate(aux_output, size=input_size, mode='bilinear', align_corners=True)
        
        return output, aux_output

####################################
# 5. LightweightSegNet模型
####################################
class LightweightSegNet(nn.Module):
    """轻量级分割网络，参数量小于5M"""
    def __init__(self, input_channels=128, num_classes=20, dropout_rate=0.2, width_multiplier=0.25):
        super(LightweightSegNet, self).__init__()
        
        # 设置通道数，使用width_multiplier控制模型大小
        base_channels = int(32 * width_multiplier)
        
        # 减少输入通道数，这是参数减少的关键
        reduction_channels = int(64 * width_multiplier)
        
        # 初始通道降维 - 从128降到较小的通道数
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(input_channels, reduction_channels, kernel_size=3, stride=1),
            nn.Conv2d(reduction_channels, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder路径 - 轻量级结构
        enc_channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        
        # Encoder Block 1 - 1/2分辨率
        self.enc1 = nn.Sequential(
            nn.Conv2d(base_channels, enc_channels[1], kernel_size=3, stride=2, padding=1, groups=base_channels),
            nn.BatchNorm2d(enc_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels[1], enc_channels[1], kernel_size=1),
            nn.BatchNorm2d(enc_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder Block 2 - 1/4分辨率
        self.enc2 = nn.Sequential(
            nn.Conv2d(enc_channels[1], enc_channels[1], kernel_size=3, stride=2, padding=1, groups=enc_channels[1]),
            nn.BatchNorm2d(enc_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels[1], enc_channels[2], kernel_size=1),
            nn.BatchNorm2d(enc_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder Block 3 - 1/8分辨率
        self.enc3 = nn.Sequential(
            nn.Conv2d(enc_channels[2], enc_channels[2], kernel_size=3, stride=2, padding=1, groups=enc_channels[2]),
            nn.BatchNorm2d(enc_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels[2], enc_channels[3], kernel_size=1),
            nn.BatchNorm2d(enc_channels[3]),
            nn.ReLU(inplace=True)
        )
        
        # 简化版金字塔池化 - 只使用两个尺度
        self.ppm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(enc_channels[3], enc_channels[3]//2, kernel_size=1),
            nn.BatchNorm2d(enc_channels[3]//2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder路径 - 简化的上采样模块
        self.dec3 = nn.Sequential(
            nn.Conv2d(enc_channels[3] + enc_channels[3]//2, enc_channels[2], kernel_size=1),
            nn.BatchNorm2d(enc_channels[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(enc_channels[2] + enc_channels[2], enc_channels[1], kernel_size=1),
            nn.BatchNorm2d(enc_channels[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(enc_channels[1] + enc_channels[1], enc_channels[0], kernel_size=1),
            nn.BatchNorm2d(enc_channels[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # 轻量级深层监督
        self.aux_head = nn.Sequential(
            nn.Conv2d(enc_channels[2], num_classes, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )
        
        # 最终分类层
        self.final_conv = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        input_size = x.shape[2:]  # Store original input size
        
        # Stem - 缩减通道数
        x = self.stem(x)  # (B, base_channels, H, W)
        stem_features = x
        
        # 编码器
        enc1 = self.enc1(x)  # 1/2分辨率
        enc2 = self.enc2(enc1)  # 1/4分辨率
        enc3 = self.enc3(enc2)  # 1/8分辨率
        
        # 简化的金字塔池化
        ppm_features = self.ppm(enc3)  # 全局特征
        # 上采样到与enc3相同大小
        ppm_features = F.interpolate(
            ppm_features, 
            size=enc3.shape[2:], 
            mode='bilinear', 
            align_corners=True
        )
        
        # 特征融合 - 使用concat而不是复杂的注意力机制
        bottleneck = torch.cat([enc3, ppm_features], dim=1)
        
        # 解码器 - 上采样和特征融合
        dec3 = self.dec3(bottleneck)
        dec3_cat = torch.cat([dec3, enc2], dim=1)
        
        dec2 = self.dec2(dec3_cat)
        dec2_cat = torch.cat([dec2, enc1], dim=1)
        
        dec1 = self.dec1(dec2_cat)
        
        # 最终分类
        out = self.final_conv(dec1)
        
        # 确保输出尺寸与输入匹配
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        # 辅助分类分支 - 用于深层监督，确保与输入尺寸相同
        aux_out = self.aux_head(enc2)
        if aux_out.shape[2:] != input_size:
            aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)
            
        # 返回主输出和辅助输出
        return out, aux_out

####################################
# 6. EfficientLightSegNet模型
####################################
class EfficientLightSegNet(nn.Module):
    """高效轻量级分割网络，专为高维特征表示(128通道)设计"""
    def __init__(self, input_channels=128, num_classes=20, dropout_rate=0.1):
        super(EfficientLightSegNet, self).__init__()
        
        # 初始stem不降维，保持128通道
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(input_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 输出: (B, 128, 64, 64)
        
        # Encoder路径
        self.enc1 = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128, reduction_ratio=8)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv(128, 160),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            SEBlock(160, reduction_ratio=8)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            DepthwiseSeparableConv(160, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256, reduction_ratio=8)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512, reduction_ratio=16),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
        # Decoder路径
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=1)
        )
        self.dec3 = nn.Sequential(
            DepthwiseSeparableConv(512, 256),  # 512 = 256 (上采样) + 256 (跳跃连接)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 160, kernel_size=1)
        )
        self.dec2 = nn.Sequential(
            DepthwiseSeparableConv(320, 160),  # 320 = 160 (上采样) + 160 (跳跃连接)
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(160, 128, kernel_size=1)
        )
        self.dec1 = nn.Sequential(
            DepthwiseSeparableConv(256, 128),  # 256 = 128 (上采样) + 128 (跳跃连接)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 辅助分类头（从第二个编码器阶段）
        self.aux_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        
        # 最终分类层
        self.final = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Stem
        x = self.stem(x)
        
        # Encoder路径
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 辅助输出
        aux_out = self.aux_head(enc3)
        
        # Decoder路径
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # 最终分类
        out = self.final(x)
        
        # 确保输出尺寸与输入匹配
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
            
        # 确保辅助输出尺寸也与输入匹配
        if aux_out.shape[2:] != input_size:
            aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)
            
        return out, aux_out

####################################
# 7. 平衡型高效分割网络EffectiveSegNet
####################################
class EffectiveSegNet(nn.Module):
    """平衡型分割网络，参数量控制在5M以内但具有足够学习能力"""
    def __init__(self, input_channels=128, num_classes=20, dropout_rate=0.1):
        super(EffectiveSegNet, self).__init__()
        
        # 初始通道降维 - 从128降到64通道，但保留足够信息
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder通道配置
        enc_channels = [64, 128, 256, 512]
        
        # Encoder Block 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(enc_channels[0], enc_channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels[1], enc_channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder Block 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(enc_channels[1], enc_channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels[2], enc_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder Block 3
        self.enc3 = nn.Sequential(
            nn.Conv2d(enc_channels[2], enc_channels[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels[3], enc_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[3]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # ASPP模块 (Atrous Spatial Pyramid Pooling) - 多尺度特征提取
        self.aspp = nn.ModuleList([
            nn.Sequential(  # 全局池化分支
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(enc_channels[3], 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(  # 1x1卷积分支
                nn.Conv2d(enc_channels[3], 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(  # 膨胀率=6分支
                nn.Conv2d(enc_channels[3], 256, kernel_size=3, padding=6, dilation=6, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(  # 膨胀率=12分支
                nn.Conv2d(enc_channels[3], 256, kernel_size=3, padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(  # 膨胀率=18分支
                nn.Conv2d(enc_channels[3], 256, kernel_size=3, padding=18, dilation=18, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        ])
        
        # ASPP特征融合
        self.aspp_fusion = nn.Sequential(
            nn.Conv2d(256*5, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # Decoder部分
        # 第一个上采样块
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合1 (与enc2特征融合)
        self.fusion1 = nn.Sequential(
            nn.Conv2d(256 + enc_channels[2], 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 第二个上采样块
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合2 (与enc1特征融合)
        self.fusion2 = nn.Sequential(
            nn.Conv2d(128 + enc_channels[1], 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 第三个上采样块
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合3 (与stem特征融合)
        self.fusion3 = nn.Sequential(
            nn.Conv2d(64 + enc_channels[0], 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 最终分类层
        self.final_conv = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # 辅助损失分支 (从enc2产生)
        self.aux_head = nn.Sequential(
            nn.Conv2d(enc_channels[2], 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        input_size = x.shape[2:]  # 保存输入尺寸
        
        # Stem
        x_stem = self.stem(x)  # (B, 64, H, W)
        
        # Encoder
        x_enc1 = self.enc1(x_stem)  # (B, 128, H/2, W/2)
        x_enc2 = self.enc2(x_enc1)  # (B, 256, H/4, W/4)
        x_enc3 = self.enc3(x_enc2)  # (B, 512, H/8, W/8)
        
        # ASPP特征提取
        aspp_outputs = []
        for i, aspp_branch in enumerate(self.aspp):
            if i == 0:  # 全局池化分支需要上采样
                out = aspp_branch(x_enc3)
                out = F.interpolate(out, size=x_enc3.shape[2:], mode='bilinear', align_corners=True)
                aspp_outputs.append(out)
            else:
                aspp_outputs.append(aspp_branch(x_enc3))
        
        # 融合ASPP特征
        aspp_result = torch.cat(aspp_outputs, dim=1)
        aspp_result = self.aspp_fusion(aspp_result)  # (B, 256, H/8, W/8)
        
        # 辅助分类分支
        aux_output = self.aux_head(x_enc2)  # (B, num_classes, H/4, W/4)
        aux_output = F.interpolate(aux_output, size=input_size, mode='bilinear', align_corners=True)
        
        # Decoder
        # 第一次上采样+融合
        x = self.dec1(aspp_result)  # (B, 256, H/4, W/4)
        x = torch.cat([x, x_enc2], dim=1)  # (B, 256+256, H/4, W/4)
        x = self.fusion1(x)  # (B, 256, H/4, W/4)
        
        # 第二次上采样+融合
        x = self.dec2(x)  # (B, 128, H/2, W/2)
        x = torch.cat([x, x_enc1], dim=1)  # (B, 128+128, H/2, W/2)
        x = self.fusion2(x)  # (B, 128, H/2, W/2)
        
        # 第三次上采样+融合
        x = self.dec3(x)  # (B, 64, H, W)
        x = torch.cat([x, x_stem], dim=1)  # (B, 64+64, H, W)
        x = self.fusion3(x)  # (B, 64, H, W)
        
        # 最终分类
        x = self.final_conv(x)  # (B, num_classes, H, W)
        
        # 确保输出与输入尺寸相同
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
            
        return x, aux_output

####################################
# 8. Lovász损失和组合损失函数
####################################
# Lovász loss实现
def lovasz_grad(gt_sorted):
    """计算Lovász扩展的梯度"""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # 避免单例情况下除以0
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class SimplifiedLovaszSoftmax(nn.Module):
    """简化版Lovász损失，专注于IoU优化"""
    def __init__(self, ignore_index=None):
        super(SimplifiedLovaszSoftmax, self).__init__()
        self.ignore_index = ignore_index
        
    def forward(self, probas, labels):
        # 确保probas是概率分布
        if probas.dim() == 4:  # (B, C, H, W)
            probas = F.softmax(probas, dim=1)
            
        # 平铺为1D
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)
        labels = labels.view(-1)  # (B*H*W,)
        
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            probas, labels = probas[mask], labels[mask]
            
        if len(labels) == 0:
            return torch.tensor(0.0, device=probas.device)
            
        loss = 0.0
        # 只针对前10个类别计算Lovász损失，降低计算量
        max_classes = min(10, C)
        for c in range(max_classes):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            fg_sorted = fg[perm]
            grad = lovasz_grad(fg_sorted)
            loss = loss + torch.dot(errors_sorted, grad)
                
        return loss / max_classes

class CombinedCELovászLoss(nn.Module):
    def __init__(self, ignore_index=19, ce_weight=0.5, lovasz_weight=0.5, distribution=None):
        super(CombinedCELovászLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        
        # 配置CE损失
        if distribution is not None:
            weights = []
            for i, w in enumerate(distribution):
                if i == ignore_index:
                    weights.append(0.0)
                else:
                    weights.append(1.0 / w if w > 0 else 1.0)
            
            # 归一化权重
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
            self.register_buffer('weight_tensor', torch.tensor(weights, dtype=torch.float))
            self.ce_loss = nn.CrossEntropyLoss(
                weight=None,  # Will set this in forward pass
                ignore_index=ignore_index
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self.register_buffer('weight_tensor', None)
            
        # 配置简化版Lovász损失
        self.lovasz_loss = SimplifiedLovaszSoftmax(ignore_index=ignore_index)
        
    def forward(self, inputs, targets):
        # 计算交叉熵损失 - 确保权重在正确的设备上
        if hasattr(self, 'weight_tensor') and self.weight_tensor is not None:
            # 使用设备上的权重
            self.ce_loss.weight = self.weight_tensor.to(inputs.device)
        
        ce_loss = self.ce_loss(inputs, targets)
        
        # 计算Lovász损失 (需要先转换为概率)
        lovasz_loss = self.lovasz_loss(inputs, targets)
        
        # 组合损失
        return self.ce_weight * ce_loss + self.lovasz_weight * lovasz_loss

####################################
# 9. 数据集划分与 DataLoader
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
# 10. 计算指标
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
# 11. 训练、验证与测试过程
####################################
def train_model(model, train_loader, val_loader, device, num_classes, epochs=300, lr=3e-4, model_name="effective_segnet"):
    # 定义类别分布（用于loss计算）
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
    
    # 使用组合损失函数
    criterion = CombinedCELovászLoss(
        ignore_index=19, 
        ce_weight=0.5, 
        lovasz_weight=0.5,
        distribution=distribution
    )
    
    # 优化器: AdamW + 权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 学习率调度器: OneCycleLR - 先上升后下降
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # 预热占总步数的20%
        anneal_strategy='cos'  # 余弦退火
    )
    
    best_val_miou = 0
    checkpoint_dir = os.path.join("checkpoints", "downstream")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_pastis_patch_seg_{model_name}_ckpt.pth")
    
    # EMA模型 - 保持权重的指数滑动平均
    ema_model = None
    ema_decay = 0.99
    
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for batch_idx, (rep, target) in enumerate(progress_bar):
            rep = rep.to(device)       # (B, 128, H, W)
            target = target.to(device) # (B, H, W)
            
            optimizer.zero_grad()
            
            # 前向传播 (返回主输出和辅助输出)
            output, aux_output = model(rep)
            
            # 计算主损失和辅助损失
            main_loss = criterion(output, target)
            aux_loss = criterion(aux_output, target) if aux_output is not None else 0
            
            # 总损失 = 主损失 + 辅助损失 * 权重
            loss = main_loss + 0.4 * aux_loss
            
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            
            # 计算指标
            pred = output.argmax(dim=1)  # (B, H, W)
                
            acc, f1, miou = compute_segmentation_metrics(pred, target, num_classes)
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{acc:.4f}",
                "F1": f"{f1:.4f}",
                "mIoU": f"{miou:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6f}"
            })
            
            # 更新EMA模型
            if ema_model is None:
                ema_model = {name: param.clone().detach() for name, param in model.named_parameters()}
            else:
                for name, param in model.named_parameters():
                    ema_model[name] = ema_model[name] * ema_decay + param.clone().detach() * (1 - ema_decay)
        
        # 验证阶段
        # 前几个epoch设置为warm-up，不验证
        validation_start = 30
        if epoch <= validation_start:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {np.mean(train_losses):.4f} | "
                  f"Skipping validation (warmup)")
            continue
        
        # 保存当前模型参数
        current_model_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # 如果有EMA模型，则使用EMA参数进行验证
        if ema_model is not None and epoch > validation_start + 50:  # 仅在训练后期使用EMA
            for name, param in model.named_parameters():
                param.data.copy_(ema_model[name])
        
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        val_miou = 0.0
        with torch.no_grad():
            for rep, target in val_loader:
                rep = rep.to(device)
                target = target.to(device)
                
                output, aux_output = model(rep)
                
                # 只使用主输出计算验证损失
                loss = criterion(output, target)
                val_loss += loss.item()
                
                pred = output.argmax(dim=1)
                    
                acc, f1, miou = compute_segmentation_metrics(pred, target, num_classes)
                val_acc += acc
                val_f1 += f1
                val_miou += miou
                
        # 恢复模型参数
        for name, param in model.named_parameters():
            param.data.copy_(current_model_params[name])
        
        num_val = len(val_loader)
        avg_val_loss = val_loss / num_val
        avg_val_acc = val_acc / num_val
        avg_val_f1 = val_f1 / num_val
        avg_val_miou = val_miou / num_val
        print(f"Epoch {epoch}/{epochs} - Train Loss: {np.mean(train_losses):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | "
              f"Val F1: {avg_val_f1:.4f} | Val mIoU: {avg_val_miou:.4f}")
        
        # 保存验证最好的 checkpoint
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            
            # 如果使用了EMA，保存EMA模型
            if ema_model is not None and epoch > validation_start + 50:
                for name, param in model.named_parameters():
                    param.data.copy_(ema_model[name])
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_miou': avg_val_miou
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved new best checkpoint at epoch {epoch} with Val Loss: {avg_val_loss:.4f}, Val mIoU: {avg_val_miou:.4f}")
            
            # 恢复原来的模型参数
            for name, param in model.named_parameters():
                param.data.copy_(current_model_params[name])

####################################
# 12. 测试过程
####################################
def test_model(model, test_loader, device, num_classes, model_name="effective_segnet"):
    """测试模型并返回详细的评估指标"""
    # 加载最佳 checkpoint
    checkpoint_path = os.path.join("checkpoints", "downstream", f"best_pastis_patch_seg_{model_name}_ckpt.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}")
        if 'val_miou' in checkpoint:
            print(f"Validation mIoU: {checkpoint['val_miou']:.4f}")
    else:
        print("No checkpoint found. Using current model for testing.")
    
    # 定义分布
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
    
    # 使用组合损失
    criterion = CombinedCELovászLoss(
        ignore_index=19, 
        ce_weight=0.5, 
        lovasz_weight=0.5,
        distribution=distribution
    )
    
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for rep, target in tqdm(test_loader, desc="Testing"):
            rep = rep.to(device)
            target = target.to(device)
            
            # 模型输出
            output, _ = model(rep)
            
            # 只使用主输出
            loss = criterion(output, target)
            pred = output.argmax(dim=1)
                
            test_loss += loss.item()
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
    
    return avg_test_loss, overall_acc, overall_f1, overall_iou, per_class

####################################
# 13. 主函数和参数解析
####################################
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation models on PASTIS patches')
    parser.add_argument('--model', type=str, default='effective_segnet', 
                        choices=['unet', 'depthwise_unet', 'efficient_light_segnet', 'lightweight_segnet', 'effective_segnet'],
                        help='Model type to use')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    return parser.parse_args()

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 数据所在的根目录（请根据实际情况修改路径）
    rep_root_dir = "/scratch/zf281/pastis/representation/dawn_val_acc_75946"
    label_root_dir = "/scratch/zf281/pastis/data/ANNOTATIONS"
    metadata_path = "/scratch/zf281/pastis/data/metadata.geojson"
    rep_files, target_files = get_file_lists(rep_root_dir, label_root_dir)
    
    # 根据 metadata 中的 Fold 字段进行数据集划分
    (train_rep, train_target), (val_rep, val_target), (test_rep, test_target) = split_data_by_metadata(rep_files, target_files, metadata_path)
    
    # 训练集开启增强的数据增强
    train_dataset = PastisPatchDataset(train_rep, train_target, augment=True)
    val_dataset = PastisPatchDataset(val_rep, val_target, augment=False)
    test_dataset = PastisPatchDataset(test_rep, test_target, augment=False)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 20  # 总共20个类别，其中标签19为空标签
    
    # 深度可分离卷积UNet参数
    depthwise_unet_config = {
        "in_channels": 128,
        "num_classes": num_classes,
        "features": [128, 256, 512],
        "use_channel_attention": True,
        "channel_reduction": 16,
        "use_residual": True,
        "dropout_rate": 0.1
    }
    
    # EfficientLightSegNet参数
    efficient_light_segnet_config = {
        "input_channels": 128,
        "num_classes": num_classes,
        "dropout_rate": 0.1
    }
    
    # 选择模型 - 使用我们的新模型
    model_type = "effective_segnet"  # 可选: "unet", "depthwise_unet", "efficient_light_segnet", "effective_segnet"
    
    if model_type == "unet":
        model = UNet(in_channels=128, num_classes=num_classes, dropout=0.1)
        model_name = "unet"
    elif model_type == "depthwise_unet":
        model = DepthwiseUNet(**depthwise_unet_config)
        model_name = "depthwise_unet"
    elif model_type == "efficient_light_segnet":
        model = EfficientLightSegNet(**efficient_light_segnet_config)
        model_name = "efficient_light_segnet"
    elif model_type == "effective_segnet":
        model = EffectiveSegNet(
            input_channels=128,
            num_classes=num_classes,
            dropout_rate=0.1
        )
        model_name = "effective_segnet"
    elif model_type == "lightweight_segnet":
        model = LightweightSegNet(
            input_channels=128,
            num_classes=num_classes,
            dropout_rate=0.2,
            width_multiplier=0.25
        )
        model_name = "lightweight_segnet"
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    model = model.to(device)
    
    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型 {model_name} 的参数量: {num_params:,}")
    
    # 打印可训练参数量
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n可训练参数量: {num_trainable_params:,} ({num_trainable_params/1e6:.2f}M)")
    
    # 打印模型架构
    print(f"模型架构:\n{model.__class__.__name__}")
    
    # 训练模型
    epochs = 300   # 更多epoch以充分训练
    lr = 3e-4      # 较小的学习率
    
    train_model(model, train_loader, val_loader, device, num_classes, epochs=epochs, lr=lr, model_name=model_name)
    
    # 测试模型
    test_model(model, test_loader, device, num_classes, model_name=model_name)

if __name__ == "__main__":
    main()