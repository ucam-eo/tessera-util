import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NUMEXPR_MAX_THREADS"] = "24"
import time
import math
import subprocess
import argparse
import logging
from datetime import datetime
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import wandb

from datasets.ssl_dataset import HDF5Dataset_Multimodal_Tiles_Iterable
from models.modules import TransformerEncoder, SpectralTemporalTransformer
from models.ssl_model import BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate
from utils.metrics import rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import glob
import re
import torch.nn.functional as F
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

NUM_DAYS = 365
NUM_BANDS = 10
EMBED_DIM = 512
OUTPUT_DIM = 128


# 归一化参数
S2_BAND_MEAN = np.array([1711.0938,1308.8511,1546.4543,3010.1293,3106.5083,
                           2068.3044,2685.0845,2931.5889,2514.6928,1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026,1862.9751,1803.1792,1741.7837,1677.4543,
                          1888.7862,1736.3090,1715.8104,1514.5199,1398.4779], dtype=np.float32)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def linear_probe_evaluate(model, val_loader, device='cuda'):
    """
    使用验证集计算模型嵌入后训练出的 logistic regression 分类器的表现，
    返回 accuracy, weighted F1 score 以及混淆矩阵。
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    # max_samples = 20000
    with torch.no_grad():
        for s2_sample, mask, label in val_loader:
            s2_sample = s2_sample.to(device)
            mask = mask.to(device)
            out = model(s2_sample, mask)
            if isinstance(out, tuple):
                out = out[1]
            emb = out.cpu().numpy()
            embeddings_list.append(emb)
            # 注意：label 可能需要 .cpu() 后再转换成 numpy 数组
            labels_list.extend(label.cpu().numpy())
            # if len(labels_list) >= max_samples:
            #     break
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels_arr = np.array(labels_list, dtype=np.int64)
    N = embeddings.shape[0]
    if N < 2:
        return 1.0, 1.0, None
    np.random.seed(42)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.3 * N)
    train_idx = idx[:split]
    test_idx = idx[split:]
    X_train = embeddings[train_idx]
    y_train = labels_arr[train_idx]
    X_test = embeddings[test_idx]
    y_test = labels_arr[test_idx]
    clf = LogisticRegression(max_iter=100000, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, digits=4)
    return acc, f1, cm, cr

class AustrianCropValDataset(Dataset):
    """
    验证集数据读取，仅使用哨兵2数据。
    数据预处理说明：
      - bands_downsample_100.npy：原始形状 (T, H, W, 10)，T小于365，补全至365个时间步，
        没有数据的时间步用0填充。
      - masks_downsample_100.npy：原始形状 (T, H, W)，其中0表示无效，1表示有效；
        扩展至365个时间步，新mask含义：
            0：缺失，
            1：有值但无效（原mask==0），
            2：有值且有效（原mask==1）。
      - doys.npy：形状 (T,)，每个值范围在1~365，表示对应观测的时间步。
      - labels：形状 (H, W) 对应真实标签（返回时减1）。
      
    最终返回：
         s2_sample：形状 (365, 10) 的bands数据，
         mask：      形状 (365,) 的新mask，
         label：     标量（对应地物类别，标签减1）。
         
    注意：此版本不做任何像元的排除或筛选，直接使用所有像元。
    """
    def __init__(self,
                 val_s2_bands_file_path,
                 val_s2_masks_file_path,
                 val_s2_doy_file_path,
                 val_labels_path,
                 standardize=True):
        super().__init__()
        # 加载数据
        self.s2_bands_data = np.load(val_s2_bands_file_path)   # shape: (T, H, W, 10)
        self.s2_masks_data = np.load(val_s2_masks_file_path)     # shape: (T, H, W)
        self.s2_doys_data  = np.load(val_s2_doy_file_path)        # shape: (T,)
        self.labels        = np.load(val_labels_path)            # shape: (H, W)

        self.standardize = standardize
        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std  = S2_BAND_STD

        # 获取空间维度
        T, H, W, C = self.s2_bands_data.shape
        # 不进行任何排除，直接使用所有像元
        indices = np.indices((H, W)).reshape(2, -1).T
        
        self.valid_pixels = []
        for (i, j) in indices:
            if self.labels[i, j] == 0:
                continue
            self.valid_pixels.append((i, j))

    def __len__(self):
        return len(self.valid_pixels)

    def _process_s2(self, bands, masks, doys):
        """
        将某个像元的哨兵2数据扩展到365个时间步。
        参数：
           bands：原始bands数据，形状 (T, 10)
           masks：原始mask数据，形状 (T,) ，取值0或1
           doys：  原始日期数组，形状 (T,) ，取值范围1~365
        返回：
           new_bands：形状 (365, 10)，未归一化的bands数据（之后只对有效数据归一化），
                       无数据的时刻为0；
           new_mask： 形状 (365,) ，新mask，其中：
                      0：缺失，
                      1：有值但无效（原mask==0），
                      2：有值且有效（原mask==1）。
        """
        num_days = 365
        C = bands.shape[-1]
        new_bands = np.zeros((num_days, C), dtype=bands.dtype)
        new_mask = np.zeros((num_days,), dtype=np.int32)
        
        # 根据doys数组，将原始T个时间步的数据放到对应位置
        for t in range(len(doys)):
            day = int(doys[t])
            pos = day - 1  # 转为0-index
            if masks[t] == 1:
                new_mask[pos] = 2
            # else:
            #     new_mask[pos] = 1
            new_bands[pos] = bands[t]
        # 转为float
        new_bands = new_bands.astype(np.float32)
        if self.standardize:
            # 仅对有效数据（new_mask != 0）的时间步进行归一化，其它时刻保持0
            valid_idx = (new_mask != 0)
            new_bands[valid_idx] = (new_bands[valid_idx] - self.s2_band_mean) / (self.s2_band_std + 1e-9)
        return new_bands, new_mask

    def __getitem__(self, idx):
        i, j = self.valid_pixels[idx]
        label = self.labels[i, j] - 1
        
        # 获取该像元在所有T时刻的bands和mask数据
        s2_bands_ij = self.s2_bands_data[:, i, j, :]  # (T, 10)
        s2_masks_ij = self.s2_masks_data[:, i, j]       # (T,)
        
        # 扩展到365个时间步
        s2_sample, mask = self._process_s2(s2_bands_ij, s2_masks_ij, self.s2_doys_data)
        
        # 转为torch tensor
        s2_sample = torch.tensor(s2_sample, dtype=torch.float32)
        mask      = torch.tensor(mask, dtype=torch.int64)
        label     = torch.tensor(label, dtype=torch.long)
        
        return s2_sample, mask, label

# 自定义位置编码
class PositionalEncoding(nn.Module):
    """
    Transformer模型的位置编码
    """
    def __init__(self, d_model, max_len=NUM_DAYS):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册缓冲区(不是参数但应保存在state_dict中)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        将位置编码添加到输入嵌入中
        
        参数:
            x: 形状为(batch_size, seq_len, d_model)的输入嵌入
            
        返回:
            x + 位置编码
        """
        return x + self.pe[:, :x.size(1), :]

# 时间注意力池化，用于减少序列维度
class TemporalAttentionPool(nn.Module):
    """
    基于注意力的池化，用于减少时间维度
    """
    def __init__(self, embed_dim):
        super(TemporalAttentionPool, self).__init__()
        self.attention = nn.Linear(embed_dim, 1)
        
    def forward(self, x, mask):
        """
        在时间维度上应用注意力池化
        
        参数:
            x: 形状为(batch_size, seq_len, embed_dim)的输入张量
            mask: 形状为(batch_size, seq_len)的二进制掩码
                  其中0表示要忽略的位置
                  
        返回:
            pooled: 形状为(batch_size, embed_dim)的张量
        """
        # 获取注意力分数
        scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        
        # 将mask为0的位置的分数设置为-inf(以忽略这些位置)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # 应用注意力权重到嵌入
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch_size, embed_dim)
        
        return pooled

# 时间序列Transformer编码器
class TimeSeriesTransformer(nn.Module):
    """
    基于Transformer的时间序列数据编码器
    """
    def __init__(self, input_dim, embed_dim, output_dim, nhead=8, num_layers=4, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # 光谱波段的嵌入
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # 云标志矩阵 - 用于标记云遮挡的时间步
        self.cloud_flag_matrix = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 时间池化和最终投影
        self.temporal_pool = TemporalAttentionPool(embed_dim)
        self.final_projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x, mask):
        """
        Transformer模型的前向传递
        
        参数:
            x: 形状为(batch_size, seq_len, input_dim)的输入张量
            mask: 形状为(batch_size, seq_len)的掩码张量，值为0, 1, 2
                 0: 无数据, 1: 云遮挡数据, 2: 有效数据
                 
        返回:
            output: 形状为(batch_size, output_dim)的张量
        """
        batch_size, seq_len, _ = x.shape
        
        # 将输入投影到嵌入维度
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # 创建云掩码(mask=1的位置)
        cloud_mask = (mask == 1).unsqueeze(-1).expand(-1, -1, self.embed_dim) # (batch_size, seq_len, embed_dim)
        
        # 将云标志变换应用于所有云遮挡时间步
        # 需要重塑进行批量矩阵乘法
        reshaped_embedded = embedded.reshape(-1, self.embed_dim)  # (batch_size*seq_len, embed_dim)
        transformed = torch.mm(reshaped_embedded, self.cloud_flag_matrix)  # (batch_size*seq_len, embed_dim)
        transformed = transformed.reshape(batch_size, seq_len, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        
        # 只在mask == 1(云遮挡)的位置应用变换
        embedded = torch.where(cloud_mask, transformed, embedded)
        
        # 添加位置编码
        embedded = self.pos_encoder(embedded)
        
        # 为transformer创建关键填充掩码(原始掩码为0的位置为True)
        # 这告诉transformer哪些位置要忽略
        key_padding_mask = (mask == 0)
        
        # 应用Transformer编码器
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=key_padding_mask)
        # 输出形状: (batch_size, seq_len, embed_dim)
        
        # 应用时间注意力池化
        pooled = self.temporal_pool(encoded, mask)
        # 输出形状: (batch_size, embed_dim)
        
        # 投影到输出维度
        output = self.final_projection(pooled)
        # 输出形状: (batch_size, output_dim)
        
        return output

class IterableSentinel2PixelDataset(IterableDataset):
    """
    哨兵2像素时序数据集的可迭代版本，用于处理大型数据集
    """
    def __init__(self, data_dir, transform=True, cloud_mask_ratio=0.7, shuffle=True):
        """
        初始化数据集
        
        参数:
            data_dir: 包含.npy文件的目录路径
            transform: 是否应用数据增强
            cloud_mask_ratio: 增强时要将多少比例的云遮挡时间步(mask=1)改为无数据(mask=0)
            shuffle: 是否打乱数据
        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.cloud_mask_ratio = cloud_mask_ratio + random.uniform(-0.1, 0.1)
        self.shuffle = shuffle
        
        # 查找所有的bands和masks文件
        bands_files = sorted(glob.glob(os.path.join(data_dir, "*_bands.npy")))
        masks_files = sorted(glob.glob(os.path.join(data_dir, "*_masks.npy")))
        
        # 提取前缀以匹配文件对
        self.pairs = []
        
        # 使用正则表达式提取前缀
        pattern = r"(B\d+_F\d+)_365_"
        
        # 构建前缀到文件路径的映射
        prefix_to_bands = {}
        prefix_to_masks = {}
        
        for bf in bands_files:
            match = re.search(pattern, os.path.basename(bf))
            if match:
                prefix = match.group(1)
                prefix_to_bands[prefix] = bf
        
        for mf in masks_files:
            match = re.search(pattern, os.path.basename(mf))
            if match:
                prefix = match.group(1)
                prefix_to_masks[prefix] = mf
        
        # 查找共同的前缀并创建文件对
        common_prefixes = set(prefix_to_bands.keys()) & set(prefix_to_masks.keys())
        for prefix in common_prefixes:
            self.pairs.append((prefix_to_bands[prefix], prefix_to_masks[prefix]))
        
        print(f"找到了 {len(self.pairs)} 对匹配的bands和masks文件")
        
        # 计算总样本数（使用内存映射模式，不加载整个数组）
        self.file_sample_counts = []
        total_samples = 0
        
        for bands_file, masks_file in self.pairs:
            # 使用mmap_mode='r'仅加载形状信息而不加载数据
            bands_shape = np.load(bands_file, mmap_mode='r').shape
            num_samples = bands_shape[0]
            self.file_sample_counts.append(num_samples)
            total_samples += num_samples
        
        self.total_samples = total_samples
        print(f"总样本数: {self.total_samples}")
    
    def __iter__(self):
        """返回数据集的迭代器"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:  # 单个worker
            return self._get_iterator(0, len(self.pairs))
        else:  # 多个worker
            # 将文件对均匀分配给每个worker
            per_worker = int(math.ceil(len(self.pairs) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.pairs))
            return self._get_iterator(start, end)
    
    def _get_iterator(self, start_idx, end_idx):
        """
        返回文件对子集的迭代器
        
        参数:
            start_idx: 文件对的起始索引
            end_idx: 文件对的结束索引
        
        返回:
            产生样本的迭代器
        """
        # 处理分配给此worker的文件对
        for idx in range(start_idx, end_idx):
            bands_file, masks_file = self.pairs[idx]
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            print(f"Worker {worker_id} 正在加载 {os.path.basename(bands_file)}")
            
            # 将当前文件加载到内存中
            try:
                bands = np.load(bands_file)
                masks = np.load(masks_file)
                
                # 创建索引列表并可选地打乱
                indices = list(range(len(bands)))
                if self.shuffle:
                    random.shuffle(indices)
                
                # 从当前文件产生样本
                for i in indices:
                    band = bands[i]  # (365, 10)
                    mask = masks[i]  # (365,)
                    
                    # 应用数据增强
                    if self.transform:
                        mask1, mask2 = self.create_augmentations(mask)
                    else:
                        mask1, mask2 = mask.copy(), mask.copy()
                    
                    # 转换为PyTorch张量
                    bands_tensor = torch.FloatTensor(band)
                    mask1_tensor = torch.LongTensor(mask1)
                    mask2_tensor = torch.LongTensor(mask2)
                    
                    yield bands_tensor, mask1_tensor, mask2_tensor
            except Exception as e:
                print(f"Worker {worker_id} 加载文件时出错: {e}")
                continue
    
    def create_augmentations(self, mask):
        """
        创建两个增强的掩码
        
        参数:
            mask: 形状为(365,)的掩码数组
            
        返回:
            mask1, mask2: 两个增强掩码，其中指定比例的云遮挡(值=1)时间步被随机更改为0
        """
        # 创建两个新掩码
        mask1 = mask.copy()
        mask2 = mask.copy()
        
        # 找出mask为1的索引(云遮挡数据)
        cloudy_indices = np.where(mask == 1)[0]
        
        if len(cloudy_indices) > 0:  # 只有当有云遮挡索引时才继续
            # 随机选择比例的索引更改为0
            num_to_change = max(1, int(self.cloud_mask_ratio * len(cloudy_indices)))
            
            # 对mask1
            indices_to_change = np.random.choice(cloudy_indices, num_to_change, replace=False)
            mask1[indices_to_change] = 0
            
            # 对mask2(不同的随机选择)
            indices_to_change = np.random.choice(cloudy_indices, num_to_change, replace=False)
            mask2[indices_to_change] = 0
        
        return mask1, mask2
    
    def __len__(self):
        """返回数据集的总大小（用于估计总迭代次数）"""
        return self.total_samples

class MultimodalBTModel(nn.Module):
    def __init__(self, s2_backbone, projector, return_repr=False):
        """
        fusion_method: 'sum', 'concat' 或 'transformer'
        若使用'transformer'则需要提供latent_dim
        """
        super().__init__()
        self.s2_backbone = s2_backbone
        self.projector = projector
        self.return_repr = return_repr
            
        # self.dim_reducer = nn.Sequential(
        #     nn.Linear(512, 128)
        # )

    def forward(self, x, mask):
        s2_repr = self.s2_backbone(x, mask)
        # s2_repr = self.dim_reducer(s2_repr)
        feats = self.projector(s2_repr)
        if self.return_repr:
            return feats, s2_repr
        return feats

def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training")
    parser.add_argument('--config', type=str, default="configs/ssl_config_365.py", help="Path to config file (e.g. configs/ssl_config.py)")
    return parser.parse_args()

def main():
    args_cli = parse_args()
    # 加载配置
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    # 日志设置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    run_name = f"BT_Iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_run = wandb.init(project="btfm-365", name=run_name, config=config)

    # 创建数据集
    data_dir = "data/d-pixel-365/s2"
    dataset = IterableSentinel2PixelDataset(data_dir=data_dir, transform=True, cloud_mask_ratio=0.8)
    
    # 更新配置中的总样本数（如果需要）
    if config.get('total_samples') is None:
        config['total_samples'] = dataset.total_samples
    
    total_steps = config['epochs'] * config['total_samples'] // config['batch_size']
    logging.info(f"Total steps = {total_steps}")
    
    # 创建数据加载器，设置为8个workers
    batch_size = config['batch_size']
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=24,
        pin_memory=True
    )
    
    # 建立模型
    s2_backbone = TimeSeriesTransformer(
        input_dim=NUM_BANDS,
        embed_dim=EMBED_DIM,
        output_dim=OUTPUT_DIM
    )
    
    projector = ProjectionHead(128, 2048, 2048)
    
    model = MultimodalBTModel(s2_backbone, projector=projector, return_repr=True)
    model = model.to(device)
    
    val_dataset = AustrianCropValDataset(
        val_s2_bands_file_path=config['val_s2_bands_file_path'],
        val_s2_masks_file_path=config['val_s2_masks_file_path'],
        val_s2_doy_file_path=config['val_s2_doy_file_path'],
        val_labels_path=config['val_labels_path'],
        standardize=True
    )
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=8)
    
    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])

    logging.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    weight_params = [p for n, p in model.named_parameters() if p.ndim > 1]
    bias_params   = [p for n, p in model.named_parameters() if p.ndim == 1]
    # optimizer = torch.optim.SGD([{'params': weight_params}, {'params': bias_params}],
    #                             lr=config['learning_rate'], momentum=0.9, weight_decay=1e-6)
    optimizer = torch.optim.AdamW([{'params': weight_params}, {'params': bias_params}],
                               lr=config['learning_rate'], weight_decay=1e-6)
    
    # 根据配置控制是否启用AMP
    if config.get('apply_amp', False):
        scaler = amp.GradScaler()
    else:
        scaler = None

    step = 0
    examples = 0
    last_time = time.time()
    last_examples = 0
    rolling_loss = []
    rolling_size = 40
    best_val_acc = 0.0
    # 获取时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_{timestamp}.pt")

    for epoch in range(config['epochs']):
        model.train()
        for batch_idx, (bands, mask1, mask2) in enumerate(dataloader):
            # 将数据移至设备
            bands = bands.to(device)    # (batch_size, 365, 10)
            mask1 = mask1.to(device)    # (batch_size, 365)
            mask2 = mask2.to(device)    # (batch_size, 365)
            
            adjust_learning_rate(optimizer, step, total_steps, config['learning_rate'],
                                 config['warmup_ratio'], config['plateau_ratio'])
            optimizer.zero_grad()
            # 使用apply_amp参数决定是否启用自动混合精度
            with (amp.autocast() if config.get('apply_amp', False) else nullcontext()):
                # 前向传递
                z1, repr1 = model(bands, mask1)  # (batch_size, 128)
                z2, repr2 = model(bands, mask2)  # (batch_size, 128)
                loss_main, bar_main, off_main = criterion(z1, z2)
                loss_mix = 0.0
                total_loss = loss_main + loss_mix
            if config.get('apply_amp', False):
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()
            examples += mask1.size(0)
            if step % config['log_interval_steps'] == 0:
                current_time = time.time()
                exps = (examples - last_examples) / (current_time - last_time)
                last_time = current_time
                last_examples = examples
                rolling_loss.append(loss_main.item())
                if len(rolling_loss) > rolling_size:
                    rolling_loss = rolling_loss[-rolling_size:]
                avg_loss = sum(rolling_loss) / len(rolling_loss)
                current_lr = optimizer.param_groups[0]['lr']
                erank_z = rankme(z1)
                erank_repr = rankme(repr1)
                logging.info(f"[Epoch={epoch}, Step={step}] Loss={loss_main.item():.2f}, MixLoss={loss_mix:.2f}, AvgLoss={avg_loss:.2f}, LR={current_lr:.4f}, batchsize={mask1.size(0)}, Examples/sec={exps:.2f}, Rank(z)={erank_z:.4f}, Rank(repr)={erank_repr:.4f}")
                wandb_dict = {
                    "epoch": epoch,
                    "loss_main": loss_main.item(),
                    "mix_loss": loss_mix,
                    "avg_loss": avg_loss,
                    "lr": current_lr,
                    "examples/sec": exps,
                    "total_loss": total_loss.item(),
                    "rank_z": erank_z,
                    "rank_repr": erank_repr,
                }
                cross_corr_img = None
                cross_corr_img_repr = None
                if step % (10 * config['log_interval_steps']) == 0:
                    try:
                        fig_cc = plot_cross_corr(z1, z2)
                        cross_corr_img = wandb.Image(fig_cc)
                        plt.close(fig_cc)
                    except Exception:
                        pass
                    try:
                        fig_cc_repr = plot_cross_corr(repr1, repr2)
                        cross_corr_img_repr = wandb.Image(fig_cc_repr)
                        plt.close(fig_cc_repr)
                    except Exception:
                        pass
                if cross_corr_img:
                    wandb_dict["cross_corr"] = cross_corr_img
                if cross_corr_img_repr:
                    wandb_dict["cross_corr_repr"] = cross_corr_img_repr
                wandb.log(wandb_dict, step=step)
            # 验证
            if step % config['val_interval_steps'] == 0:
                model.eval()
                with torch.no_grad():
                    val_acc, val_f1, val_cm, val_cr = linear_probe_evaluate(model, val_loader, device)
                    logging.info(f"Validation accuracy: {val_acc:.4f}")
                    logging.info(f"Validation F1 score: {val_f1:.4f}")
                    logging.info(f"Validation confusion matrix:\n{val_cm}")
                    logging.info(f"Validation classification report:\n{val_cr}")
                    wandb.log({
                        "val_accuracy": val_acc,
                        "val_f1": val_f1,
                        # "val_cm": wandb.HeatMap(val_cm, title="Validation Confusion Matrix", xticklabels=True, yticklabels=True),
                        "val_cr": wandb.Html(f"<pre>{val_cr}</pre>"),
                    }, step=step)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_checkpoint(model, optimizer, epoch, step, val_acc, best_ckpt_path)
                        
                model.train()
            step += 1
        logging.info(f"Epoch {epoch} finished, current step = {step}")
    logging.info("Training completed.")
    wandb_run.finish()

if __name__ == "__main__":
    main()