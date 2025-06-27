# src/datasets/downstream_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import pandas as pd
import time
import gc
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 均值和方差
S2_BAND_MEAN = np.array([1711.0938,1308.8511,1546.4543,3010.1293,3106.5083,
                        2068.3044,2685.0845,2931.5889,2514.6928,1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026,1862.9751,1803.1792,1741.7837,1677.4543,
                        1888.7862,1736.3090,1715.8104,1514.5199,1398.4779], dtype=np.float32)
S1_BAND_MEAN = np.array([5484.0407,3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334,1726.0670], dtype=np.float32)

class AustrianCrop(Dataset):
    """
    用于下游分类任务的数据集读取。
    读取 Sentinel-2 (s2) 和 Sentinel-1 (s1) 数据，以及对应标签和 field_ids，
    并根据传入的 fids 划分训练/验证/测试。
    """
    def __init__(self, 
                 s2_bands_file_path,
                 s2_masks_file_path,
                 s2_doy_file_path,
                 s1_asc_bands_file_path,
                 s1_asc_doy_file_path,
                 s1_desc_bands_file_path,
                 s1_desc_doy_file_path,
                 labels_path,
                 field_ids_path,
                 train_fids=None,
                 val_fids=None,
                 test_fids=None,
                 lon_lat_path=None,
                 t2m_path=None,
                 lai_hv_path=None,
                 lai_lv_path=None,
                 split='train',
                 num_augmentation_pairs=1,
                 shuffle=True,
                 is_training=True,
                 standardize=True,
                 min_valid_timesteps=0,
                 sample_size_s2=20,
                 sample_size_s1=20):
        super().__init__()
        self.s2_bands_data = np.load(s2_bands_file_path)
        self.s2_masks_data = np.load(s2_masks_file_path)
        self.s2_doys_data  = np.load(s2_doy_file_path)
        self.s1_asc_bands_data = np.load(s1_asc_bands_file_path)
        self.s1_asc_doys_data  = np.load(s1_asc_doy_file_path)
        self.s1_desc_bands_data = np.load(s1_desc_bands_file_path)
        self.s1_desc_doys_data  = np.load(s1_desc_doy_file_path)
        
        # 测试float16
        # self.s2_bands_data = np.load(s2_bands_file_path).astype(np.float16)
        # self.s2_masks_data = np.load(s2_masks_file_path)
        # self.s2_doys_data  = np.load(s2_doy_file_path)
        # self.s1_asc_bands_data = np.load(s1_asc_bands_file_path).astype(np.float16)
        # self.s1_asc_doys_data  = np.load(s1_asc_doy_file_path)
        # self.s1_desc_bands_data = np.load(s1_desc_bands_file_path).astype(np.float16)
        # self.s1_desc_doys_data  = np.load(s1_desc_doy_file_path)
        
        # 生成mask：全0判断是否有效
        self.s1_asc_masks_data = (self.s1_asc_bands_data.sum(axis=-1) != 0)
        self.s1_desc_masks_data = (self.s1_desc_bands_data.sum(axis=-1) != 0)

        self.labels = np.load(labels_path)
        self.field_ids = np.load(field_ids_path)

        self.lon_lat_path = lon_lat_path
        self.t2m_path     = t2m_path
        self.lai_hv_path  = lai_hv_path
        self.lai_lv_path  = lai_lv_path

        self.train_fids = train_fids
        self.val_fids = val_fids
        self.test_fids = test_fids

        self.split = split
        self.shuffle = shuffle
        self.is_training = is_training
        self.standardize = standardize
        
        self.min_valid_timesteps = min_valid_timesteps
        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1
        self.num_augmentation_pairs = num_augmentation_pairs

        # 构建所有像素坐标
        _, H, W, _ = self.s2_bands_data.shape
        all_coords = np.indices((H, W)).reshape(2, -1).T
        
        self.valid_pixels = []
        for (i, j) in all_coords:
            fid = self.field_ids[i, j]
            if self.labels[i, j] == 0:
                continue
            if self.split == 'train' and (fid not in self.train_fids):
                continue
            if self.split == 'val' and (fid not in self.val_fids):
                continue
            if self.split == 'test' and (fid not in self.test_fids):
                continue
            s2_valid_count = self.s2_masks_data[:, i, j].sum()
            s1_asc_valid_count = self.s1_asc_masks_data[:, i, j].sum()
            s1_desc_valid_count= self.s1_desc_masks_data[:, i, j].sum()
            if (s2_valid_count >= self.min_valid_timesteps) and ((s1_asc_valid_count + s1_desc_valid_count) >= self.min_valid_timesteps):
                self.valid_pixels.append((i, j))
        if self.shuffle:
            np.random.shuffle(self.valid_pixels)
        
        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std = S1_BAND_STD

    def __len__(self):
        return len(self.valid_pixels)

    def _augment_s2(self, s2_bands, s2_masks, s2_doys):
        valid_idx = np.nonzero(s2_masks)[0]
        if len(valid_idx) < self.sample_size_s2:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s2, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s2, replace=False)
        sampled_idx = np.sort(sampled_idx)
        sub_bands = s2_bands[sampled_idx, :]
        sub_doys  = s2_doys[sampled_idx]
        if self.standardize:
            sub_bands = (sub_bands - self.s2_band_mean) / (self.s2_band_std + 1e-9)
        # doys_norm = sub_doys / 365.0
        # sin_doy = np.sin(2*np.pi*doys_norm).reshape(-1,1)
        # cos_doy = np.cos(2*np.pi*doys_norm).reshape(-1,1)
        # result = np.hstack((sub_bands, sin_doy, cos_doy))
        
        # 直接把doy连接到band上
        result = np.hstack((sub_bands, sub_doys.reshape(-1, 1)))
        return torch.tensor(result, dtype=torch.float32)

    def _augment_s1(self, s1_asc_bands, s1_asc_doys, s1_desc_bands, s1_desc_doys):
        s1_bands_all = np.concatenate([s1_asc_bands, s1_desc_bands], axis=0)
        s1_doys_all  = np.concatenate([s1_asc_doys,  s1_desc_doys], axis=0)
        valid_mask = np.any(s1_bands_all != 0, axis=-1)
        valid_idx = np.nonzero(valid_mask)[0]
        if len(valid_idx) < self.sample_size_s1:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s1, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s1, replace=False)
        sampled_idx = np.sort(sampled_idx)
        sub_bands = s1_bands_all[sampled_idx, :]
        sub_doys  = s1_doys_all[sampled_idx]
        if self.standardize:
            sub_bands = (sub_bands - self.s1_band_mean) / (self.s1_band_std + 1e-9)
        # doys_norm = sub_doys / 365.0
        # sin_doy = np.sin(2*np.pi*doys_norm).reshape(-1,1)
        # cos_doy = np.cos(2*np.pi*doys_norm).reshape(-1,1)
        # result = np.hstack((sub_bands, sin_doy, cos_doy))
        # 直接把doy连接到band上
        result = np.hstack((sub_bands, sub_doys.reshape(-1, 1)))
        return torch.tensor(result, dtype=torch.float32)

    def __getitem__(self, idx):
        i, j = self.valid_pixels[idx]
        s2_bands_ij = self.s2_bands_data[:, i, j, :]
        s2_masks_ij = self.s2_masks_data[:, i, j]
        s2_doys_ij  = self.s2_doys_data

        s1_asc_bands_ij = self.s1_asc_bands_data[:, i, j, :]
        s1_asc_doys_ij  = self.s1_asc_doys_data
        s1_desc_bands_ij = self.s1_desc_bands_data[:, i, j, :]
        s1_desc_doys_ij  = self.s1_desc_doys_data

        s2_sample = self._augment_s2(s2_bands_ij, s2_masks_ij, s2_doys_ij)
        s1_sample = self._augment_s1(s1_asc_bands_ij, s1_asc_doys_ij,
                                     s1_desc_bands_ij, s1_desc_doys_ij)
        label = self.labels[i, j] - 1  # 0-based label
        return {
            's2_sample': s2_sample,
            's1_sample': s1_sample,
            'label': label
        }

def austrian_crop_collate_fn(batch):
    s2_samples = [item['s2_sample'] for item in batch]
    s1_samples = [item['s1_sample'] for item in batch]
    labels     = [item['label'] for item in batch]
    s2_samples = torch.stack(s2_samples, dim=0)
    s1_samples = torch.stack(s1_samples, dim=0)
    labels     = torch.tensor(labels, dtype=torch.long)
    return s2_samples, s1_samples, labels

class BorneoCropRegression(Dataset):
    """
    用于回归任务的数据集。
    从 50NNL_subset 中读取光学（S2）与 SAR（S1）数据（以及 doy 与 mask 信息），
    同时加载 CHM 文件（形状 (1, H, W)）作为回归目标（冠层高度）。
    
    预处理：
      1. 排除 CHM 中为 NaN 的像素。
      2. 对于 SAR 数据，计算每个像素在所有时刻、所有通道上的和为 0 时视为无效（也可根据 min_valid_timesteps 剔除）。
    
    数据增强：
      _augment_s2 与 _augment_s1 函数均根据配置中指定的采样步数（sample_size_s2、sample_size_s1）
      从有效时刻中随机采样，并在光学/SAR波段上分别进行标准化，同时计算 doy 的正弦与余弦特征，
      从而保证最终输出的时间维度固定（一般为 20）。
    """
    def __init__(self, 
                 s2_bands_file_path,
                 s2_masks_file_path,
                 s2_doy_file_path,
                 s1_asc_bands_file_path,
                 s1_asc_doy_file_path,
                 s1_desc_bands_file_path,
                 s1_desc_doy_file_path,
                 chm_path,
                 split='train',
                 shuffle=True,
                 standardize=True,
                 min_valid_timesteps=0,
                 sample_size_s2=20,
                 sample_size_s1=20):
        super().__init__()
        # 加载 50NNL_subset 下的各个 numpy 文件
        self.s2_bands_data = np.load(s2_bands_file_path)      # shape: (T_s2, H, W, C_s2_orig)  (如 10 个通道)
        self.s2_masks_data = np.load(s2_masks_file_path)       # shape: (T_s2, H, W)
        self.s2_doys_data  = np.load(s2_doy_file_path)         # shape: (T_s2,)

        self.s1_asc_bands_data = np.load(s1_asc_bands_file_path) # shape: (T_s1, H, W, C_s1_orig) (如 2 个通道)
        self.s1_asc_doys_data  = np.load(s1_asc_doy_file_path)   # shape: (T_s1,)
        self.s1_desc_bands_data = np.load(s1_desc_bands_file_path)
        self.s1_desc_doys_data  = np.load(s1_desc_doy_file_path)
        # 根据 SAR 波段是否全 0 生成 mask
        self.s1_asc_masks_data = (self.s1_asc_bands_data.sum(axis=-1) != 0)  # shape: (T_s1, H, W)
        self.s1_desc_masks_data = (self.s1_desc_bands_data.sum(axis=-1) != 0)

        # 加载 CHM 文件，形状 (1, H, W) ，取第 0 个维度得到 (H, W)
        self.chm = np.load(chm_path)
        self.chm = self.chm[0]

        self.split = split
        self.shuffle = shuffle
        self.standardize = standardize
        self.min_valid_timesteps = min_valid_timesteps
        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1

        # 获取图像尺寸（假设 s2_bands_data 的形状为 (T, H, W, C)）
        _, self.H, self.W, _ = self.s2_bands_data.shape

        # 构建所有像素坐标，并剔除无效像素：
        # ① CHM 为 NaN 的像素；
        # ② 光学或 SAR 中有效时刻数不足（可设置 min_valid_timesteps）。
        all_coords = np.indices((self.H, self.W)).reshape(2, -1).T
        self.valid_pixels = []
        chm_nan_count = 0
        invalid_timestep_count = 0
        for (i, j) in all_coords:
            # 排除 CHM 为 NaN 的像素
            if np.isnan(self.chm[i, j]) or self.chm[i, j] <= 0:
                chm_nan_count += 1
                continue
            s2_valid_count = self.s2_masks_data[:, i, j].sum()
            s1_valid_count = self.s1_asc_masks_data[:, i, j].sum() + self.s1_desc_masks_data[:, i, j].sum()
            if (s2_valid_count < self.min_valid_timesteps) or (s1_valid_count < self.min_valid_timesteps):
                invalid_timestep_count += 1
                continue
            self.valid_pixels.append((i, j))
        logging.info(f"Excluded {chm_nan_count} pixels due to CHM NaN values.")
        logging.info(f"Excluded {invalid_timestep_count} pixels due to insufficient valid timesteps in S2/S1 data.")
        logging.info(f"Total valid pixels: {len(self.valid_pixels)}")
        if self.shuffle:
            np.random.shuffle(self.valid_pixels)

        # 设置标准化所用的均值和标准差（常量）
        self.s2_band_mean = S2_BAND_MEAN  # shape (10,)
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN  # shape (2,)
        self.s1_band_std = S1_BAND_STD

    def _augment_s2(self, s2_bands, s2_masks, s2_doys):
        """
        s2_bands: (T_s2, C_s2_orig)
        s2_masks: (T_s2,)
        s2_doys:  (T_s2,)
        采样 sample_size_s2 个有效时刻，并将原始波段归一化后拼接 sin 与 cos(doy)
        """
        valid_idx = np.nonzero(s2_masks)[0]
        if len(valid_idx) < self.sample_size_s2:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s2, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s2, replace=False)
        sampled_idx = np.sort(sampled_idx)
        sub_bands = s2_bands[sampled_idx, :]  # (sample_size_s2, C_s2_orig)
        sub_doys  = s2_doys[sampled_idx]       # (sample_size_s2,)
        if self.standardize:
            sub_bands = (sub_bands - self.s2_band_mean) / (self.s2_band_std + 1e-9)
        doys_norm = sub_doys / 365.0
        sin_doy = np.sin(2 * np.pi * doys_norm).reshape(-1, 1)
        cos_doy = np.cos(2 * np.pi * doys_norm).reshape(-1, 1)
        result = np.hstack((sub_bands, sin_doy, cos_doy))  # 最终通道数 = C_s2_orig + 2
        return torch.tensor(result, dtype=torch.float32)
    
    def _augment_s1(self, s1_asc_bands, s1_asc_doys, s1_desc_bands, s1_desc_doys):
        """
        修复原函数中 valid_idx 为空时报错的问题：
        如果 valid_idx 为空，则直接使用所有时刻的索引。
        """
        s1_bands_all = np.concatenate([s1_asc_bands, s1_desc_bands], axis=0)  # (T_s1*2, C_s1_orig)
        s1_doys_all  = np.concatenate([s1_asc_doys, s1_desc_doys], axis=0)     # (T_s1*2,)
        valid_mask = np.any(s1_bands_all != 0, axis=-1)
        valid_idx = np.nonzero(valid_mask)[0]
        # logging.info(f"valid_idx: {valid_idx}")
        if len(valid_idx) == 0:
            # 如果该像素所有时刻均为 0，则使用所有时刻的索引（即采样到的结果仍为全 0）
            valid_idx = np.arange(s1_bands_all.shape[0])
        if len(valid_idx) < self.sample_size_s1:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s1, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s1, replace=False)
        sampled_idx = np.sort(sampled_idx)
        sub_bands = s1_bands_all[sampled_idx, :]
        sub_doys  = s1_doys_all[sampled_idx]
        if self.standardize:
            sub_bands = (sub_bands - self.s1_band_mean) / (self.s1_band_std + 1e-9)
        doys_norm = sub_doys / 365.0
        sin_doy = np.sin(2 * np.pi * doys_norm).reshape(-1, 1)
        cos_doy = np.cos(2 * np.pi * doys_norm).reshape(-1, 1)
        result = np.hstack((sub_bands, sin_doy, cos_doy))
        return torch.tensor(result, dtype=torch.float32)

    def __len__(self):
        return len(self.valid_pixels)

    def __getitem__(self, idx):
        i, j = self.valid_pixels[idx]
        # 光学（S2）数据：取所有时刻中该像素处的波段数据与 mask
        s2_bands_ij = self.s2_bands_data[:, i, j, :]   # (T_s2, C_s2_orig)
        s2_masks_ij = self.s2_masks_data[:, i, j]        # (T_s2,)
        s2_doys_ij  = self.s2_doys_data                  # (T_s2,)

        # SAR 数据：分别取上升与下降方向
        s1_asc_bands_ij = self.s1_asc_bands_data[:, i, j, :]  # (T_s1, C_s1_orig)
        s1_asc_doys_ij  = self.s1_asc_doys_data               # (T_s1,)
        s1_desc_bands_ij = self.s1_desc_bands_data[:, i, j, :]  # (T_s1, C_s1_orig)
        s1_desc_doys_ij  = self.s1_desc_doys_data               # (T_s1,)

        s2_sample = self._augment_s2(s2_bands_ij, s2_masks_ij, s2_doys_ij)  # (sample_size_s2, C_s2_orig+2)
        s1_sample = self._augment_s1(s1_asc_bands_ij, s1_asc_doys_ij,
                                     s1_desc_bands_ij, s1_desc_doys_ij)       # (sample_size_s1, C_s1_orig+2)
        # 回归目标：CHM 中该像素的冠层高度（转换为 tensor，形状 (1,)）
        target = self.chm[i, j]
        target = torch.tensor([target], dtype=torch.float32)
        target_mean = 33.573902
        target_std = 18.321875
        target = (target - target_mean) / target_std
        return {
            's2_sample': s2_sample,
            's1_sample': s1_sample,
            'target': target
        }

# ------------------------------
# collate_fn 用于 DataLoader
# ------------------------------
def borneo_crop_regression_collate_fn(batch):
    s2_samples = [item['s2_sample'] for item in batch]
    s1_samples = [item['s1_sample'] for item in batch]
    targets    = [item['target'] for item in batch]
    s2_samples = torch.stack(s2_samples, dim=0)
    s1_samples = torch.stack(s1_samples, dim=0)
    targets    = torch.stack(targets, dim=0)
    return s2_samples, s1_samples, targets



class EthiopiaDataset(Dataset):
    """埃塞俄比亚作物分类数据集 - 优化版"""
    
    def __init__(self, 
                 polygons_locations_csv,
                 data_processed_dir,
                 data_raw_dir,
                 class_names,
                 split='train',
                 samples_per_class=10,
                 standardize=True,
                 max_seq_len_s2=40,
                 max_seq_len_s1=40,
                 s2_band_mean=None,
                 s2_band_std=None,
                 s1_band_mean=None,
                 s1_band_std=None):
        """初始化埃塞俄比亚作物数据集。"""
        super().__init__()
        start_time = time.time()
        print(f"初始化 {split} 数据集...")
        
        self.data_processed_dir = data_processed_dir
        self.data_raw_dir = data_raw_dir
        self.split = split
        self.samples_per_class = samples_per_class
        self.standardize = standardize
        self.max_seq_len_s2 = max_seq_len_s2
        self.max_seq_len_s1 = max_seq_len_s1
        self.class_names = class_names
        self.class_to_idx = {cls.lower(): i for i, cls in enumerate(class_names)}
        
        # 设置标准化值
        self.s2_band_mean = s2_band_mean if s2_band_mean is not None else S2_BAND_MEAN
        self.s2_band_std = s2_band_std if s2_band_std is not None else S2_BAND_STD
        self.s1_band_mean = s1_band_mean if s1_band_mean is not None else S1_BAND_MEAN
        self.s1_band_std = s1_band_std if s1_band_std is not None else S1_BAND_STD
        
        # 加载多边形-MGRS映射的CSV
        print("加载CSV数据...")
        self.df = pd.read_csv(polygons_locations_csv)
        
        # 过滤掉未匹配到任何MGRS瓦片的多边形
        self.df = self.df[self.df['found_locations'] > 0]
        
        # 获取类别信息
        if 'c_class' in self.df.columns:
            self.df['class_idx'] = self.df['c_class'].apply(lambda x: self.class_to_idx.get(x.lower(), -1))
            # 只保留有效类别索引的行
            self.df = self.df[self.df['class_idx'] != -1]
        else:
            raise ValueError("在CSV中找不到c_class列")
        
        # 重置索引，防止索引超出范围错误
        self.df = self.df.reset_index(drop=True)
        
        # 划分训练/验证集
        self.prepare_splits()
        
        # 预处理位置信息
        print("预处理位置信息...")
        self.sample_locations = []
        for idx in self.indices:
            row = self.df.iloc[idx]
            locations = self._parse_locations(row['mgrs_locations'])
            if locations:
                self.sample_locations.append({
                    'idx': idx,
                    'mgrs_tile': locations[0][0],
                    'x': locations[0][1],
                    'y': locations[0][2],
                    'label': row['class_idx']
                })
        
        # 优化：按MGRS瓦片分组，这样我们可以一次性加载每个瓦片所需的所有点
        self.tile_to_samples = {}
        for sample in self.sample_locations:
            mgrs_tile = sample['mgrs_tile']
            if mgrs_tile not in self.tile_to_samples:
                self.tile_to_samples[mgrs_tile] = []
            self.tile_to_samples[mgrs_tile].append(sample)
        
        # 获取数据文件信息
        print("检查数据文件...")
        self.data_info = {}
        for mgrs_tile in self.tile_to_samples.keys():
            s2_bands_path = os.path.join(self.data_processed_dir, mgrs_tile, "bands.npy")
            s2_masks_path = os.path.join(self.data_processed_dir, mgrs_tile, "masks.npy")
            s2_doys_path = os.path.join(self.data_processed_dir, mgrs_tile, "doys.npy")
            s1_asc_path = os.path.join(self.data_processed_dir, mgrs_tile, "sar_ascending.npy")
            s1_asc_doy_path = os.path.join(self.data_processed_dir, mgrs_tile, "sar_ascending_doy.npy")
            s1_desc_path = os.path.join(self.data_processed_dir, mgrs_tile, "sar_descending.npy")
            s1_desc_doy_path = os.path.join(self.data_processed_dir, mgrs_tile, "sar_descending_doy.npy")
            
            self.data_info[mgrs_tile] = {
                's2_bands_path': s2_bands_path if os.path.exists(s2_bands_path) else None,
                's2_masks_path': s2_masks_path if os.path.exists(s2_masks_path) else None,
                's2_doys_path': s2_doys_path if os.path.exists(s2_doys_path) else None,
                's1_asc_path': s1_asc_path if os.path.exists(s1_asc_path) else None,
                's1_asc_doy_path': s1_asc_doy_path if os.path.exists(s1_asc_doy_path) else None,
                's1_desc_path': s1_desc_path if os.path.exists(s1_desc_path) else None,
                's1_desc_doy_path': s1_desc_doy_path if os.path.exists(s1_desc_doy_path) else None,
            }
        
        # 预加载所有doys数据，因为它们很小
        print("预加载doys数据...")
        self.doys_cache = {}
        for mgrs_tile, info in self.data_info.items():
            if info['s2_doys_path'] and os.path.exists(info['s2_doys_path']):
                try:
                    self.doys_cache[f"{mgrs_tile}_s2"] = np.load(info['s2_doys_path'])
                except Exception as e:
                    print(f"加载{mgrs_tile}的S2 DOY数据出错: {e}")
                    
            if info['s1_asc_doy_path'] and os.path.exists(info['s1_asc_doy_path']):
                try:
                    self.doys_cache[f"{mgrs_tile}_s1_asc"] = np.load(info['s1_asc_doy_path'])
                except Exception as e:
                    print(f"加载{mgrs_tile}的S1升轨DOY数据出错: {e}")
                    
            if info['s1_desc_doy_path'] and os.path.exists(info['s1_desc_doy_path']):
                try:
                    self.doys_cache[f"{mgrs_tile}_s1_desc"] = np.load(info['s1_desc_doy_path'])
                except Exception as e:
                    print(f"加载{mgrs_tile}的S1降轨DOY数据出错: {e}")
        
        # 数据缓存 - 只存储提取的点数据
        self.data_cache = {}
        
        print(f"数据集初始化完成，用时 {time.time() - start_time:.2f}s")
    
    def _parse_locations(self, mgrs_locations_str):
        """解析MGRS位置字符串为元组列表 [(mgrs_tile, x, y), ...]"""
        locations = []
        if isinstance(mgrs_locations_str, str):
            loc_strs = mgrs_locations_str.split('; ')
            for loc_str in loc_strs:
                parts = loc_str.split(':')
                if len(parts) == 2:
                    mgrs_tile = parts[0]
                    coords = parts[1].strip('()').split(',')
                    if len(coords) == 2:
                        x, y = int(coords[0]), int(coords[1])
                        locations.append((mgrs_tile, x, y))
        return locations
    
    def prepare_splits(self):
        """准备训练和验证划分。"""
        train_indices = []
        val_indices = []
        
        for class_idx in range(len(self.class_names)):
            # 获取该类别的行索引
            class_indices = self.df[self.df['class_idx'] == class_idx].index.tolist()
            
            if len(class_indices) <= self.samples_per_class:
                # 如果样本数少于请求数，全部用于训练
                train_indices.extend(class_indices)
            else:
                # 否则，划分训练/验证
                np.random.shuffle(class_indices)
                train_indices.extend(class_indices[:self.samples_per_class])
                val_indices.extend(class_indices[self.samples_per_class:])
        
        # 根据划分设置适当的索引
        if self.split == 'train':
            self.indices = train_indices
        else:  # 'val'或其他
            self.indices = val_indices
            
        print(f"{self.split}划分: {len(self.indices)}个样本")
    
    def _extract_single_pixel(self, mmap_array, y, x, default_shape=None):
        """安全地从内存映射数组中提取单个像素的数据"""
        try:
            if mmap_array is None:
                return None
            # 提取数据
            if default_shape is None:
                return mmap_array[:, y, x, :].copy()
            else:
                return mmap_array[:, y, x, :].copy() if len(mmap_array.shape) > 3 else mmap_array[:, y, x].copy()
        except Exception as e:
            print(f"提取像素({y},{x})出错: {e}")
            return None
    
    def _extract_point_data(self, mgrs_tile, x, y):
        """高效提取单个点的所有数据，具有更好的错误处理"""
        # 检查缓存
        cache_key = f"{mgrs_tile}_{x}_{y}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        info = self.data_info[mgrs_tile]
        result = {}
        
        # 第一步：尝试先确定数组形状，避免加载整个数组
        try:
            # 获取S2数据
            if info['s2_bands_path']:
                try:
                    # 使用小片段加载以获取形状信息
                    with open(info['s2_bands_path'], 'rb') as f:
                        # 读取NPY文件头
                        version = np.lib.format.read_magic(f)
                        shape, _, _ = np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f)
                    
                    # 使用内存映射方式加载
                    s2_bands_mmap = np.load(info['s2_bands_path'], mmap_mode='r')
                    # 只提取特定位置的数据并复制到内存中
                    result['s2_bands'] = self._extract_single_pixel(s2_bands_mmap, y, x)
                    # 手动清理
                    del s2_bands_mmap
                except Exception as e:
                    print(f"提取 {mgrs_tile} 的S2波段数据时出错: {e}")
                    result['s2_bands'] = None
            else:
                result['s2_bands'] = None
                
            if info['s2_masks_path']:
                try:
                    s2_masks_mmap = np.load(info['s2_masks_path'], mmap_mode='r')
                    result['s2_masks'] = s2_masks_mmap[:, y, x].copy()
                    del s2_masks_mmap
                except Exception as e:
                    print(f"提取 {mgrs_tile} 的S2掩码数据时出错: {e}")
                    # 如果有波段数据但没有掩码，假设所有时间步都有效
                    if result['s2_bands'] is not None:
                        result['s2_masks'] = np.ones(result['s2_bands'].shape[0], dtype=bool)
                    else:
                        result['s2_masks'] = None
            else:
                # 如果没有掩码文件但有波段数据，假设所有时间步都有效
                if result['s2_bands'] is not None:
                    result['s2_masks'] = np.ones(result['s2_bands'].shape[0], dtype=bool)
                else:
                    result['s2_masks'] = None
            
            # 获取S2 DOY数据
            result['s2_doys'] = self.doys_cache.get(f"{mgrs_tile}_s2", None)
        
            # 获取S1升轨数据
            if info['s1_asc_path']:
                try:
                    s1_asc_mmap = np.load(info['s1_asc_path'], mmap_mode='r')
                    result['s1_asc_bands'] = self._extract_single_pixel(s1_asc_mmap, y, x)
                    del s1_asc_mmap
                    
                    # 生成掩码
                    if result['s1_asc_bands'] is not None:
                        result['s1_asc_masks'] = np.any(result['s1_asc_bands'] != 0, axis=1)
                    else:
                        result['s1_asc_masks'] = None
                        
                    result['s1_asc_doys'] = self.doys_cache.get(f"{mgrs_tile}_s1_asc", None)
                except Exception as e:
                    print(f"提取 {mgrs_tile} 的S1升轨数据时出错: {e}")
                    result['s1_asc_bands'] = None
                    result['s1_asc_masks'] = None
                    result['s1_asc_doys'] = None
            else:
                result['s1_asc_bands'] = None
                result['s1_asc_masks'] = None
                result['s1_asc_doys'] = None
            
            # 获取S1降轨数据
            if info['s1_desc_path']:
                try:
                    s1_desc_mmap = np.load(info['s1_desc_path'], mmap_mode='r')
                    result['s1_desc_bands'] = self._extract_single_pixel(s1_desc_mmap, y, x)
                    del s1_desc_mmap
                    
                    # 生成掩码
                    if result['s1_desc_bands'] is not None:
                        result['s1_desc_masks'] = np.any(result['s1_desc_bands'] != 0, axis=1)
                    else:
                        result['s1_desc_masks'] = None
                        
                    result['s1_desc_doys'] = self.doys_cache.get(f"{mgrs_tile}_s1_desc", None)
                except Exception as e:
                    print(f"提取 {mgrs_tile} 的S1降轨数据时出错: {e}")
                    result['s1_desc_bands'] = None
                    result['s1_desc_masks'] = None
                    result['s1_desc_doys'] = None
            else:
                result['s1_desc_bands'] = None
                result['s1_desc_masks'] = None
                result['s1_desc_doys'] = None
                
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"处理 {mgrs_tile} 的数据时出错: {e}")
            result = {
                's2_bands': None,
                's2_masks': None,
                's2_doys': None,
                's1_asc_bands': None,
                's1_asc_doys': None,
                's1_asc_masks': None,
                's1_desc_bands': None,
                's1_desc_doys': None,
                's1_desc_masks': None
            }
        
        # 存入缓存
        self.data_cache[cache_key] = result
        return result
    
    def _augment_s2(self, s2_bands, s2_masks, s2_doys):
        """处理S2数据为模型输入，更好地处理无效数据"""
        # 检查是否有有效数据
        if s2_bands is None or len(s2_bands) == 0:
            # 如果没有有效数据，创建零张量
            result = np.zeros((self.max_seq_len_s2, 11), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
            
        # 找到掩码为True的有效索引
        valid_idx = np.nonzero(s2_masks)[0] if s2_masks is not None else np.arange(len(s2_bands))
        
        if len(valid_idx) == 0:
            # 如果没有有效数据，创建零张量
            result = np.zeros((self.max_seq_len_s2, 11), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
        
        # 采样索引
        if len(valid_idx) < self.max_seq_len_s2:
            sampled_idx = np.random.choice(valid_idx, size=self.max_seq_len_s2, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.max_seq_len_s2, replace=False)
        sampled_idx = np.sort(sampled_idx)
        
        # 获取波段和doys
        sub_bands = s2_bands[sampled_idx, :]
        sub_doys = s2_doys[sampled_idx] if s2_doys is not None else np.zeros(len(sampled_idx))
        
        # 标准化
        if self.standardize:
            sub_bands = (sub_bands - self.s2_band_mean) / (self.s2_band_std + 1e-9)
        
        # 组合波段和doys
        result = np.hstack((sub_bands, sub_doys.reshape(-1, 1)))
        return torch.tensor(result, dtype=torch.float32)
    
    def _augment_s1(self, s1_asc_bands, s1_asc_doys, s1_asc_masks, s1_desc_bands, s1_desc_doys, s1_desc_masks):
        """处理S1数据为模型输入，更好地处理无效数据"""
        # 处理S1数据缺失的情况
        s1_bands_all = []
        s1_doys_all = []
        
        if s1_asc_bands is not None and len(s1_asc_bands) > 0:
            s1_bands_all.append(s1_asc_bands)
            if s1_asc_doys is not None:
                s1_doys_all.append(s1_asc_doys)
        
        if s1_desc_bands is not None and len(s1_desc_bands) > 0:
            s1_bands_all.append(s1_desc_bands)
            if s1_desc_doys is not None:
                s1_doys_all.append(s1_desc_doys)
        
        # 如果没有有效数据，返回零
        if not s1_bands_all:
            result = np.zeros((self.max_seq_len_s1, 3), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
        
        try:
            s1_bands_all = np.concatenate(s1_bands_all, axis=0)
            s1_doys_all = np.concatenate(s1_doys_all, axis=0) if s1_doys_all else np.zeros(s1_bands_all.shape[0])
        except Exception as e:
            print(f"合并S1数据时出错: {e}")
            result = np.zeros((self.max_seq_len_s1, 3), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
        
        # 找到有效索引
        valid_mask = np.any(s1_bands_all != 0, axis=-1)
        valid_idx = np.nonzero(valid_mask)[0]
        
        if len(valid_idx) == 0:
            # 如果没有有效数据，创建零张量
            result = np.zeros((self.max_seq_len_s1, 3), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
        
        # 采样索引
        if len(valid_idx) < self.max_seq_len_s1:
            sampled_idx = np.random.choice(valid_idx, size=self.max_seq_len_s1, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.max_seq_len_s1, replace=False)
        sampled_idx = np.sort(sampled_idx)
        
        # 获取波段和doys
        sub_bands = s1_bands_all[sampled_idx, :]
        sub_doys = s1_doys_all[sampled_idx] if len(s1_doys_all) > 0 else np.zeros(len(sampled_idx))
        
        # 标准化
        if self.standardize:
            sub_bands = (sub_bands - self.s1_band_mean) / (self.s1_band_std + 1e-9)
        
        # 组合波段和doys
        result = np.hstack((sub_bands, sub_doys.reshape(-1, 1)))
        return torch.tensor(result, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sample_locations)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        try:
            sample = self.sample_locations[idx]
            mgrs_tile = sample['mgrs_tile']
            x = sample['x']
            y = sample['y']
            
            # 高效提取点数据
            point_data = self._extract_point_data(mgrs_tile, x, y)
            
            # 处理数据
            s2_sample = self._augment_s2(point_data.get('s2_bands'), point_data.get('s2_masks'), point_data.get('s2_doys'))
            s1_sample = self._augment_s1(
                point_data.get('s1_asc_bands'), point_data.get('s1_asc_doys'), point_data.get('s1_asc_masks'),
                point_data.get('s1_desc_bands'), point_data.get('s1_desc_doys'), point_data.get('s1_desc_masks')
            )
            
            # 获取标签
            label = sample['label']
            
            return {
                's2_sample': s2_sample,
                's1_sample': s1_sample,
                'label': label
            }
        except Exception as e:
            print(f"处理样本{idx}时出错: {e}")
            # 返回默认值
            return {
                's2_sample': torch.zeros((self.max_seq_len_s2, 11), dtype=torch.float32),
                's1_sample': torch.zeros((self.max_seq_len_s1, 3), dtype=torch.float32),
                'label': sample['label'] if 'label' in sample else 0
            }


def ethiopia_crop_collate_fn(batch):
    """埃塞俄比亚数据集的整理函数"""
    s2_samples = [item['s2_sample'] for item in batch]
    s1_samples = [item['s1_sample'] for item in batch]
    labels = [item['label'] for item in batch]
    
    s2_samples = torch.stack(s2_samples, dim=0)
    s1_samples = torch.stack(s1_samples, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return s2_samples, s1_samples, labels



class AustrianCropSelectByClass(Dataset):
    """
    Dataset for downstream classification with pixel selection based on class counts.
    Selects a specified number of pixels per class, rather than using field IDs.
    """
    def __init__(self,
                 s2_bands_file_path,
                 s2_masks_file_path,
                 s2_doy_file_path,
                 s1_asc_bands_file_path,
                 s1_asc_doy_file_path,
                 s1_desc_bands_file_path,
                 s1_desc_doy_file_path,
                 labels_path,
                 train_indices,
                 val_indices,
                 test_indices,
                 split='train',
                 shuffle=True,
                 is_training=True,
                 standardize=True,
                 min_valid_timesteps=0,
                 sample_size_s2=20,
                 sample_size_s1=20):
        """
        Args:
            s2_bands_file_path: Path to Sentinel-2 bands data (numpy array)
            s2_masks_file_path: Path to Sentinel-2 masks data (numpy array)
            s2_doy_file_path: Path to Sentinel-2 day-of-year data (numpy array)
            s1_asc_bands_file_path: Path to Sentinel-1 ascending bands data (numpy array)
            s1_asc_doy_file_path: Path to Sentinel-1 ascending day-of-year data (numpy array)
            s1_desc_bands_file_path: Path to Sentinel-1 descending bands data (numpy array)
            s1_desc_doy_file_path: Path to Sentinel-1 descending day-of-year data (numpy array)
            labels_path: Path to labels data (numpy array)
            train_indices: List of (y, x) coordinates for training
            val_indices: List of (y, x) coordinates for validation
            test_indices: List of (y, x) coordinates for testing
            split: One of 'train', 'val', or 'test'
            shuffle: Whether to shuffle the indices
            is_training: Whether in training mode
            standardize: Whether to standardize the data
            min_valid_timesteps: Minimum number of valid timesteps required
            sample_size_s2: Number of S2 timesteps to sample
            sample_size_s1: Number of S1 timesteps to sample
        """
        super().__init__()
        self.s2_bands_data = np.load(s2_bands_file_path)
        self.s2_masks_data = np.load(s2_masks_file_path)
        self.s2_doys_data = np.load(s2_doy_file_path)
        self.s1_asc_bands_data = np.load(s1_asc_bands_file_path)
        self.s1_asc_doys_data = np.load(s1_asc_doy_file_path)
        self.s1_desc_bands_data = np.load(s1_desc_bands_file_path)
        self.s1_desc_doys_data = np.load(s1_desc_doy_file_path)
        
        # Generate masks for S1 data: check if there's any valid data
        self.s1_asc_masks_data = (self.s1_asc_bands_data.sum(axis=-1) != 0)
        self.s1_desc_masks_data = (self.s1_desc_bands_data.sum(axis=-1) != 0)

        self.labels = np.load(labels_path)
        
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        self.split = split
        self.shuffle = shuffle
        self.is_training = is_training
        self.standardize = standardize
        
        self.min_valid_timesteps = min_valid_timesteps
        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1
        
        # Select indices based on split
        if self.split == 'train':
            self.valid_pixels = self.train_indices
        elif self.split == 'val':
            self.valid_pixels = self.val_indices
        elif self.split == 'test':
            self.valid_pixels = self.test_indices
        else:
            raise ValueError(f"Invalid split: {self.split}")
            
        # Filter pixels based on minimum valid timesteps
        if self.min_valid_timesteps > 0:
            filtered_pixels = []
            for i, j in self.valid_pixels:
                s2_valid_count = self.s2_masks_data[:, i, j].sum()
                s1_asc_valid_count = self.s1_asc_masks_data[:, i, j].sum()
                s1_desc_valid_count = self.s1_desc_masks_data[:, i, j].sum()
                if (s2_valid_count >= self.min_valid_timesteps) and ((s1_asc_valid_count + s1_desc_valid_count) >= self.min_valid_timesteps):
                    filtered_pixels.append((i, j))
            self.valid_pixels = filtered_pixels
            
        if self.shuffle:
            np.random.shuffle(self.valid_pixels)
        
        # Set mean and std for standardization
        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std = S1_BAND_STD
        
        logging.info(f"AustrianCropSelectByClass {split} set: {len(self.valid_pixels)} pixels")

    def __len__(self):
        return len(self.valid_pixels)

    def _augment_s2(self, s2_bands, s2_masks, s2_doys):
        """Process S2 data by sampling and adding DOY information"""
        valid_idx = np.nonzero(s2_masks)[0]
        if len(valid_idx) < self.sample_size_s2:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s2, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s2, replace=False)
        sampled_idx = np.sort(sampled_idx)
        sub_bands = s2_bands[sampled_idx, :]
        sub_doys = s2_doys[sampled_idx]
        
        if self.standardize:
            sub_bands = (sub_bands - self.s2_band_mean) / (self.s2_band_std + 1e-9)
        
        # Append DOY information to bands - critical for model compatibility
        result = np.hstack((sub_bands, sub_doys.reshape(-1, 1)))
        return torch.tensor(result, dtype=torch.float32)

    def _augment_s1(self, s1_asc_bands, s1_asc_doys, s1_desc_bands, s1_desc_doys):
        """Process S1 data by combining ascending and descending passes and adding DOY information"""
        s1_bands_all = np.concatenate([s1_asc_bands, s1_desc_bands], axis=0)
        s1_doys_all = np.concatenate([s1_asc_doys, s1_desc_doys], axis=0)
        valid_mask = np.any(s1_bands_all != 0, axis=-1)
        valid_idx = np.nonzero(valid_mask)[0]
        
        if len(valid_idx) < self.sample_size_s1:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s1, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.sample_size_s1, replace=False)
        
        sampled_idx = np.sort(sampled_idx)
        sub_bands = s1_bands_all[sampled_idx, :]
        sub_doys = s1_doys_all[sampled_idx]
        
        if self.standardize:
            sub_bands = (sub_bands - self.s1_band_mean) / (self.s1_band_std + 1e-9)
        
        # Append DOY information to bands - critical for model compatibility
        result = np.hstack((sub_bands, sub_doys.reshape(-1, 1)))
        return torch.tensor(result, dtype=torch.float32)

    def __getitem__(self, idx):
        i, j = self.valid_pixels[idx]
        s2_bands_ij = self.s2_bands_data[:, i, j, :]
        s2_masks_ij = self.s2_masks_data[:, i, j]
        s2_doys_ij = self.s2_doys_data

        s1_asc_bands_ij = self.s1_asc_bands_data[:, i, j, :]
        s1_asc_doys_ij = self.s1_asc_doys_data
        s1_desc_bands_ij = self.s1_desc_bands_data[:, i, j, :]
        s1_desc_doys_ij = self.s1_desc_doys_data

        s2_sample = self._augment_s2(s2_bands_ij, s2_masks_ij, s2_doys_ij)
        s1_sample = self._augment_s1(s1_asc_bands_ij, s1_asc_doys_ij,
                                     s1_desc_bands_ij, s1_desc_doys_ij)
        label = self.labels[i, j] - 1  # 0-based label
        return {
            's2_sample': s2_sample,
            's1_sample': s1_sample,
            'label': label
        }

def select_pixels_by_class(labels, pixels_per_class, val_test_split_ratio=0):
    """
    Select pixels for training, validation, and testing based on class counts.
    
    Args:
        labels: 2D numpy array of labels
        pixels_per_class: Number of pixels to select per class
        val_test_split_ratio: Ratio of validation to testing pixels (0-1)
            0 = all non-training pixels go to test set
            1 = all non-training pixels go to validation set
            
    Returns:
        train_indices, val_indices, test_indices: Lists of (y, x) coordinates
    """
    H, W = labels.shape
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Group pixel coordinates by class
    class_to_pixels = defaultdict(list)
    for y in range(H):
        for x in range(W):
            cls = labels[y, x]
            if cls > 0:  # Skip background class
                class_to_pixels[cls].append((y, x))
    
    # For each class, select training, validation, and test pixels
    for cls in sorted(class_to_pixels.keys()):
        pixels = class_to_pixels[cls]
        np.random.shuffle(pixels)
        
        # Take PIXELS_PER_CLASS for training
        pixels_needed = min(pixels_per_class, len(pixels))
        train_indices.extend(pixels[:pixels_needed])
        
        # Remaining pixels go to val or test
        remaining = pixels[pixels_needed:]
        
        if val_test_split_ratio == 0:
            # All remaining to test
            test_indices.extend(remaining)
        elif val_test_split_ratio == 1:
            # All remaining to val
            val_indices.extend(remaining)
        else:
            # Split according to ratio
            split_idx = int(len(remaining) * val_test_split_ratio)
            val_indices.extend(remaining[:split_idx])
            test_indices.extend(remaining[split_idx:])
    
    return train_indices, val_indices, test_indices

def austrian_crop_class_based_collate_fn(batch):
    """
    Collate function for AustrianCropSelectByClass.
    Stacks samples and returns tensors ready for model input.
    """
    s2_samples = [item['s2_sample'] for item in batch]
    s1_samples = [item['s1_sample'] for item in batch]
    labels = [item['label'] for item in batch]
    s2_samples = torch.stack(s2_samples, dim=0)
    s1_samples = torch.stack(s1_samples, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return s2_samples, s1_samples, labels


class EthiopiaInferenceDataset(Dataset):
    """
    Dataset for inference on Ethiopia crop classification data.
    Does not perform train/val split - just loads all samples to generate representations.
    """
    
    def __init__(self, 
                 polygons_locations_csv,
                 data_processed_dir,
                 data_raw_dir,
                 class_names,
                 standardize=True,
                 max_seq_len_s2=40,
                 max_seq_len_s1=40,
                 s2_band_mean=None,
                 s2_band_std=None,
                 s1_band_mean=None,
                 s1_band_std=None):
        """Initialize Ethiopia inference dataset."""
        super().__init__()
        start_time = time.time()
        print(f"Initializing inference dataset...")
        
        self.data_processed_dir = data_processed_dir
        self.data_raw_dir = data_raw_dir
        self.standardize = standardize
        self.max_seq_len_s2 = max_seq_len_s2
        self.max_seq_len_s1 = max_seq_len_s1
        self.class_names = class_names
        self.class_to_idx = {cls.lower(): i for i, cls in enumerate(class_names)}
        
        # Set normalization values
        self.s2_band_mean = s2_band_mean if s2_band_mean is not None else S2_BAND_MEAN
        self.s2_band_std = s2_band_std if s2_band_std is not None else S2_BAND_STD
        self.s1_band_mean = s1_band_mean if s1_band_mean is not None else S1_BAND_MEAN
        self.s1_band_std = s1_band_std if s1_band_std is not None else S1_BAND_STD
        
        # Load polygon-MGRS mapping CSV
        print("Loading CSV data...")
        self.df = pd.read_csv(polygons_locations_csv)
        
        # Filter out polygons with no matching MGRS tiles
        self.df = self.df[self.df['found_locations'] > 0]
        
        # Get class information
        if 'c_class' in self.df.columns:
            self.df['class_idx'] = self.df['c_class'].apply(lambda x: self.class_to_idx.get(x.lower(), -1))
            # Only keep rows with valid class indices
            self.df = self.df[self.df['class_idx'] != -1]
        else:
            raise ValueError("Cannot find c_class column in CSV")
        
        # Reset index to prevent out-of-range errors
        self.df = self.df.reset_index(drop=True)
        
        # Use all valid samples (no train/val split)
        self.indices = self.df.index.tolist()
        
        # Preprocess location information
        print("Preprocessing location information...")
        self.sample_locations = []
        for idx in self.indices:
            row = self.df.iloc[idx]
            locations = self._parse_locations(row['mgrs_locations'])
            if locations:
                self.sample_locations.append({
                    'idx': idx,
                    'mgrs_tile': locations[0][0],
                    'x': locations[0][1],
                    'y': locations[0][2],
                    'label': row['class_idx']
                })
        
        # Group by MGRS tile for efficient loading
        self.tile_to_samples = {}
        for sample in self.sample_locations:
            mgrs_tile = sample['mgrs_tile']
            if mgrs_tile not in self.tile_to_samples:
                self.tile_to_samples[mgrs_tile] = []
            self.tile_to_samples[mgrs_tile].append(sample)
        
        # Get data file information
        print("Checking data files...")
        self.data_info = {}
        for mgrs_tile in self.tile_to_samples.keys():
            s2_bands_path = os.path.join(self.data_processed_dir, mgrs_tile, "bands.npy")
            s2_masks_path = os.path.join(self.data_processed_dir, mgrs_tile, "masks.npy")
            s2_doys_path = os.path.join(self.data_processed_dir, mgrs_tile, "doys.npy")
            s1_asc_path = os.path.join(self.data_processed_dir, mgrs_tile, "sar_ascending.npy")
            s1_asc_doy_path = os.path.join(self.data_processed_dir, mgrs_tile, "sar_ascending_doy.npy")
            s1_desc_path = os.path.join(self.data_processed_dir, mgrs_tile, "sar_descending.npy")
            s1_desc_doy_path = os.path.join(self.data_processed_dir, mgrs_tile, "sar_descending_doy.npy")
            
            self.data_info[mgrs_tile] = {
                's2_bands_path': s2_bands_path if os.path.exists(s2_bands_path) else None,
                's2_masks_path': s2_masks_path if os.path.exists(s2_masks_path) else None,
                's2_doys_path': s2_doys_path if os.path.exists(s2_doys_path) else None,
                's1_asc_path': s1_asc_path if os.path.exists(s1_asc_path) else None,
                's1_asc_doy_path': s1_asc_doy_path if os.path.exists(s1_asc_doy_path) else None,
                's1_desc_path': s1_desc_path if os.path.exists(s1_desc_path) else None,
                's1_desc_doy_path': s1_desc_doy_path if os.path.exists(s1_desc_doy_path) else None,
            }
        
        # Preload all doys data (they're small)
        print("Preloading doys data...")
        self.doys_cache = {}
        for mgrs_tile, info in self.data_info.items():
            if info['s2_doys_path'] and os.path.exists(info['s2_doys_path']):
                try:
                    self.doys_cache[f"{mgrs_tile}_s2"] = np.load(info['s2_doys_path'])
                except Exception as e:
                    print(f"Error loading S2 DOY data for {mgrs_tile}: {e}")
                    
            if info['s1_asc_doy_path'] and os.path.exists(info['s1_asc_doy_path']):
                try:
                    self.doys_cache[f"{mgrs_tile}_s1_asc"] = np.load(info['s1_asc_doy_path'])
                except Exception as e:
                    print(f"Error loading S1 ascending DOY data for {mgrs_tile}: {e}")
                    
            if info['s1_desc_doy_path'] and os.path.exists(info['s1_desc_doy_path']):
                try:
                    self.doys_cache[f"{mgrs_tile}_s1_desc"] = np.load(info['s1_desc_doy_path'])
                except Exception as e:
                    print(f"Error loading S1 descending DOY data for {mgrs_tile}: {e}")
        
        # Data cache - only store extracted point data
        self.data_cache = {}
        
        print(f"Dataset initialization completed in {time.time() - start_time:.2f}s")
    
    def _parse_locations(self, mgrs_locations_str):
        """Parse MGRS location string to list of tuples [(mgrs_tile, x, y), ...]"""
        locations = []
        if isinstance(mgrs_locations_str, str):
            loc_strs = mgrs_locations_str.split('; ')
            for loc_str in loc_strs:
                parts = loc_str.split(':')
                if len(parts) == 2:
                    mgrs_tile = parts[0]
                    coords = parts[1].strip('()').split(',')
                    if len(coords) == 2:
                        x, y = int(coords[0]), int(coords[1])
                        locations.append((mgrs_tile, x, y))
        return locations
    
    def _extract_single_pixel(self, mmap_array, y, x, default_shape=None):
        """Safely extract single pixel data from a memory-mapped array"""
        try:
            if mmap_array is None:
                return None
            # Extract data
            if default_shape is None:
                return mmap_array[:, y, x, :].copy()
            else:
                return mmap_array[:, y, x, :].copy() if len(mmap_array.shape) > 3 else mmap_array[:, y, x].copy()
        except Exception as e:
            print(f"Error extracting pixel ({y},{x}): {e}")
            return None
    
    def _extract_point_data(self, mgrs_tile, x, y):
        """Efficiently extract all data for a single point, with better error handling"""
        # Check cache
        cache_key = f"{mgrs_tile}_{x}_{y}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        info = self.data_info[mgrs_tile]
        result = {}
        
        # First step: Try to determine array shape first, avoid loading the entire array
        try:
            # Get S2 data
            if info['s2_bands_path']:
                try:
                    # Load a small fragment to get shape information
                    with open(info['s2_bands_path'], 'rb') as f:
                        # Read NPY file header
                        version = np.lib.format.read_magic(f)
                        shape, _, _ = np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f)
                    
                    # Load using memory mapping
                    s2_bands_mmap = np.load(info['s2_bands_path'], mmap_mode='r')
                    # Only extract data at specific location and copy to memory
                    result['s2_bands'] = self._extract_single_pixel(s2_bands_mmap, y, x)
                    # Manual cleanup
                    del s2_bands_mmap
                except Exception as e:
                    print(f"Error extracting S2 band data for {mgrs_tile}: {e}")
                    result['s2_bands'] = None
            else:
                result['s2_bands'] = None
                
            if info['s2_masks_path']:
                try:
                    s2_masks_mmap = np.load(info['s2_masks_path'], mmap_mode='r')
                    result['s2_masks'] = s2_masks_mmap[:, y, x].copy()
                    del s2_masks_mmap
                except Exception as e:
                    print(f"Error extracting S2 mask data for {mgrs_tile}: {e}")
                    # If there's band data but no mask, assume all timesteps are valid
                    if result['s2_bands'] is not None:
                        result['s2_masks'] = np.ones(result['s2_bands'].shape[0], dtype=bool)
                    else:
                        result['s2_masks'] = None
            else:
                # If no mask file but have band data, assume all timesteps are valid
                if result['s2_bands'] is not None:
                    result['s2_masks'] = np.ones(result['s2_bands'].shape[0], dtype=bool)
                else:
                    result['s2_masks'] = None
            
            # Get S2 DOY data
            result['s2_doys'] = self.doys_cache.get(f"{mgrs_tile}_s2", None)
        
            # Get S1 ascending data
            if info['s1_asc_path']:
                try:
                    s1_asc_mmap = np.load(info['s1_asc_path'], mmap_mode='r')
                    result['s1_asc_bands'] = self._extract_single_pixel(s1_asc_mmap, y, x)
                    del s1_asc_mmap
                    
                    # Generate mask
                    if result['s1_asc_bands'] is not None:
                        result['s1_asc_masks'] = np.any(result['s1_asc_bands'] != 0, axis=1)
                    else:
                        result['s1_asc_masks'] = None
                        
                    result['s1_asc_doys'] = self.doys_cache.get(f"{mgrs_tile}_s1_asc", None)
                except Exception as e:
                    print(f"Error extracting S1 ascending data for {mgrs_tile}: {e}")
                    result['s1_asc_bands'] = None
                    result['s1_asc_masks'] = None
                    result['s1_asc_doys'] = None
            else:
                result['s1_asc_bands'] = None
                result['s1_asc_masks'] = None
                result['s1_asc_doys'] = None
            
            # Get S1 descending data
            if info['s1_desc_path']:
                try:
                    s1_desc_mmap = np.load(info['s1_desc_path'], mmap_mode='r')
                    result['s1_desc_bands'] = self._extract_single_pixel(s1_desc_mmap, y, x)
                    del s1_desc_mmap
                    
                    # Generate mask
                    if result['s1_desc_bands'] is not None:
                        result['s1_desc_masks'] = np.any(result['s1_desc_bands'] != 0, axis=1)
                    else:
                        result['s1_desc_masks'] = None
                        
                    result['s1_desc_doys'] = self.doys_cache.get(f"{mgrs_tile}_s1_desc", None)
                except Exception as e:
                    print(f"Error extracting S1 descending data for {mgrs_tile}: {e}")
                    result['s1_desc_bands'] = None
                    result['s1_desc_masks'] = None
                    result['s1_desc_doys'] = None
            else:
                result['s1_desc_bands'] = None
                result['s1_desc_masks'] = None
                result['s1_desc_doys'] = None
                
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Error processing data for {mgrs_tile}: {e}")
            result = {
                's2_bands': None,
                's2_masks': None,
                's2_doys': None,
                's1_asc_bands': None,
                's1_asc_doys': None,
                's1_asc_masks': None,
                's1_desc_bands': None,
                's1_desc_doys': None,
                's1_desc_masks': None
            }
        
        # Store in cache
        self.data_cache[cache_key] = result
        return result
    
    def _augment_s2(self, s2_bands, s2_masks, s2_doys):
        """Process S2 data for model input, better handling of invalid data"""
        # Check if there's valid data
        if s2_bands is None or len(s2_bands) == 0:
            # If no valid data, create zero tensor
            result = np.zeros((self.max_seq_len_s2, 11), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
            
        # Find valid indices where mask is True
        valid_idx = np.nonzero(s2_masks)[0] if s2_masks is not None else np.arange(len(s2_bands))
        
        if len(valid_idx) == 0:
            # If no valid data, create zero tensor
            result = np.zeros((self.max_seq_len_s2, 11), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
        
        # Sample indices
        if len(valid_idx) < self.max_seq_len_s2:
            sampled_idx = np.random.choice(valid_idx, size=self.max_seq_len_s2, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.max_seq_len_s2, replace=False)
        sampled_idx = np.sort(sampled_idx)
        
        # Get bands and doys
        sub_bands = s2_bands[sampled_idx, :]
        sub_doys = s2_doys[sampled_idx] if s2_doys is not None else np.zeros(len(sampled_idx))
        
        # Normalize
        if self.standardize:
            sub_bands = (sub_bands - self.s2_band_mean) / (self.s2_band_std + 1e-9)
        
        # Combine bands and doys
        result = np.hstack((sub_bands, sub_doys.reshape(-1, 1)))
        return torch.tensor(result, dtype=torch.float32)
    
    def _augment_s1(self, s1_asc_bands, s1_asc_doys, s1_asc_masks, s1_desc_bands, s1_desc_doys, s1_desc_masks):
        """Process S1 data for model input, better handling of invalid data"""
        # Handle missing S1 data
        s1_bands_all = []
        s1_doys_all = []
        
        if s1_asc_bands is not None and len(s1_asc_bands) > 0:
            s1_bands_all.append(s1_asc_bands)
            if s1_asc_doys is not None:
                s1_doys_all.append(s1_asc_doys)
        
        if s1_desc_bands is not None and len(s1_desc_bands) > 0:
            s1_bands_all.append(s1_desc_bands)
            if s1_desc_doys is not None:
                s1_doys_all.append(s1_desc_doys)
        
        # If no valid data, return zeros
        if not s1_bands_all:
            result = np.zeros((self.max_seq_len_s1, 3), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
        
        try:
            s1_bands_all = np.concatenate(s1_bands_all, axis=0)
            s1_doys_all = np.concatenate(s1_doys_all, axis=0) if s1_doys_all else np.zeros(s1_bands_all.shape[0])
        except Exception as e:
            print(f"Error merging S1 data: {e}")
            result = np.zeros((self.max_seq_len_s1, 3), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
        
        # Find valid indices
        valid_mask = np.any(s1_bands_all != 0, axis=-1)
        valid_idx = np.nonzero(valid_mask)[0]
        
        if len(valid_idx) == 0:
            # If no valid data, create zero tensor
            result = np.zeros((self.max_seq_len_s1, 3), dtype=np.float32)
            return torch.tensor(result, dtype=torch.float32)
        
        # Sample indices
        if len(valid_idx) < self.max_seq_len_s1:
            sampled_idx = np.random.choice(valid_idx, size=self.max_seq_len_s1, replace=True)
        else:
            sampled_idx = np.random.choice(valid_idx, size=self.max_seq_len_s1, replace=False)
        sampled_idx = np.sort(sampled_idx)
        
        # Get bands and doys
        sub_bands = s1_bands_all[sampled_idx, :]
        sub_doys = s1_doys_all[sampled_idx] if len(s1_doys_all) > 0 else np.zeros(len(sampled_idx))
        
        # Normalize
        if self.standardize:
            sub_bands = (sub_bands - self.s1_band_mean) / (self.s1_band_std + 1e-9)
        
        # Combine bands and doys
        result = np.hstack((sub_bands, sub_doys.reshape(-1, 1)))
        return torch.tensor(result, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sample_locations)
    
    def __getitem__(self, idx):
        """Get data sample"""
        try:
            sample = self.sample_locations[idx]
            mgrs_tile = sample['mgrs_tile']
            x = sample['x']
            y = sample['y']
            
            # Extract point data efficiently
            point_data = self._extract_point_data(mgrs_tile, x, y)
            
            # Process data
            s2_sample = self._augment_s2(point_data.get('s2_bands'), point_data.get('s2_masks'), point_data.get('s2_doys'))
            s1_sample = self._augment_s1(
                point_data.get('s1_asc_bands'), point_data.get('s1_asc_doys'), point_data.get('s1_asc_masks'),
                point_data.get('s1_desc_bands'), point_data.get('s1_desc_doys'), point_data.get('s1_desc_masks')
            )
            
            # Get label
            label = sample['label']
            
            return s2_sample, s1_sample, label
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return default values
            return (
                torch.zeros((self.max_seq_len_s2, 11), dtype=torch.float32),
                torch.zeros((self.max_seq_len_s1, 3), dtype=torch.float32),
                sample['label'] if 'label' in sample else 0
            )

def ethiopia_crop_infer_fn(batch):
    """Collation function for inference"""
    s2_samples = [item[0] for item in batch]
    s1_samples = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    s2_samples = torch.stack(s2_samples, dim=0)
    s1_samples = torch.stack(s1_samples, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return s2_samples, s1_samples, labels


