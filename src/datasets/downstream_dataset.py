# src/datasets/downstream_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

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
        doys_norm = sub_doys / 365.0
        sin_doy = np.sin(2*np.pi*doys_norm).reshape(-1,1)
        cos_doy = np.cos(2*np.pi*doys_norm).reshape(-1,1)
        result = np.hstack((sub_bands, sin_doy, cos_doy))
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
        doys_norm = sub_doys / 365.0
        sin_doy = np.sin(2*np.pi*doys_norm).reshape(-1,1)
        cos_doy = np.cos(2*np.pi*doys_norm).reshape(-1,1)
        result = np.hstack((sub_bands, sin_doy, cos_doy))
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