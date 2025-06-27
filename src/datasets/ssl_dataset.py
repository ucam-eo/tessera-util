# src/datasets/ssl_dataset.py

import os
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info, Dataset
import logging

import multiprocessing
import os
import pickle
from tqdm import tqdm
import threading
from joblib import Parallel, delayed
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 均值和方差
S2_BAND_MEAN = np.array([1711.0938,1308.8511,1546.4543,3010.1293,3106.5083,
                        2068.3044,2685.0845,2931.5889,2514.6928,1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026,1862.9751,1803.1792,1741.7837,1677.4543,
                        1888.7862,1736.3090,1715.8104,1514.5199,1398.4779], dtype=np.float32)
S1_BAND_MEAN = np.array([5484.0407,3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334,1726.0670], dtype=np.float32)

class HDF5Dataset_Multimodal_Tiles_Iterable(IterableDataset):
    """
    预处理数据加载：读取预处理好的 .npy 文件，目录结构为：
       data_root/
         ├─ aug1/
         │    ├─ s2/   -> 每个 .npy 文件形状大致为 (N, 20, 12)
         │    └─ s1/   -> 每个 .npy 文件形状大致为 (N, 20, 4)
         └─ aug2/
              ├─ s2/
              └─ s1/
    每个文件组中的四个 .npy 文件分别对应 s2_aug1, s2_aug2, s1_aug1, s1_aug2，
    yield 单个样本。
    """
    def __init__(self,
                 data_root,
                 min_valid_timesteps=10,
                 sample_size_s2=20,
                 sample_size_s1=20,
                 standardize=True,
                 shuffle_tiles=False):
        super().__init__()
        self.data_root = data_root
        self.min_valid_timesteps = min_valid_timesteps
        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1
        self.standardize = standardize
        self.shuffle_tiles = shuffle_tiles

        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std = S1_BAND_STD

        # 构建数据路径
        self.aug1_s2_dir = os.path.join(data_root, "aug1", "s2")
        self.aug2_s2_dir = os.path.join(data_root, "aug2", "s2")
        self.aug1_s1_dir = os.path.join(data_root, "aug1", "s1")
        self.aug2_s1_dir = os.path.join(data_root, "aug2", "s1")
        for d in [self.aug1_s2_dir, self.aug2_s2_dir, self.aug1_s1_dir, self.aug2_s1_dir]:
            if not os.path.exists(d):
                raise RuntimeError(f"Directory {d} not found!")
        file_names = sorted(os.listdir(self.aug1_s2_dir))
        np.random.shuffle(file_names)
        self.file_groups = []
        for fn in file_names:
            file_path_aug1_s2 = os.path.join(self.aug1_s2_dir, fn)
            file_path_aug2_s2 = os.path.join(self.aug2_s2_dir, fn)
            file_path_aug1_s1 = os.path.join(self.aug1_s1_dir, fn)
            file_path_aug2_s1 = os.path.join(self.aug2_s1_dir, fn)
            if os.path.exists(file_path_aug2_s2) and os.path.exists(file_path_aug1_s1) and os.path.exists(file_path_aug2_s1):
                self.file_groups.append({
                    "s2_aug1": file_path_aug1_s2,
                    "s2_aug2": file_path_aug2_s2,
                    "s1_aug1": file_path_aug1_s1,
                    "s1_aug2": file_path_aug2_s1
                })
        if len(self.file_groups) == 0:
            raise RuntimeError("No valid file groups found in preprocessed dataset!")

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            groups_to_process = self.file_groups
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(self.file_groups) / num_workers))
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_groups))
            groups_to_process = self.file_groups[start:end]
        if self.shuffle_tiles:
            np.random.shuffle(groups_to_process)
        for group in groups_to_process:
            s2_aug1_array = np.load(group["s2_aug1"])
            s2_aug2_array = np.load(group["s2_aug2"])
            s1_aug1_array = np.load(group["s1_aug1"])
            s1_aug2_array = np.load(group["s1_aug2"])
            n_samples = min(s2_aug1_array.shape[0],
                            s2_aug2_array.shape[0],
                            s1_aug1_array.shape[0],
                            s1_aug2_array.shape[0])
            for i in range(n_samples):
                yield {
                    "s2_aug1": torch.tensor(s2_aug1_array[i], dtype=torch.float32),
                    "s2_aug2": torch.tensor(s2_aug2_array[i], dtype=torch.float32),
                    "s1_aug1": torch.tensor(s1_aug1_array[i], dtype=torch.float32),
                    "s1_aug2": torch.tensor(s1_aug2_array[i], dtype=torch.float32)
                }


class AustrianCropValidation(Dataset):
    """
    用于验证集评估。与之前提供的版本一致。
    不做两次增强，只返回 (s2_sample, s1_sample, label)。
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
                 sample_size_s2=20,
                 sample_size_s1=20,
                 min_valid_timesteps=0,
                 standardize=True):
        super().__init__()
        self.s2_bands_data = np.load(s2_bands_file_path)
        self.s2_masks_data = np.load(s2_masks_file_path)
        self.s2_doys_data  = np.load(s2_doy_file_path)
        self.s1_asc_bands_data = np.load(s1_asc_bands_file_path)
        self.s1_asc_doys_data  = np.load(s1_asc_doy_file_path)
        self.s1_desc_bands_data= np.load(s1_desc_bands_file_path)
        self.s1_desc_doys_data = np.load(s1_desc_doy_file_path)
        self.labels = np.load(labels_path)

        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1
        self.standardize = standardize
        self.min_valid_timesteps = min_valid_timesteps
        
        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std = S1_BAND_STD

        t_s2, H, W, _ = self.s2_bands_data.shape
        indices = np.indices((H, W)).reshape(2, -1).T

        self.valid_pixels = []
        for (i, j) in indices:
            if self.labels[i, j] == 0:
                continue
            s2_valid = self.s2_masks_data[:, i, j].sum()
            asc_valid = np.any(self.s1_asc_bands_data[:, i, j, :] != 0, axis=-1).sum()
            desc_valid = np.any(self.s1_desc_bands_data[:, i, j, :] != 0, axis=-1).sum()
            if s2_valid >= self.min_valid_timesteps and (asc_valid + desc_valid) >= self.min_valid_timesteps:
                self.valid_pixels.append((i, j))

    def __len__(self):
        return len(self.valid_pixels)

    def _augment_s2(self, s2_bands, s2_masks, s2_doys):
        valid_idx = np.nonzero(s2_masks)[0]
        size = self.sample_size_s2
        if len(valid_idx) < size:
            idx_chosen = np.random.choice(valid_idx, size=size, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=size, replace=False)
        idx_chosen = np.sort(idx_chosen)

        sub_bands = s2_bands[idx_chosen, :]
        sub_doys  = s2_doys[idx_chosen]

        if self.standardize:
            sub_bands = (sub_bands - self.s2_band_mean) / (self.s2_band_std + 1e-9)

        # doys_norm = sub_doys / 365.
        # sin_doy = np.sin(2 * np.pi * doys_norm).reshape(-1, 1)
        # cos_doy = np.cos(2 * np.pi * doys_norm).reshape(-1, 1)
        # arr = np.hstack([sub_bands, sin_doy, cos_doy])
        
        #直接把doy加入到bands中
        arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])
        return torch.tensor(arr, dtype=torch.float32)

    def _augment_s1(self, asc_bands_ij, asc_doys, desc_bands_ij, desc_doys):
        asc_valid_idx = np.nonzero(np.any(asc_bands_ij != 0, axis=-1))[0]
        desc_valid_idx = np.nonzero(np.any(desc_bands_ij != 0, axis=-1))[0]
        asc_bands = asc_bands_ij[asc_valid_idx, :]
        asc_d = asc_doys[asc_valid_idx]
        desc_bands = desc_bands_ij[desc_valid_idx, :]
        desc_d = desc_doys[desc_valid_idx]

        s1_bands_all = np.concatenate([asc_bands, desc_bands], axis=0)
        s1_doys_all = np.concatenate([asc_d, desc_d], axis=0)
        if len(s1_bands_all) == 0:
            arr = np.zeros((self.sample_size_s1, 4), dtype=np.float32)
            return torch.tensor(arr, dtype=torch.float32)

        if len(s1_bands_all) < self.sample_size_s1:
            idx_chosen = np.random.choice(len(s1_bands_all), size=self.sample_size_s1, replace=True)
        else:
            idx_chosen = np.random.choice(len(s1_bands_all), size=self.sample_size_s1, replace=False)
        idx_chosen = np.sort(idx_chosen)

        sub_bands = s1_bands_all[idx_chosen, :]
        sub_d = s1_doys_all[idx_chosen]

        if self.standardize:
            sub_bands = (sub_bands - self.s1_band_mean) / (self.s1_band_std + 1e-9)

        # doys_norm = sub_d / 365.
        # sin_doy = np.sin(2 * np.pi * doys_norm).reshape(-1, 1)
        # cos_doy = np.cos(2 * np.pi * doys_norm).reshape(-1, 1)
        # arr = np.hstack([sub_bands, sin_doy, cos_doy])
        #直接把doy加入到bands中
        arr = np.hstack([sub_bands, sub_d.reshape(-1, 1)])
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx):
        i, j = self.valid_pixels[idx]
        label = self.labels[i, j] - 1

        s2_bands_ij = self.s2_bands_data[:, i, j, :]
        s2_masks_ij = self.s2_masks_data[:, i, j]
        s2_doys_ij  = self.s2_doys_data

        asc_bands_ij = self.s1_asc_bands_data[:, i, j, :]
        asc_doys_ij = self.s1_asc_doys_data
        desc_bands_ij = self.s1_desc_bands_data[:, i, j, :]
        desc_doys_ij = self.s1_desc_doys_data

        s2_sample = self._augment_s2(s2_bands_ij, s2_masks_ij, s2_doys_ij)
        s1_sample = self._augment_s1(asc_bands_ij, asc_doys_ij, desc_bands_ij, desc_doys_ij)
        return s2_sample, s1_sample, label
    
class SingleTileInferenceDataset(Dataset):
    """
    用于单Tile推理的数据集，仅返回像素的 (i, j) 以及对应的 s2/s1 数据。
    在 __getitem__ 里不做随机采样，因为要重复多次随机采样(10次)后取平均。
    """
    def __init__(self,
                 tile_path,
                 min_valid_timesteps=10,
                 standardize=True):
        super().__init__()
        
        # self.tile_path = tile_path
        # 加一个/data_processed后缀
        self.tile_path = os.path.join(tile_path, "data_processed")
        self.min_valid_timesteps = min_valid_timesteps
        self.standardize = standardize

        # 加载 S2
        s2_bands_path = os.path.join(self.tile_path, "bands.npy")    # (t_s2, H, W, 10)
        s2_masks_path = os.path.join(self.tile_path, "masks.npy")    # (t_s2, H, W)
        s2_doy_path   = os.path.join(self.tile_path, "doys.npy")     # (t_s2,)

        # >>> 修复UInt16类型: 转换为 int32 等通用类型 <
        self.s2_bands = np.load(s2_bands_path).astype(np.float32)
        self.s2_masks = np.load(s2_masks_path).astype(np.int32)
        self.s2_doys  = np.load(s2_doy_path).astype(np.int32)   # 避免uint16, 转成int32

        # 加载 S1 asc
        s1_asc_bands_path = os.path.join(self.tile_path, "sar_ascending.npy")      # (t_s1a, H, W, 2)
        s1_asc_doy_path   = os.path.join(self.tile_path, "sar_ascending_doy.npy")  # (t_s1a,)

        self.s1_asc_bands = np.load(s1_asc_bands_path).astype(np.float32)
        self.s1_asc_doys  = np.load(s1_asc_doy_path).astype(np.int32)

        # 加载 S1 desc
        s1_desc_bands_path = os.path.join(self.tile_path, "sar_descending.npy")      # (t_s1d, H, W, 2)
        s1_desc_doy_path   = os.path.join(self.tile_path, "sar_descending_doy.npy")  # (t_s1d,)

        self.s1_desc_bands = np.load(s1_desc_bands_path).astype(np.float32)
        self.s1_desc_doys  = np.load(s1_desc_doy_path).astype(np.int32)

        # 形状
        self.t_s2, self.H, self.W, _ = self.s2_bands.shape

        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std = S1_BAND_STD

        # 过滤像素
        self.valid_pixels = []
        ij_coords = np.indices((self.H, self.W)).reshape(2, -1).T
        for idx, (i, j) in enumerate(ij_coords):
            # s2 有效帧数
            s2_mask_ij = self.s2_masks[:, i, j]
            s2_valid = s2_mask_ij.sum()

            # s1 asc
            s1_asc_ij = self.s1_asc_bands[:, i, j, :]  # (t_s1a, 2)
            s1_asc_valid = np.any(s1_asc_ij != 0, axis=-1).sum()

            # s1 desc
            s1_desc_ij = self.s1_desc_bands[:, i, j, :]  # (t_s1d, 2)
            s1_desc_valid = np.any(s1_desc_ij != 0, axis=-1).sum()

            s1_total_valid = s1_asc_valid + s1_desc_valid
            
            # 新增: 检查S2频段是否所有值都为0
            s2_bands_ij = self.s2_bands[:, i, j, :]  # (t_s2, 10)
            s2_nonzero = np.any(s2_bands_ij != 0)  # 检查是否有任何非零值
            
            # 只有当S2频段有非零值且满足其他条件时，才保留该像素
            if s2_nonzero and (s2_valid >= self.min_valid_timesteps) and (s1_total_valid >= self.min_valid_timesteps):
                # 保留该像素
                self.valid_pixels.append((idx, i, j))

        logging.info(f"[SingleTileInferenceDataset] tile={tile_path}, total_valid_pixels={len(self.valid_pixels)}")

    def __len__(self):
        return len(self.valid_pixels)

    def __getitem__(self, index):
        global_idx, i, j = self.valid_pixels[index]

        # 将整个像素的 s2, s1 数据都取出
        s2_bands_ij = self.s2_bands[:, i, j, :]  # (t_s2, 10)
        s2_masks_ij = self.s2_masks[:, i, j]     # (t_s2,)
        s2_doys_ij  = self.s2_doys              # (t_s2,)

        s1_asc_bands_ij = self.s1_asc_bands[:, i, j, :]  # (t_s1a, 2)
        s1_asc_doys_ij  = self.s1_asc_doys               # (t_s1a,)

        s1_desc_bands_ij = self.s1_desc_bands[:, i, j, :]  # (t_s1d, 2)
        s1_desc_doys_ij  = self.s1_desc_doys               # (t_s1d,)

        sample = {
            "global_idx": global_idx,
            "i": i,
            "j": j,
            "s2_bands": s2_bands_ij,
            "s2_masks": s2_masks_ij,
            "s2_doys": s2_doys_ij,

            "s1_asc_bands": s1_asc_bands_ij,
            "s1_asc_doys": s1_asc_doys_ij,
            "s1_desc_bands": s1_desc_bands_ij,
            "s1_desc_doys": s1_desc_doys_ij,
        }
        return sample    

class PastisPatchInferenceDataset(Dataset):
    """
    用于单个 PASTIS patch 推理的数据集，
    读取 patch 文件夹下的 bands/masks/doys/sar_asc*/sar_desc* 等，
    并对每个像素返回 (i, j) 以及对应的时序数据 (无随机采样)。
    """

    def __init__(self,
                 patch_path,
                 min_valid_timesteps=0,
                 standardize=True):
        super().__init__()
        self.patch_path = patch_path
        self.min_valid_timesteps = min_valid_timesteps
        self.standardize = standardize

        # 加载 S2
        s2_bands_path = os.path.join(patch_path, "bands.npy")   # (T_s2, H, W, 10)
        s2_masks_path = os.path.join(patch_path, "masks.npy")   # (T_s2, H, W)
        s2_doy_path   = os.path.join(patch_path, "doys.npy")    # (T_s2,)

        self.s2_bands = np.load(s2_bands_path).astype(np.float32)
        self.s2_masks = np.load(s2_masks_path).astype(np.int32)
        self.s2_doys  = np.load(s2_doy_path).astype(np.int32)

        # 加载 S1 asc
        s1_asc_bands_path = os.path.join(patch_path, "sar_ascending.npy")      # (T_s1a, H, W, 2)
        s1_asc_doy_path   = os.path.join(patch_path, "sar_ascending_doy.npy")  # (T_s1a,)

        self.s1_asc_bands = np.load(s1_asc_bands_path).astype(np.float32)
        self.s1_asc_doys  = np.load(s1_asc_doy_path).astype(np.int32)

        # 加载 S1 desc
        s1_desc_bands_path = os.path.join(patch_path, "sar_descending.npy")      # (T_s1d, H, W, 2)
        s1_desc_doy_path   = os.path.join(patch_path, "sar_descending_doy.npy")  # (T_s1d,)

        self.s1_desc_bands = np.load(s1_desc_bands_path).astype(np.float32)
        self.s1_desc_doys  = np.load(s1_desc_doy_path).astype(np.int32)

        # 获取形状
        self.T_s2, self.H, self.W, _ = self.s2_bands.shape
        self.T_s1a = self.s1_asc_bands.shape[0]
        self.T_s1d = self.s1_desc_bands.shape[0]

        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std  = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std  = S1_BAND_STD

        # 收集 valid_pixels
        self.valid_pixels = []
        ij_coords = np.indices((self.H, self.W)).reshape(2, -1).T
        for idx, (i, j) in enumerate(ij_coords):
            # s2 有效帧
            s2_mask_ij = self.s2_masks[:, i, j]
            s2_valid = s2_mask_ij.sum()

            # s1 asc
            s1_asc_ij = self.s1_asc_bands[:, i, j, :]
            s1_asc_valid = np.any(s1_asc_ij != 0, axis=-1).sum()

            # s1 desc
            s1_desc_ij = self.s1_desc_bands[:, i, j, :]
            s1_desc_valid = np.any(s1_desc_ij != 0, axis=-1).sum()

            s1_total_valid = s1_asc_valid + s1_desc_valid

            if (s2_valid >= self.min_valid_timesteps) and (s1_total_valid >= self.min_valid_timesteps):
                self.valid_pixels.append((idx, i, j))

        logging.info(f"[PastisPatchInferenceDataset] patch={patch_path}, total_valid_pixels={len(self.valid_pixels)}")

    def __len__(self):
        return len(self.valid_pixels)

    def __getitem__(self, index):
        global_idx, i, j = self.valid_pixels[index]
        # 取出 s2
        s2_bands_ij = self.s2_bands[:, i, j, :]  # (T_s2, 10)
        s2_masks_ij = self.s2_masks[:, i, j]     # (T_s2,)
        s2_doys_ij  = self.s2_doys              # (T_s2,)

        # asc
        s1_asc_bands_ij = self.s1_asc_bands[:, i, j, :]  # (T_s1a, 2)
        s1_asc_doys_ij  = self.s1_asc_doys               # (T_s1a,)

        # desc
        s1_desc_bands_ij = self.s1_desc_bands[:, i, j, :] # (T_s1d, 2)
        s1_desc_doys_ij  = self.s1_desc_doys              # (T_s1d,)

        sample = {
            "global_idx": global_idx,
            "i": i,
            "j": j,
            "s2_bands": s2_bands_ij,
            "s2_masks": s2_masks_ij,
            "s2_doys": s2_doys_ij,

            "s1_asc_bands": s1_asc_bands_ij,
            "s1_asc_doys": s1_asc_doys_ij,
            "s1_desc_bands": s1_desc_bands_ij,
            "s1_desc_doys": s1_desc_doys_ij,
        }
        return sample

    @property
    def shape(self):
        """
        返回 (H, W)，用于外部构建输出数组
        """
        return (self.H, self.W)




def random_spatial_transform(patch):
    """
    对输入 patch (H, W, T, B) 进行随机旋转和翻转。
    注意：旋转与翻转仅在 H, W 维度上进行。
    """
    # 随机旋转 0, 90, 180, 270 度
    k = np.random.randint(0, 4)
    patch = np.rot90(patch, k, axes=(0, 1))  # 只在 (H, W) 维旋转

    # 随机上下翻转 (沿 H 维)
    if np.random.rand() < 0.5:
        patch = np.flip(patch, axis=0)

    # 随机左右翻转 (沿 W 维)
    if np.random.rand() < 0.5:
        patch = np.flip(patch, axis=1)

    return patch


class PatchDataset_Multimodal_Iterable(IterableDataset):
    """
    该数据集用于读取预处理后的 patch 级多模态数据，目录结构为：
       data_root/
         ├─ aug1/
         │    ├─ s2/   -> 每个 .npy 文件形状大致为 (N, H, W, T, B_s2)
         │    └─ s1/   -> 每个 .npy 文件形状大致为 (N, H, W, T, B_s1)
         └─ aug2/
              ├─ s2/
              └─ s1/

    其中，aug1/s2 和 aug2/s2 分别表示同一块区域但不同增强下的 Sentinel-2 数据；
    aug1/s1 和 aug2/s1 分别表示同一块区域但不同增强下的 Sentinel-1 数据。
    
    该数据集每次 yield 一个字典：
        {
            "s2_aug1": (H, W, T, B_s2),
            "s2_aug2": (H, W, T, B_s2),
            "s1_aug1": (H, W, T, B_s1),
            "s1_aug2": (H, W, T, B_s1)
        }
    同时，我们对 aug1 内部 (s2_aug1, s1_aug1) 和 aug2 内部 (s2_aug2, s1_aug2)
    分别进行随机旋转与翻转，以支持空间增强。
    """
    def __init__(self,
                 data_root,
                 shuffle_files=True,
                 shuffle_tiles=False):
        """
        Args:
            data_root (str): 数据存放的根目录。
            shuffle_files (bool): 是否在构建数据集时随机打乱文件顺序。
            shuffle_tiles (bool): 是否在迭代样本时对每个 worker 的文件分组再进行随机打乱。
        """
        super().__init__()
        self.data_root = data_root
        self.shuffle_tiles = shuffle_tiles
        
        # aug1 & aug2 的子目录
        self.aug1_s2_dir = os.path.join(data_root, "aug1", "s2")
        self.aug2_s2_dir = os.path.join(data_root, "aug2", "s2")
        self.aug1_s1_dir = os.path.join(data_root, "aug1", "s1")
        self.aug2_s1_dir = os.path.join(data_root, "aug2", "s1")
        
        # 检查目录是否存在
        for d in [self.aug1_s2_dir, self.aug2_s2_dir, self.aug1_s1_dir, self.aug2_s1_dir]:
            if not os.path.exists(d):
                raise RuntimeError(f"Directory {d} not found!")
                
        # 获取所有 aug1/s2 文件名（假设同名文件与 aug2/s2, aug1/s1, aug2/s1 对应）
        file_names = sorted(os.listdir(self.aug1_s2_dir))
        if shuffle_files:
            np.random.shuffle(file_names)
        
        # 构建文件组列表
        self.file_groups = []
        for fn in file_names:
            file_path_aug1_s2 = os.path.join(self.aug1_s2_dir, fn)
            file_path_aug2_s2 = os.path.join(self.aug2_s2_dir, fn)
            file_path_aug1_s1 = os.path.join(self.aug1_s1_dir, fn)
            file_path_aug2_s1 = os.path.join(self.aug2_s1_dir, fn)
            
            # 同时存在这四个对应文件才构成有效组
            if (os.path.exists(file_path_aug1_s2)
                and os.path.exists(file_path_aug2_s2)
                and os.path.exists(file_path_aug1_s1)
                and os.path.exists(file_path_aug2_s1)):
                self.file_groups.append({
                    "s2_aug1": file_path_aug1_s2,
                    "s2_aug2": file_path_aug2_s2,
                    "s1_aug1": file_path_aug1_s1,
                    "s1_aug2": file_path_aug2_s1
                })
        
        if len(self.file_groups) == 0:
            raise RuntimeError("No valid file groups found in dataset directories!")
    
    def __len__(self):
        """为了与 IterableDataset 保持兼容，这里返回文件组数，但不保证真实 sample 数。"""
        return len(self.file_groups)
    
    def __iter__(self):
        # 根据 worker 信息进行文件切分，保证多进程时不重复处理同一文件
        worker_info = get_worker_info()
        if worker_info is None:
            # 单进程
            groups_to_process = self.file_groups
        else:
            # 多进程时进行拆分
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(self.file_groups) / num_workers))
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_groups))
            groups_to_process = self.file_groups[start:end]
        
        # 是否在读取前再次 shuffle 文件组
        if self.shuffle_tiles:
            np.random.shuffle(groups_to_process)
        
        # 依次读取各文件组
        for group in groups_to_process:
            # 读取四份增广数据，形状大约为 (N, H, W, T, B)
            s2_aug1_array = np.load(group["s2_aug1"])
            s2_aug2_array = np.load(group["s2_aug2"])
            s1_aug1_array = np.load(group["s1_aug1"])
            s1_aug2_array = np.load(group["s1_aug2"])
            
            # 确定该文件组实际可用的 sample 数
            n_samples = min(s2_aug1_array.shape[0],
                            s2_aug2_array.shape[0],
                            s1_aug1_array.shape[0],
                            s1_aug2_array.shape[0])
            
            for i in range(n_samples):
                # 取出第 i 个样本
                s2_aug1_patch = s2_aug1_array[i]  # (H, W, T, B_s2)
                s2_aug2_patch = s2_aug2_array[i]
                s1_aug1_patch = s1_aug1_array[i]  # (H, W, T, B_s1)
                s1_aug2_patch = s1_aug2_array[i]
                
                # 对 aug1 (S2 & S1) 一起做随机空间变换
                # 保持它们的空间增强一致（同一次旋转、翻转）。
                aug1_transform_k = np.random.randint(0, 4)
                # 先旋转
                s2_aug1_patch = np.rot90(s2_aug1_patch, aug1_transform_k, axes=(0, 1))
                s1_aug1_patch = np.rot90(s1_aug1_patch, aug1_transform_k, axes=(0, 1))
                # 再翻转
                if np.random.rand() < 0.5:
                    s2_aug1_patch = np.flip(s2_aug1_patch, axis=0)
                    s1_aug1_patch = np.flip(s1_aug1_patch, axis=0)
                if np.random.rand() < 0.5:
                    s2_aug1_patch = np.flip(s2_aug1_patch, axis=1)
                    s1_aug1_patch = np.flip(s1_aug1_patch, axis=1)

                # 对 aug2 (S2 & S1) 也做随机空间变换（另一套随机参数）
                aug2_transform_k = np.random.randint(0, 4)
                s2_aug2_patch = np.rot90(s2_aug2_patch, aug2_transform_k, axes=(0, 1))
                s1_aug2_patch = np.rot90(s1_aug2_patch, aug2_transform_k, axes=(0, 1))
                if np.random.rand() < 0.5:
                    s2_aug2_patch = np.flip(s2_aug2_patch, axis=0)
                    s1_aug2_patch = np.flip(s1_aug2_patch, axis=0)
                if np.random.rand() < 0.5:
                    s2_aug2_patch = np.flip(s2_aug2_patch, axis=1)
                    s1_aug2_patch = np.flip(s1_aug2_patch, axis=1)
                
                # 创建数组副本以消除负步长，然后转为 torch.Tensor 并返回
                yield {
                    "s2_aug1": torch.tensor(s2_aug1_patch.copy(), dtype=torch.float32),
                    "s2_aug2": torch.tensor(s2_aug2_patch.copy(), dtype=torch.float32),
                    "s1_aug1": torch.tensor(s1_aug1_patch.copy(), dtype=torch.float32),
                    "s1_aug2": torch.tensor(s1_aug2_patch.copy(), dtype=torch.float32)
                }






class AustrianCropValidationPatch(Dataset):
    """
    用于验证集评估，提供基于patch的数据，针对性能进行了大量优化。
    返回 (s2_sample, s1_sample, label)，
    其中s2_sample形状为（patch_size, patch_size, sample_size_s2，band_s2+doy_s2），
    s1_sample形状为（patch_size, patch_size, sample_size_s1，band_s1+doy_s1）
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
                 patch_size=7,
                 stride=0,
                 sample_size_s2=20,
                 sample_size_s1=20,
                 min_valid_timesteps=0,
                 max_cloud_ratio=0.2,
                 standardize=True,
                 cache_dir=None,
                 n_workers=None,
                 precompute=True):
        super().__init__()
        
        # 设置CPU工作进程数量
        self.n_workers = n_workers if n_workers is not None else max(1, multiprocessing.cpu_count() - 2)
        
        # 数据参数
        self.patch_size = patch_size
        assert patch_size % 2 == 1, "patch_size must be odd"
        self.stride = stride
        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1
        self.standardize = standardize
        self.min_valid_timesteps = min_valid_timesteps
        self.max_cloud_ratio = max_cloud_ratio
        
        # 缓存设置
        self.cache_dir = cache_dir
        self.precompute = precompute
        self._setup_cache()
        
        # 数据加载（使用mmap_mode提高大文件加载效率）
        print("Loading data files...")
        self.s2_bands_data = np.load(s2_bands_file_path, mmap_mode='r')
        self.s2_masks_data = np.load(s2_masks_file_path, mmap_mode='r')
        self.s2_doys_data  = np.load(s2_doy_file_path)
        self.s1_asc_bands_data = np.load(s1_asc_bands_file_path, mmap_mode='r')
        self.s1_asc_doys_data  = np.load(s1_asc_doy_file_path)
        self.s1_desc_bands_data= np.load(s1_desc_bands_file_path, mmap_mode='r')
        self.s1_desc_doys_data = np.load(s1_desc_doy_file_path)
        self.labels = np.load(labels_path)
        
        # 标准化参数
        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std = S1_BAND_STD
        
        # 预计算有效像素
        print("Identifying valid pixels...")
        self.valid_pixels = self._find_valid_pixels()
        
        # 预计算patch索引
        print("Precomputing patch indices...")
        self.patch_indices = self._precompute_patch_indices()
        
        # 预计算数据（可选）
        if self.precompute:
            self._precompute_data()
    
    def _setup_cache(self):
        """设置缓存目录"""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # 创建缓存文件名的唯一标识符
            cache_id = f"ps{self.patch_size}_st{self.stride}_s2{self.sample_size_s2}_s1{self.sample_size_s1}_mcr{self.max_cloud_ratio}"
            self.valid_pixels_cache = os.path.join(self.cache_dir, f"valid_pixels_{cache_id}.pkl")
            self.patch_indices_cache = os.path.join(self.cache_dir, f"patch_indices_{cache_id}.pkl")
            self.samples_cache_dir = os.path.join(self.cache_dir, f"samples_{cache_id}")
            os.makedirs(self.samples_cache_dir, exist_ok=True)
    
    def _check_pixel_validity(self, coord):
        """检查像素是否有效，作为类方法而不是内部函数，以避免pickle错误"""
        i, j = coord
        s2_valid = np.sum(self.s2_masks_data[:, i, j])
        asc_valid = np.sum(np.any(self.s1_asc_bands_data[:, i, j, :] != 0, axis=1))
        desc_valid = np.sum(np.any(self.s1_desc_bands_data[:, i, j, :] != 0, axis=1))
        if s2_valid >= self.min_valid_timesteps and (asc_valid + desc_valid) >= self.min_valid_timesteps:
            return (i, j)
        return None
    
    def _find_valid_pixels(self):
        """查找所有有效像素，并尝试从缓存加载"""
        if self.cache_dir and os.path.exists(self.valid_pixels_cache):
            print(f"Loading valid pixels from cache: {self.valid_pixels_cache}")
            with open(self.valid_pixels_cache, 'rb') as f:
                return pickle.load(f)
        
        t_s2, H, W, _ = self.s2_bands_data.shape
        
        # 创建标签掩码，排除标签为0的像素
        label_mask = self.labels != 0
        
        # 生成候选坐标
        coords = []
        for i in range(H):
            for j in range(W):
                if label_mask[i, j]:
                    coords.append((i, j))
        
        print(f"Processing {len(coords)} potential valid pixels...")
        
        # 使用joblib处理像素有效性检查 - 避免pickle错误
        valid_pixels = []
        
        # 使用分批处理避免大型任务队列
        batch_size = 1000
        batches = [coords[i:i+batch_size] for i in range(0, len(coords), batch_size)]
        
        for batch in tqdm(batches, desc="Processing pixel batches"):
            results = []
            for i, j in batch:
                # 简单的顺序检查 - 针对大量内存场景优化
                s2_valid = np.sum(self.s2_masks_data[:, i, j])
                asc_valid = np.sum(np.any(self.s1_asc_bands_data[:, i, j, :] != 0, axis=1))
                desc_valid = np.sum(np.any(self.s1_desc_bands_data[:, i, j, :] != 0, axis=1))
                if s2_valid >= self.min_valid_timesteps and (asc_valid + desc_valid) >= self.min_valid_timesteps:
                    results.append((i, j))
            valid_pixels.extend(results)
        
        # 缓存结果
        if self.cache_dir:
            with open(self.valid_pixels_cache, 'wb') as f:
                pickle.dump(valid_pixels, f)
        
        return valid_pixels
    
    def _mirror_idx(self, idx, size):
        """将索引镜像到合法范围内"""
        return np.clip(size - 1 - np.abs(size - 1 - idx), 0, size - 1)
    
    def _precompute_patch_indices(self):
        """预计算每个patch的像素索引"""
        if self.cache_dir and os.path.exists(self.patch_indices_cache):
            print(f"Loading patch indices from cache: {self.patch_indices_cache}")
            with open(self.patch_indices_cache, 'rb') as f:
                return pickle.load(f)
        
        t_s2, H, W, _ = self.s2_bands_data.shape
        half_size = self.patch_size // 2
        
        # 创建所有可能的索引偏移
        offsets = np.arange(-half_size, half_size + 1, 1 + self.stride)
        rows_offsets, cols_offsets = np.meshgrid(offsets, offsets, indexing='ij')
        
        # 为每个有效像素预计算patch索引
        patch_indices = {}
        
        # 分批处理，以避免内存压力
        batch_size = 1000
        batches = [self.valid_pixels[i:i+batch_size] for i in range(0, len(self.valid_pixels), batch_size)]
        
        for batch in tqdm(batches, desc="Computing patch indices"):
            for i, j in batch:
                # 计算原始索引
                rows = i + rows_offsets.flatten()
                cols = j + cols_offsets.flatten()
                
                # 应用镜像边界条件
                rows = self._mirror_idx(rows, H)
                cols = self._mirror_idx(cols, W)
                
                # 保存索引
                patch_indices[(i, j)] = (rows.reshape(rows_offsets.shape), 
                                        cols.reshape(cols_offsets.shape))
        
        # 缓存结果
        if self.cache_dir:
            with open(self.patch_indices_cache, 'wb') as f:
                pickle.dump(patch_indices, f)
        
        return patch_indices
    
    def _precompute_data(self):
        """预计算所有样本数据"""
        print(f"Precomputing samples...")
        
        # 检查哪些样本已经被缓存
        if self.cache_dir:
            existing_cache = set()
            for idx in range(len(self.valid_pixels)):
                cache_file = os.path.join(self.samples_cache_dir, f"sample_{idx}.pkl")
                if os.path.exists(cache_file):
                    existing_cache.add(idx)
            
            # 找出需要计算的样本
            to_compute = [idx for idx in range(len(self.valid_pixels)) if idx not in existing_cache]
            if not to_compute:
                print("All samples already cached, skipping precomputation.")
                return
            print(f"Found {len(existing_cache)} cached samples, need to compute {len(to_compute)} more.")
        else:
            to_compute = list(range(len(self.valid_pixels)))
        
        # 使用多线程进行IO密集型操作
        n_threads = min(16, self.n_workers)
        
        # 分批处理以减少内存压力
        batch_size = 100
        batches = [to_compute[i:i+batch_size] for i in range(0, len(to_compute), batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} samples)")
            
            # 定义处理函数
            def process_sample(idx):
                sample = self._get_sample_internal(idx)
                if self.cache_dir:
                    cache_file = os.path.join(self.samples_cache_dir, f"sample_{idx}.pkl")
                    with open(cache_file, 'wb') as f:
                        pickle.dump(sample, f)
            
            # 使用线程池处理当前批次
            threads = []
            for idx in batch:
                thread = threading.Thread(target=process_sample, args=(idx,))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
    
    def _get_patch_vectorized(self, data, center_i, center_j):
        """
        向量化方式从数据中提取以(center_i, center_j)为中心的patch
        """
        rows, cols = self.patch_indices[(center_i, center_j)]
        patch_shape = rows.shape
        
        # 根据数据维度执行不同的切片操作
        if len(data.shape) == 3:  # (T, H, W)
            T = data.shape[0]
            patch = np.zeros((T, *patch_shape), dtype=data.dtype)
            for t in range(T):
                patch[t] = data[t][rows, cols]
        else:  # (T, H, W, C)
            T, _, _, C = data.shape
            patch = np.zeros((T, *patch_shape, C), dtype=data.dtype)
            for t in range(T):
                for c in range(C):
                    patch[t, :, :, c] = data[t][rows, cols, c]
        
        return patch, patch_shape
    
    def _augment_s2_patch_vectorized(self, s2_bands_patch, s2_masks_patch, s2_doys):
        """
        向量化方式为S2 patch数据进行时间增强
        """
        T, patch_h, patch_w, C = s2_bands_patch.shape
        
        # 计算每个时间步的云覆盖率
        cloud_ratios = 1.0 - np.mean(s2_masks_patch, axis=(1, 2))
        
        # 选择云覆盖率低于阈值的时间步
        valid_idx = np.where(cloud_ratios <= self.max_cloud_ratio)[0]
        
        # 如果没有满足条件的时间步，选择云量最少的几个
        if len(valid_idx) == 0:
            valid_idx = np.argsort(cloud_ratios)[:max(1, self.sample_size_s2 // 4)]
        
        # 随机选择时间步
        size = self.sample_size_s2
        if len(valid_idx) < size:
            idx_chosen = np.random.choice(valid_idx, size=size, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=size, replace=False)
        idx_chosen = np.sort(idx_chosen)
        
        # 提取选定的时间步
        sub_bands = s2_bands_patch[idx_chosen]
        sub_doys = s2_doys[idx_chosen]
        
        # 标准化
        if self.standardize:
            sub_bands = (sub_bands - self.s2_band_mean) / (self.s2_band_std + 1e-9)
        
        # 创建输出数组: (patch_h, patch_w, sample_size, C+1)
        result = np.zeros((patch_h, patch_w, size, C + 1), dtype=np.float32)
        
        # 填充波段数据
        result[:, :, :, :-1] = np.transpose(sub_bands, (1, 2, 0, 3))
        
        # 填充DOY数据 - 广播到所有像素
        for i in range(patch_h):
            for j in range(patch_w):
                result[i, j, :, -1] = sub_doys
        
        return torch.tensor(result, dtype=torch.float32)
    
    def _augment_s1_patch_vectorized(self, asc_bands_patch, asc_doys, desc_bands_patch, desc_doys):
        """
        向量化方式为S1 patch数据进行时间增强
        """
        Ta, patch_h, patch_w, Ca = asc_bands_patch.shape
        Td, _, _, Cd = desc_bands_patch.shape
        
        # 创建结果数组
        result = np.zeros((patch_h, patch_w, self.sample_size_s1, Ca + 1), dtype=np.float32)
        
        # 对每个像素处理S1数据
        for i in range(patch_h):
            for j in range(patch_w):
                # 获取有效时间步的掩码
                asc_valid_mask = np.any(asc_bands_patch[:, i, j, :] != 0, axis=1)
                desc_valid_mask = np.any(desc_bands_patch[:, i, j, :] != 0, axis=1)
                
                # 选择有效数据
                asc_bands_valid = asc_bands_patch[asc_valid_mask, i, j, :]
                asc_doys_valid = asc_doys[asc_valid_mask]
                desc_bands_valid = desc_bands_patch[desc_valid_mask, i, j, :]
                desc_doys_valid = desc_doys[desc_valid_mask]
                
                # 合并升降轨数据
                s1_bands = []
                s1_doys = []
                
                if len(asc_bands_valid) > 0:
                    s1_bands.append(asc_bands_valid)
                    s1_doys.append(asc_doys_valid)
                
                if len(desc_bands_valid) > 0:
                    s1_bands.append(desc_bands_valid)
                    s1_doys.append(desc_doys_valid)
                
                # 如果没有有效数据，继续下一个像素
                if not s1_bands:
                    continue
                
                # 合并数据
                s1_bands_all = np.vstack(s1_bands)
                s1_doys_all = np.concatenate(s1_doys)
                
                # 选择时间步
                if len(s1_bands_all) < self.sample_size_s1:
                    idx_chosen = np.random.choice(len(s1_bands_all), size=self.sample_size_s1, replace=True)
                else:
                    idx_chosen = np.random.choice(len(s1_bands_all), size=self.sample_size_s1, replace=False)
                
                # 提取数据
                sub_bands = s1_bands_all[idx_chosen]
                sub_doys = s1_doys_all[idx_chosen]
                
                # 标准化
                if self.standardize:
                    sub_bands = (sub_bands - self.s1_band_mean) / (self.s1_band_std + 1e-9)
                
                # 填充结果数组
                result[i, j, :, :-1] = sub_bands
                result[i, j, :, -1] = sub_doys
        
        return torch.tensor(result, dtype=torch.float32)
    
    def _get_sample_internal(self, idx):
        """内部方法：获取单个样本的处理逻辑"""
        # 获取中心像素坐标
        i, j = self.valid_pixels[idx]
        
        # 获取标签
        label = torch.tensor([self.labels[i, j] - 1], dtype=torch.long)
        
        # 提取patch
        s2_bands_patch, patch_shape = self._get_patch_vectorized(self.s2_bands_data, i, j)
        s2_masks_patch, _ = self._get_patch_vectorized(self.s2_masks_data, i, j)
        s1_asc_bands_patch, _ = self._get_patch_vectorized(self.s1_asc_bands_data, i, j)
        s1_desc_bands_patch, _ = self._get_patch_vectorized(self.s1_desc_bands_data, i, j)
        
        # 时间增强
        s2_sample = self._augment_s2_patch_vectorized(s2_bands_patch, s2_masks_patch, self.s2_doys_data)
        s1_sample = self._augment_s1_patch_vectorized(
            s1_asc_bands_patch, self.s1_asc_doys_data,
            s1_desc_bands_patch, self.s1_desc_doys_data
        )
        
        return s2_sample, s1_sample, label
    
    def __len__(self):
        return len(self.valid_pixels)
    
    def __getitem__(self, idx):
        # 如果已经预计算了所有数据并使用缓存
        if self.cache_dir and self.precompute:
            cache_file = os.path.join(self.samples_cache_dir, f"sample_{idx}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # 否则实时计算
        return self._get_sample_internal(idx)


# 线程安全的数据批处理器
class ThreadSafeDataLoader:
    """线程安全的数据加载器，无需依赖多进程"""
    def __init__(self, dataset, batch_size=1, shuffle=False, num_threads=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_threads = num_threads
        
        # 计算总批次数
        self.num_samples = len(dataset)
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        # 生成索引
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        # 将索引分成批次
        batch_indices = [
            indices[i:i+self.batch_size] 
            for i in range(0, self.num_samples, self.batch_size)
        ]
        
        for batch_idx in range(len(batch_indices)):
            # 为当前批次的每个样本创建结果列表
            s2_samples = [None] * len(batch_indices[batch_idx])
            s1_samples = [None] * len(batch_indices[batch_idx])
            labels = [None] * len(batch_indices[batch_idx])
            
            # 创建线程
            threads = []
            
            def load_sample(local_idx, global_idx):
                s2_sample, s1_sample, label = self.dataset[global_idx]
                s2_samples[local_idx] = s2_sample
                s1_samples[local_idx] = s1_sample
                labels[local_idx] = label
            
            # 启动线程
            for local_idx, global_idx in enumerate(batch_indices[batch_idx]):
                thread = threading.Thread(target=load_sample, args=(local_idx, global_idx))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 将列表转换为批次张量
            s2_batch = torch.stack(s2_samples)
            s1_batch = torch.stack(s1_samples)
            labels_batch = torch.cat(labels)
            
            yield s2_batch, s1_batch, labels_batch