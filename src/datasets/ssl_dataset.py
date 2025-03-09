# src/datasets/ssl_dataset.py

import os
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info, Dataset
import logging

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

        doys_norm = sub_doys / 365.
        sin_doy = np.sin(2 * np.pi * doys_norm).reshape(-1, 1)
        cos_doy = np.cos(2 * np.pi * doys_norm).reshape(-1, 1)
        arr = np.hstack([sub_bands, sin_doy, cos_doy])
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

        doys_norm = sub_d / 365.
        sin_doy = np.sin(2 * np.pi * doys_norm).reshape(-1, 1)
        cos_doy = np.cos(2 * np.pi * doys_norm).reshape(-1, 1)
        arr = np.hstack([sub_bands, sin_doy, cos_doy])
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
        self.tile_path = tile_path
        self.min_valid_timesteps = min_valid_timesteps
        self.standardize = standardize

        # 加载 S2
        s2_bands_path = os.path.join(tile_path, "bands.npy")    # (t_s2, H, W, 10)
        s2_masks_path = os.path.join(tile_path, "masks.npy")    # (t_s2, H, W)
        s2_doy_path   = os.path.join(tile_path, "doys.npy")     # (t_s2,)

        # >>> 修复UInt16类型: 转换为 int32 等通用类型 <<<
        self.s2_bands = np.load(s2_bands_path).astype(np.float32)
        self.s2_masks = np.load(s2_masks_path).astype(np.int32)
        self.s2_doys  = np.load(s2_doy_path).astype(np.int32)   # 避免uint16, 转成int32

        # 加载 S1 asc
        s1_asc_bands_path = os.path.join(tile_path, "sar_ascending.npy")      # (t_s1a, H, W, 2)
        s1_asc_doy_path   = os.path.join(tile_path, "sar_ascending_doy.npy")  # (t_s1a,)

        self.s1_asc_bands = np.load(s1_asc_bands_path).astype(np.float32)
        self.s1_asc_doys  = np.load(s1_asc_doy_path).astype(np.int32)

        # 加载 S1 desc
        s1_desc_bands_path = os.path.join(tile_path, "sar_descending.npy")      # (t_s1d, H, W, 2)
        s1_desc_doy_path   = os.path.join(tile_path, "sar_descending_doy.npy")  # (t_s1d,)

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
            if (s2_valid >= self.min_valid_timesteps) and (s1_total_valid >= self.min_valid_timesteps):
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
