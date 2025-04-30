# src/datasets/ssl_dataset.py

import os
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info, Dataset
import logging
import time
    

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
    不做两次增强，只返回 (s2_sample, s1_sample, label, field_id, pixel_pos)。
    加入了 field_id 以便进行基于 field 的分割。
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
                 field_id_path=None,
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
        
        # 加载field_id数据（如果提供）
        self.field_ids = None
        if field_id_path is not None and os.path.exists(field_id_path):
            self.field_ids = np.load(field_id_path)

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

        # 直接把doy加入到bands中
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

        # 直接把doy加入到bands中
        arr = np.hstack([sub_bands, sub_d.reshape(-1, 1)])
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx):
        i, j = self.valid_pixels[idx]
        label = self.labels[i, j] - 1  # 地物类别，从0开始

        s2_bands_ij = self.s2_bands_data[:, i, j, :]
        s2_masks_ij = self.s2_masks_data[:, i, j]
        s2_doys_ij  = self.s2_doys_data

        asc_bands_ij = self.s1_asc_bands_data[:, i, j, :]
        asc_doys_ij = self.s1_asc_doys_data
        desc_bands_ij = self.s1_desc_bands_data[:, i, j, :]
        desc_doys_ij = self.s1_desc_doys_data

        s2_sample = self._augment_s2(s2_bands_ij, s2_masks_ij, s2_doys_ij)
        s1_sample = self._augment_s1(asc_bands_ij, asc_doys_ij, desc_bands_ij, desc_doys_ij)
        
        # 获取field_id（如果可用）
        field_id = -1
        if self.field_ids is not None:
            field_id = self.field_ids[i, j]
            
        # 返回像素位置以便之后重构
        position = torch.tensor([i, j], dtype=torch.long)
        return s2_sample, s1_sample, label, field_id, position
   
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
        # s2_bands_path = os.path.join(tile_path, "bands_downsample_100.npy")    # (t_s2, H, W, 10)
        # s2_masks_path = os.path.join(tile_path, "masks_downsample_100.npy")    # (t_s2, H, W) 
        s2_doy_path   = os.path.join(tile_path, "doys.npy")     # (t_s2,)        

        # >>> 修复UInt16类型: 转换为 int32 等通用类型 <<<
        self.s2_bands = np.load(s2_bands_path).astype(np.float32)
        self.s2_masks = np.load(s2_masks_path).astype(np.int32)
        # self.s2_bands = np.load(s2_bands_path, mmap_mode='r').astype(np.float32)
        # self.s2_masks = np.load(s2_masks_path, mmap_mode='r').astype(np.int32)
        self.s2_doys  = np.load(s2_doy_path).astype(np.int32)   # 避免uint16, 转成int32

        # 加载 S1 asc
        s1_asc_bands_path = os.path.join(tile_path, "sar_ascending.npy")      # (t_s1a, H, W, 2)
        # s1_asc_bands_path = os.path.join(tile_path, "sar_ascending_downsample_100.npy")      # (t_s1a, H, W, 2)
        s1_asc_doy_path   = os.path.join(tile_path, "sar_ascending_doy.npy")  # (t_s1a,)        

        self.s1_asc_bands = np.load(s1_asc_bands_path).astype(np.float32)
        # self.s1_asc_bands = np.load(s1_asc_bands_path, mmap_mode='r').astype(np.float32)
        self.s1_asc_doys  = np.load(s1_asc_doy_path).astype(np.int32)

        # 加载 S1 desc
        s1_desc_bands_path = os.path.join(tile_path, "sar_descending.npy")      # (t_s1d, H, W, 2)
        # s1_desc_bands_path = os.path.join(tile_path, "sar_descending_downsample_100.npy")      # (t_s1d, H, W, 2)
        s1_desc_doy_path   = os.path.join(tile_path, "sar_descending_doy.npy")  # (t_s1d,)

        self.s1_desc_bands = np.load(s1_desc_bands_path).astype(np.float32)
        # self.s1_desc_bands = np.load(s1_desc_bands_path, mmap_mode='r').astype(np.float32)
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
        # 如果min_valid_timesteps大于0，则需要过滤有效像素
        if self.min_valid_timesteps > 0:
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
        else:
            # 如果min_valid_timesteps为0，则保留所有像素
            for idx, (i, j) in enumerate(ij_coords):
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

class OptimizedTileInferenceDataset(Dataset):
    """
    Optimized dataset for single tile inference with memory-mapped arrays
    and pre-computed valid pixels to speed up data loading.
    """
    def __init__(self,
                 tile_path,
                 min_valid_timesteps=10,
                 standardize=True,
                 precompute_valid_pixels=True,
                 cache_file=None):
        super().__init__()
        self.tile_path = tile_path
        self.min_valid_timesteps = min_valid_timesteps
        self.standardize = standardize

        # Paths to data files
        s2_bands_path = os.path.join(tile_path, "bands.npy")  
        s2_masks_path = os.path.join(tile_path, "masks.npy")  
        s2_doy_path = os.path.join(tile_path, "doys.npy")     
        s1_asc_bands_path = os.path.join(tile_path, "sar_ascending.npy")
        s1_asc_doy_path = os.path.join(tile_path, "sar_ascending_doy.npy")
        s1_desc_bands_path = os.path.join(tile_path, "sar_descending.npy")
        s1_desc_doy_path = os.path.join(tile_path, "sar_descending_doy.npy")

        # Load data with memory mapping for large arrays
        start_time = time.time()
        logging.info(f"Loading data files with memory mapping...")
        
        # Load S2 data
        self.s2_bands = np.load(s2_bands_path, mmap_mode='r')
        self.s2_masks = np.load(s2_masks_path, mmap_mode='r')
        self.s2_doys = np.load(s2_doy_path)  # Small file, load fully
        
        # Load S1 ascending data
        self.s1_asc_bands = np.load(s1_asc_bands_path, mmap_mode='r')
        self.s1_asc_doys = np.load(s1_asc_doy_path)  # Small file, load fully
        
        # Load S1 descending data
        try:
            self.s1_desc_bands = np.load(s1_desc_bands_path, mmap_mode='r')
            self.s1_desc_doys = np.load(s1_desc_doy_path)  # Small file, load fully
            self.has_desc_data = True
        except (FileNotFoundError, ValueError) as e:
            # Handle empty descending data
            logging.warning(f"No valid descending data found: {e}")
            s1_desc_shape = list(self.s1_asc_bands.shape)
            s1_desc_shape[0] = 0
            self.s1_desc_bands = np.zeros(s1_desc_shape, dtype=np.float32)
            self.s1_desc_doys = np.array([], dtype=np.int32)
            self.has_desc_data = False

        # Get shape information
        self.t_s2, self.H, self.W, self.s2_bands_dim = self.s2_bands.shape
        
        # Set band mean and std values
        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std = S1_BAND_STD
        
        # Check if cache file exists
        if cache_file and os.path.exists(cache_file):
            logging.info(f"Loading valid pixels from cache file: {cache_file}")
            self.valid_pixels = np.load(cache_file)
            logging.info(f"Loaded {len(self.valid_pixels)} valid pixels from cache")
        else:
            # Find valid pixels
            if precompute_valid_pixels:
                self._find_valid_pixels()
                
                # Save to cache if requested
                if cache_file:
                    logging.info(f"Saving valid pixels to cache: {cache_file}")
                    np.save(cache_file, self.valid_pixels)
            else:
                # Use all pixels (will filter during __getitem__)
                ij_coords = np.indices((self.H, self.W)).reshape(2, -1).T
                self.valid_pixels = np.column_stack([np.arange(len(ij_coords)), ij_coords])
        
        logging.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        logging.info(f"Dataset contains {len(self.valid_pixels)} valid pixels")
    
    def _find_valid_pixels(self):
        """Find valid pixels with sufficient timesteps in both S1 and S2 data."""
        logging.info(f"Finding valid pixels with min_valid_timesteps={self.min_valid_timesteps}...")
        start_time = time.time()
        
        # Pre-compute valid counts for S2 data across all pixels
        s2_valid_counts = np.sum(self.s2_masks, axis=0)  # Shape: (H, W)
        
        # Process in chunks for S1 data to avoid memory issues
        chunk_size = 100000
        ij_coords = np.indices((self.H, self.W)).reshape(2, -1).T
        valid_pixels = []
        
        for chunk_start in range(0, len(ij_coords), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(ij_coords))
            chunk_coords = ij_coords[chunk_start:chunk_end]
            
            # Process each pixel in the chunk
            for idx, (i, j) in enumerate(chunk_coords):
                global_idx = chunk_start + idx
                
                # Check S2 valid count (already computed)
                s2_valid = s2_valid_counts[i, j]
                
                # Skip if S2 doesn't have enough data
                if s2_valid < self.min_valid_timesteps:
                    continue
                
                # Check S1 ascending data
                s1_asc_valid = 0
                if self.s1_asc_bands.shape[0] > 0:
                    s1_asc_ij = self.s1_asc_bands[:, i, j, :]  # (t_s1a, 2)
                    s1_asc_valid = np.sum(np.any(s1_asc_ij != 0, axis=1))
                
                # Check S1 descending data
                s1_desc_valid = 0
                if self.has_desc_data and self.s1_desc_bands.shape[0] > 0:
                    s1_desc_ij = self.s1_desc_bands[:, i, j, :]  # (t_s1d, 2)
                    s1_desc_valid = np.sum(np.any(s1_desc_ij != 0, axis=1))
                
                # Check if total S1 valid count meets threshold
                s1_total_valid = s1_asc_valid + s1_desc_valid
                if s1_total_valid >= self.min_valid_timesteps:
                    valid_pixels.append([global_idx, i, j])
            
            # Print progress
            if (chunk_start // chunk_size) % 10 == 0:
                logging.info(f"Processed {chunk_end}/{len(ij_coords)} pixels, found {len(valid_pixels)} valid pixels...")
        
        self.valid_pixels = np.array(valid_pixels)
        logging.info(f"Found {len(self.valid_pixels)} valid pixels in {time.time() - start_time:.2f} seconds")

    def __len__(self):
        return len(self.valid_pixels)

    def __getitem__(self, index):
        global_idx, i, j = self.valid_pixels[index]

        # Extract data for this pixel - convert to appropriate types
        s2_bands_ij = self.s2_bands[:, i, j, :].astype(np.float32)  # (t_s2, 10)
        s2_masks_ij = self.s2_masks[:, i, j].astype(np.int32)       # (t_s2,)
        s2_doys_ij = self.s2_doys.astype(np.int32)                  # (t_s2,)

        s1_asc_bands_ij = self.s1_asc_bands[:, i, j, :].astype(np.float32)  # (t_s1a, 2)
        s1_asc_doys_ij = self.s1_asc_doys.astype(np.int32)                  # (t_s1a,)

        if self.has_desc_data and self.s1_desc_bands.shape[0] > 0:
            s1_desc_bands_ij = self.s1_desc_bands[:, i, j, :].astype(np.float32)  # (t_s1d, 2)
            s1_desc_doys_ij = self.s1_desc_doys.astype(np.int32)                  # (t_s1d,)
        else:
            s1_desc_bands_ij = np.zeros((0, 2), dtype=np.float32)
            s1_desc_doys_ij = np.array([], dtype=np.int32)

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


# class SingleTileInferenceDataset(Dataset):
#     """
#     用于单Tile推理的数据集，仅返回像素的 (i, j) 以及对应的 s2/s1 数据。
#     本版本从预先切块后的文件中加载数据，参数 chunk_folder 和 chunk_id 指定对应切块文件，
#     start_row 为当前 chunk 在全图中的起始行号（用于计算全局索引）。
#     """
#     def __init__(self, tile_path, chunk_folder, chunk_id, start_row,
#                  min_valid_timesteps=10, standardize=True):
#         super().__init__()
#         self.tile_path = tile_path
#         self.chunk_folder = chunk_folder
#         self.chunk_id = chunk_id
#         self.start_row = start_row
#         self.min_valid_timesteps = min_valid_timesteps
#         self.standardize = standardize

#         # 从预先切块的文件中加载 s2 和 s1 数据
#         self.s2_bands = np.load(os.path.join(chunk_folder, f"chunk_{chunk_id}_bands.npy")).astype(np.float32)
#         self.s2_masks = np.load(os.path.join(chunk_folder, f"chunk_{chunk_id}_masks.npy")).astype(np.int32)
#         # 加载 s2_doys（较小，不切块）
#         s2_doy_path = os.path.join(tile_path, "doys.npy")
#         self.s2_doys = np.load(s2_doy_path).astype(np.int32)

#         self.s1_asc_bands = np.load(os.path.join(chunk_folder, f"chunk_{chunk_id}_sar_ascending.npy")).astype(np.float32)
#         s1_asc_doy_path = os.path.join(tile_path, "sar_ascending_doy.npy")
#         self.s1_asc_doys = np.load(s1_asc_doy_path).astype(np.int32)

#         self.s1_desc_bands = np.load(os.path.join(chunk_folder, f"chunk_{chunk_id}_sar_descending.npy")).astype(np.float32)
#         s1_desc_doy_path = os.path.join(tile_path, "sar_descending_doy.npy")
#         self.s1_desc_doys = np.load(s1_desc_doy_path).astype(np.int32)

#         # 获取切块后数据的形状
#         # s2_bands: (t_s2, H_chunk, W, channels)
#         _, self.H_chunk, self.W, _ = self.s2_bands.shape

#         self.s2_band_mean = S2_BAND_MEAN
#         self.s2_band_std = S2_BAND_STD
#         self.s1_band_mean = S1_BAND_MEAN
#         self.s1_band_std = S1_BAND_STD

#         # 过滤有效像素，仅在当前 chunk 范围内（行号为 0 ~ H_chunk-1，对应全图行号为 start_row ~ start_row+H_chunk-1）
#         self.valid_pixels = []
#         for i_rel in range(self.H_chunk):
#             i_abs = i_rel + self.start_row
#             for j in range(self.W):
#                 s2_mask_ij = self.s2_masks[:, i_rel, j]
#                 s2_valid = s2_mask_ij.sum()
#                 s1_asc_ij = self.s1_asc_bands[:, i_rel, j, :]  # (t_s1a, channels)
#                 s1_asc_valid = np.any(s1_asc_ij != 0, axis=-1).sum()
#                 s1_desc_ij = self.s1_desc_bands[:, i_rel, j, :]
#                 s1_desc_valid = np.any(s1_desc_ij != 0, axis=-1).sum()
#                 s1_total_valid = s1_asc_valid + s1_desc_valid
#                 if (s2_valid >= self.min_valid_timesteps) and (s1_total_valid >= self.min_valid_timesteps):
#                     # 保存 (局部行号, 全局行号, j)
#                     self.valid_pixels.append((i_rel, i_abs, j))
#         logging.info(f"[SingleTileInferenceDataset] tile={tile_path}, chunk_id={chunk_id}, start_row={self.start_row}, total_valid_pixels={len(self.valid_pixels)}")

#     def __len__(self):
#         return len(self.valid_pixels)

#     def __getitem__(self, index):
#         i_rel, i_abs, j = self.valid_pixels[index]
#         # 计算全局索引：i_abs * W + j
#         full_idx = i_abs * self.W + j
#         s2_bands_ij = self.s2_bands[:, i_rel, j, :]  # (t_s2, channels)
#         s2_masks_ij = self.s2_masks[:, i_rel, j]
#         s2_doys_ij = self.s2_doys
#         s1_asc_bands_ij = self.s1_asc_bands[:, i_rel, j, :]
#         s1_asc_doys_ij = self.s1_asc_doys
#         s1_desc_bands_ij = self.s1_desc_bands[:, i_rel, j, :]
#         s1_desc_doys_ij = self.s1_desc_doys
#         sample = {
#             "global_idx": full_idx,
#             "i": i_abs,
#             "j": j,
#             "s2_bands": s2_bands_ij,
#             "s2_masks": s2_masks_ij,
#             "s2_doys": s2_doys_ij,
#             "s1_asc_bands": s1_asc_bands_ij,
#             "s1_asc_doys": s1_asc_doys_ij,
#             "s1_desc_bands": s1_desc_bands_ij,
#             "s1_desc_doys": s1_desc_doys_ij,
#         }
#         return sample
    
    
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

        # logging.info(f"[PastisPatchInferenceDataset] patch={patch_path}, total_valid_pixels={len(self.valid_pixels)}")

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





class HDF5Dataset_Multimodal_Tiles_Iterable_64_Fixed(IterableDataset):
    """
    Data loader for the new structure with 64 time steps.
    Data structure:
    data_root/
      ├─ s2/   -> Each .npy file shape (N, 64, bands+doy)
      └─ s1/   -> Each .npy file shape (N, 64, bands+doy)
    """
    def __init__(self,
                 data_root,
                 min_valid_timesteps=10,
                 sample_size_s2=64,
                 sample_size_s1=64,
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

        # Build data paths
        self.s2_dir = os.path.join(data_root, "s2")
        self.s1_dir = os.path.join(data_root, "s1")
        
        for d in [self.s2_dir, self.s1_dir]:
            if not os.path.exists(d):
                raise RuntimeError(f"Directory {d} not found!")
                
        file_names = sorted(os.listdir(self.s2_dir))
        np.random.shuffle(file_names)
        self.file_groups = []
        
        for fn in file_names:
            file_path_s2 = os.path.join(self.s2_dir, fn)
            file_path_s1 = os.path.join(self.s1_dir, fn)
            
            if os.path.exists(file_path_s1):
                self.file_groups.append({
                    "s2": file_path_s2,
                    "s1": file_path_s1
                })
                
        if len(self.file_groups) == 0:
            raise RuntimeError("No valid file groups found in dataset!")

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
            s2_array = np.load(group["s2"])
            s1_array = np.load(group["s1"])
            
            n_samples = min(s2_array.shape[0], s1_array.shape[0])
            
            for i in range(n_samples):
                s2_sample = s2_array[i]  # (64, bands+doy)
                s1_sample = s1_array[i]  # (64, bands+doy)
                
                # Get valid masks (non-zero time steps)
                s2_valid_mask = np.any(s2_sample[:, :-1] != 0, axis=1)
                s1_valid_mask = np.any(s1_sample[:, :-1] != 0, axis=1)
                
                s2_valid_steps = np.sum(s2_valid_mask)
                s1_valid_steps = np.sum(s1_valid_mask)
                
                # Skip samples with too few valid time steps
                if s2_valid_steps < self.min_valid_timesteps or s1_valid_steps < self.min_valid_timesteps:
                    continue
                
                # Apply augmentations
                s2_aug1, s2_aug2 = self._augment_pair(s2_sample, s2_valid_mask, is_s2=True)
                s1_aug1, s1_aug2 = self._augment_pair(s1_sample, s1_valid_mask, is_s2=False)
                
                yield {
                    "s2_aug1": torch.tensor(s2_aug1, dtype=torch.float32),
                    "s2_aug2": torch.tensor(s2_aug2, dtype=torch.float32),
                    "s1_aug1": torch.tensor(s1_aug1, dtype=torch.float32),
                    "s1_aug2": torch.tensor(s1_aug2, dtype=torch.float32),
                    "s2_mask": torch.tensor(s2_valid_mask, dtype=torch.bool),
                    "s1_mask": torch.tensor(s1_valid_mask, dtype=torch.bool)
                }
                
    def _augment_pair(self, sample, valid_mask, is_s2=True):
        """
        Apply two different augmentations to the same sample
        """
        # Copy the original data for two augmentations
        aug1 = sample.copy()
        aug2 = sample.copy()
        
        # Apply augmentations
        aug1 = self._apply_augmentation(aug1, valid_mask, is_s2)
        aug2 = self._apply_augmentation(aug2, valid_mask, is_s2, different_seed=True)
        
        # Standardize if needed
        # if self.standardize:
        #     band_mean = self.s2_band_mean if is_s2 else self.s1_band_mean
        #     band_std = self.s2_band_std if is_s2 else self.s1_band_std
            
        #     aug1[:, :-1] = (aug1[:, :-1] - band_mean) / (band_std + 1e-9)
        #     aug2[:, :-1] = (aug2[:, :-1] - band_mean) / (band_std + 1e-9)
            
        return aug1, aug2
        
    def _apply_augmentation(self, sample, valid_mask, is_s2=True, different_seed=False):
        """
        Apply various augmentations to the sample
        """
        # Make a copy to avoid modifying the input
        aug_sample = sample.copy()
        data = aug_sample[:, :-1]
        doy = aug_sample[:, -1]
        valid_indices = np.where(valid_mask)[0]
        
        # Set a different random seed if needed
        if different_seed:
            np.random.seed(np.random.randint(0, 100000))
        
        # 1. Random time step masking (always apply)
        if len(valid_indices) > self.min_valid_timesteps:
            mask_count = np.random.randint(1, min(len(valid_indices) - self.min_valid_timesteps + 1, 
                                                max(int(len(valid_indices) * 0.2), 1)))
            mask_indices = np.random.choice(valid_indices, size=mask_count, replace=False)
            aug_sample[mask_indices, :-1] = 0  # Mask all bands except DOY
        
        # 2. Random band masking (always apply)
        band_count = data.shape[1] -1 # remove DOY
        if band_count > 1:
            mask_bands = np.random.randint(0, max(1, band_count // 4))
            if mask_bands > 0:
                band_indices = np.random.choice(band_count, size=mask_bands, replace=False)
                aug_sample[valid_indices][:, band_indices] = 0
        
        # 3. Random normalization perturbation (always apply)
        scale = np.random.uniform(0.95, 1.05)
        shift = np.random.uniform(-0.05, 0.05)
        aug_sample[valid_indices, :-1] = aug_sample[valid_indices, :-1] * scale + shift
        
        # 4. DOY adjustment (always apply)
        # doy_shift = np.random.randint(-2, 3)  # -2 to +2 days
        # aug_sample[valid_indices, -1] = np.clip(aug_sample[valid_indices, -1] + doy_shift, 1, 366)
        
        # 5. Random Gaussian noise (30-50% probability)
        if np.random.rand() < 0.4:
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, size=aug_sample[valid_indices, :-1].shape)
            aug_sample[valid_indices, :-1] += noise
        
        # 6. Random dropout with interpolation (20-30% probability)
        # if len(valid_indices) > 3 and np.random.rand() < 0.25:
        #     drop_count = np.random.randint(1, min(len(valid_indices) // 3, 5))
        #     drop_indices = np.random.choice(valid_indices, size=drop_count, replace=False)
            
        #     # Simple linear interpolation for each band
        #     for band_idx in range(data.shape[1]):
        #         for drop_idx in drop_indices:
        #             # Find nearest valid indices
        #             left_indices = valid_indices[valid_indices < drop_idx]
        #             right_indices = valid_indices[valid_indices > drop_idx]
                    
        #             if len(left_indices) > 0 and len(right_indices) > 0:
        #                 left_idx = left_indices[-1]
        #                 right_idx = right_indices[0]
        #                 left_val = aug_sample[left_idx, band_idx]
        #                 right_val = aug_sample[right_idx, band_idx]
        #                 left_doy = aug_sample[left_idx, -1]
        #                 right_doy = aug_sample[right_idx, -1]
        #                 drop_doy = aug_sample[drop_idx, -1]
                        
        #                 # Linear interpolation
        #                 if right_doy > left_doy:  # Avoid division by zero
        #                     weight = (drop_doy - left_doy) / (right_doy - left_doy)
        #                     aug_sample[drop_idx, band_idx] = left_val * (1 - weight) + right_val * weight
        
        return aug_sample

class AustrianCropValidation_64_Fixed(Dataset):
    """
    Validation dataset for the 64 time steps model.
    Processes the validation data to have fixed 64 time steps.
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
                 field_id_path=None,
                 sample_size_s2=64,
                 sample_size_s1=64,
                 min_valid_timesteps=0,
                 standardize=True):
        super().__init__()
        self.s2_bands_data = np.load(s2_bands_file_path)
        self.s2_masks_data = np.load(s2_masks_file_path)
        self.s2_doys_data = np.load(s2_doy_file_path)
        self.s1_asc_bands_data = np.load(s1_asc_bands_file_path)
        self.s1_asc_doys_data = np.load(s1_asc_doy_file_path)
        self.s1_desc_bands_data = np.load(s1_desc_bands_file_path)
        self.s1_desc_doys_data = np.load(s1_desc_doy_file_path)
        self.labels = np.load(labels_path)
        
        # Load field_id data (if provided)
        self.field_ids = None
        if field_id_path is not None and os.path.exists(field_id_path):
            self.field_ids = np.load(field_id_path)

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
            if self.labels[i, j] == 0:  # Skip background pixels
                continue
            
            # Check if the pixel has enough valid time steps
            s2_valid = self.s2_masks_data[:, i, j].sum()
            asc_valid = np.any(self.s1_asc_bands_data[:, i, j, :] != 0, axis=-1).sum()
            desc_valid = np.any(self.s1_desc_bands_data[:, i, j, :] != 0, axis=-1).sum()
            
            if s2_valid >= self.min_valid_timesteps and (asc_valid + desc_valid) >= self.min_valid_timesteps:
                self.valid_pixels.append((i, j))

    def __len__(self):
        return len(self.valid_pixels)

    def _prepare_s2_sample(self, s2_bands, s2_masks, s2_doys):
        """
        Prepare S2 sample with 64 time steps.
        """
        # Find valid time steps
        valid_indices = np.where(s2_masks)[0]
        
        # Create empty array for 64 time steps
        sample = np.zeros((self.sample_size_s2, s2_bands.shape[-1] + 1))  # +1 for DOY
        valid_mask = np.zeros(self.sample_size_s2, dtype=bool)
        
        if len(valid_indices) > 0:
            # If we have valid data
            if len(valid_indices) <= self.sample_size_s2:
                # If we have fewer valid time steps than needed, use all of them
                use_indices = valid_indices
            else:
                # If we have more valid time steps than needed, select evenly distributed ones
                indices_chunks = np.array_split(valid_indices, self.sample_size_s2)
                use_indices = [chunk[len(chunk)//2] for chunk in indices_chunks]
                
            # Fill the sample with valid data
            for i, idx in enumerate(use_indices):
                if i >= self.sample_size_s2:
                    break
                sample[i, :-1] = s2_bands[idx]
                sample[i, -1] = s2_doys[idx]
                valid_mask[i] = True
                
        # Standardize if needed
        if self.standardize:
            sample[:, :-1] = (sample[:, :-1] - self.s2_band_mean) / (self.s2_band_std + 1e-9)
            
        return sample, valid_mask

    def _prepare_s1_sample(self, asc_bands, asc_doys, desc_bands, desc_doys):
        """
        Prepare S1 sample with 64 time steps, combining ascending and descending data.
        """
        # Find valid time steps
        asc_valid = np.any(asc_bands != 0, axis=-1)
        desc_valid = np.any(desc_bands != 0, axis=-1)
        
        # Extract valid data
        asc_valid_data = [(asc_bands[i], asc_doys[i]) for i in range(len(asc_doys)) if asc_valid[i]]
        desc_valid_data = [(desc_bands[i], desc_doys[i]) for i in range(len(desc_doys)) if desc_valid[i]]
        
        # Combine and sort by DOY
        combined_data = asc_valid_data + desc_valid_data
        if combined_data:
            combined_data.sort(key=lambda x: x[1])  # Sort by DOY
            
        # Create empty array for 64 time steps
        sample = np.zeros((self.sample_size_s1, asc_bands.shape[-1] + 1))  # +1 for DOY
        valid_mask = np.zeros(self.sample_size_s1, dtype=bool)
        
        if combined_data:
            # If we have valid data
            if len(combined_data) <= self.sample_size_s1:
                # If we have fewer valid time steps than needed, use all of them
                use_data = combined_data
            else:
                # If we have more valid time steps than needed, select evenly distributed ones
                indices_chunks = np.array_split(range(len(combined_data)), self.sample_size_s1)
                use_data = [combined_data[chunk[len(chunk)//2]] for chunk in indices_chunks if len(chunk) > 0]
                
            # Fill the sample with valid data
            for i, (band, doy) in enumerate(use_data):
                if i >= self.sample_size_s1:
                    break
                sample[i, :-1] = band
                sample[i, -1] = doy
                valid_mask[i] = True
                
        # Standardize if needed
        if self.standardize:
            sample[:, :-1] = (sample[:, :-1] - self.s1_band_mean) / (self.s1_band_std + 1e-9)
            
        return sample, valid_mask

    def __getitem__(self, idx):
        i, j = self.valid_pixels[idx]
        label = self.labels[i, j] - 1  # Labels start from 0
        
        # Extract S2 data
        s2_bands_ij = self.s2_bands_data[:, i, j, :]
        s2_masks_ij = self.s2_masks_data[:, i, j]
        s2_doys_ij = self.s2_doys_data
        
        # Extract S1 data
        s1_asc_bands_ij = self.s1_asc_bands_data[:, i, j, :]
        s1_asc_doys_ij = self.s1_asc_doys_data
        s1_desc_bands_ij = self.s1_desc_bands_data[:, i, j, :]
        s1_desc_doys_ij = self.s1_desc_doys_data
        
        # Prepare samples
        s2_sample, s2_valid_mask = self._prepare_s2_sample(s2_bands_ij, s2_masks_ij, s2_doys_ij)
        s1_sample, s1_valid_mask = self._prepare_s1_sample(s1_asc_bands_ij, s1_asc_doys_ij, 
                                                       s1_desc_bands_ij, s1_desc_doys_ij)
        
        # Get field_id (if available)
        field_id = -1
        if self.field_ids is not None:
            field_id = self.field_ids[i, j]
            
        # Return pixel position for reconstruction
        position = np.array([i, j], dtype=np.int64)
        
        return (torch.tensor(s2_sample, dtype=torch.float32), 
                torch.tensor(s1_sample, dtype=torch.float32),
                label, field_id, position,
                torch.tensor(s2_valid_mask, dtype=torch.bool),
                torch.tensor(s1_valid_mask, dtype=torch.bool))
        

