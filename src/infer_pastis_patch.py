#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_pastis_patch.py

批量推理脚本：
- 从 config 文件中读取 patch_root 目录
- 遍历其中的所有子文件夹，每个子文件夹对应一个 patch_id（文件夹名）
- 扫描所有有效的 patch（检查必需的 npy 文件），
  然后按照指定的 max_patch_batch_size 分批加载数据进行推理，
  每批加载后生成对应的 representation_{patchid}.npy 保存至 config["output_dir"]
- 新增可选的可视化功能：当命令行开启 --visualize 参数时，会对推理得到的 representation (形状为 (H,W,C)) 进行 PCA 降维至3维，
  并生成一副假彩色 RGB 图像保存为 PNG
- 这样可以避免一次性加载所有 patch 导致内存不足，同时利用 GPU 加速批量推理

用法示例：
   python infer_pastis_patch.py \
       --config configs/pastis_patch_infer_config.py \
       --verbose \
       --visualize \
       --max_patch_batch_size 200

如需覆盖部分配置，可使用命令行参数 (--patch_root, --output_dir, 等)。
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch

from torch.utils.data import DataLoader, ConcatDataset

from datasets.ssl_dataset import PastisPatchInferenceDataset
from models.builder import build_ssl_model
from models.ssl_model import MultimodalBTInferenceModel

# 新增：用于 PCA 降维以及图像保存
from sklearn.decomposition import PCA
from PIL import Image

import importlib.util

# 将项目根目录添加到 sys.path 中（假设 src 和 configs 在同一目录下）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_config_module(config_file_path):
    """
    动态加载 Python config 文件 (如 pastis_patch_infer_config.py)，
    并返回其中的 config 对象 (config字典)。
    """
    spec = importlib.util.spec_from_file_location("my_dynamic_config", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["my_dynamic_config"] = config_module
    spec.loader.exec_module(config_module)
    return config_module

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for multiple PASTIS patches, generating representation maps.")
    parser.add_argument('--config', type=str, default="configs/pastis_patch_infer_config.py",
                        help="Path to config file (e.g. configs/pastis_patch_infer_config.py).")
    parser.add_argument('--patch_root', type=str, default=None,
                        help="Root directory containing multiple patch subfolders. Overrides config if provided.")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save representation_{patchid}.npy. Overrides config if provided.")
    parser.add_argument('--repeat_times', type=int, default=None,
                        help="How many times to sample each pixel. Overrides config if provided.")
    parser.add_argument('--sample_size_s2', type=int, default=None,
                        help="Number of s2 timesteps to sample. Overrides config if provided.")
    parser.add_argument('--sample_size_s1', type=int, default=None,
                        help="Number of s1 timesteps to sample. Overrides config if provided.")
    parser.add_argument('--visualize', action='store_true',
                        help="If set, generate a PCA-based RGB visualization (PNG) for each patch.")
    parser.add_argument('--max_patch_batch_size', type=int, default=200,
                        help="Max number of patches to process at once.")
    parser.add_argument('--verbose', action='store_true',
                        help="Set logging level to DEBUG for more detailed output.")
    return parser.parse_args()

def sample_s2_batch(s2_bands_batch, s2_masks_batch, s2_doys_batch,
                    band_means, band_stds, sample_size_s2, standardize=True):
    """
    针对同一个 batch (B 个像素)，对 S2 时序进行随机采样
    s2_bands_batch: shape (B, T_s2, 10)
    s2_masks_batch: shape (B, T_s2)
    s2_doys_batch : shape (B, T_s2)
    band_means: numpy array，形状 (B, 10)
    band_stds : numpy array，形状 (B, 10)

    返回: (B, sample_size_s2, 12)
    """
    B = s2_bands_batch.shape[0]
    out_list = []
    for b in range(B):
        valid_idx = np.nonzero(s2_masks_batch[b])[0]
        if len(valid_idx) == 0:
            valid_idx = np.arange(s2_masks_batch.shape[1])
        if len(valid_idx) < sample_size_s2:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=False)
        idx_chosen = np.sort(idx_chosen)
        sub_bands = s2_bands_batch[b, idx_chosen, :]  # (sample_size_s2, 10)
        sub_doys  = s2_doys_batch[b, idx_chosen]
        if standardize:
            sub_bands = (sub_bands - band_means[b]) / (band_stds[b] + 1e-9)
        # doys_norm = sub_doys / 365.0
        # sin_doy = np.sin(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
        # cos_doy = np.cos(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
        # out_arr = np.hstack([sub_bands, sin_doy, cos_doy])  # (sample_size_s2, 12)
        
        # 直接把doy放到sub_bands后面
        out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])  # (sample_size_s2, 11)
        out_list.append(out_arr.astype(np.float32))
    return np.stack(out_list, axis=0)

def sample_s1_batch(s1_asc_bands_batch, s1_asc_doys_batch,
                    s1_desc_bands_batch, s1_desc_doys_batch,
                    band_means, band_stds, sample_size_s1, standardize=True):
    """
    针对 batch 的 S1 asc + desc 合并时序抽样。
    返回: (B, sample_size_s1, 4)
    """
    B = s1_asc_bands_batch.shape[0]
    out_list = []
    for b in range(B):
        s1_bands_all = np.concatenate([s1_asc_bands_batch[b], s1_desc_bands_batch[b]], axis=0)
        s1_doys_all  = np.concatenate([s1_asc_doys_batch[b],  s1_desc_doys_batch[b]], axis=0)
        valid_mask = np.any(s1_bands_all != 0, axis=-1)
        valid_idx = np.nonzero(valid_mask)[0]
        if len(valid_idx) == 0:
            valid_idx = np.arange(s1_bands_all.shape[0])
        if len(valid_idx) < sample_size_s1:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=False)
        idx_chosen = np.sort(idx_chosen)
        sub_bands = s1_bands_all[idx_chosen, :]
        sub_doys  = s1_doys_all[idx_chosen]
        if standardize:
            sub_bands = (sub_bands - band_means[b]) / (band_stds[b] + 1e-9)
        # doys_norm = sub_doys / 365.0
        # sin_doy = np.sin(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
        # cos_doy = np.cos(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
        # out_arr = np.hstack([sub_bands, sin_doy, cos_doy])  # (sample_size_s1, 4)
        # 直接把doy放到sub_bands后面
        out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])  # (sample_size_s2, 11)
        out_list.append(out_arr.astype(np.float32))
    return np.stack(out_list, axis=0)

# PatchDatasetWrapper 保持不变，其作用为加载单个 patch 数据（仅在调用时加载该 patch）
class PatchDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, patch_path, min_valid_timesteps):
        self.patch_id = os.path.basename(os.path.normpath(patch_path))
        self.dataset = PastisPatchInferenceDataset(
            patch_path=patch_path,
            min_valid_timesteps=min_valid_timesteps,
            standardize=True
        )
        self.shape = self.dataset.shape  # (H, W)
        # 获取波段均值和标准差
        self.s2_band_mean = self.dataset.s2_band_mean
        self.s2_band_std  = self.dataset.s2_band_std
        self.s1_band_mean = self.dataset.s1_band_mean
        self.s1_band_std  = self.dataset.s1_band_std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["patch_id"] = self.patch_id
        # 将均值和标准差添加到每个样本中
        item["s2_band_mean"] = self.s2_band_mean
        item["s2_band_std"]  = self.s2_band_std
        item["s1_band_mean"] = self.s1_band_mean
        item["s1_band_std"]  = self.s1_band_std
        return item

def inference_on_all_patches(combined_dataset, patch_info, model, config, device):
    """
    对合并后的多个 patch 进行推理。结果将按 patch_id 分类保存为 representation_{patchid}.npy，
    并根据配置生成可视化 PNG 图像。
    patch_info: 字典 mapping patch_id -> (H, W)
    """
    data_loader = DataLoader(
        combined_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False
    )
    # 用于存储每个 patch 的推理结果，结构: {patch_id: [(global_idx, representation_vector), ...]}
    results_dict = {}
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            global_idxs = batch_data["global_idx"].numpy()  # (B,)
            patch_ids = batch_data["patch_id"]  # 长度 B 的字符串列表
            s2_bands_batch = batch_data["s2_bands"].numpy()     # (B, T_s2, 10)
            s2_masks_batch = batch_data["s2_masks"].numpy()       # (B, T_s2)
            s2_doys_batch  = batch_data["s2_doys"].numpy()        # (B, T_s2)
            s1_asc_bands_batch = batch_data["s1_asc_bands"].numpy()
            s1_asc_doys_batch  = batch_data["s1_asc_doys"].numpy()
            s1_desc_bands_batch = batch_data["s1_desc_bands"].numpy()
            s1_desc_doys_batch  = batch_data["s1_desc_doys"].numpy()

            s2_band_means = np.array(batch_data["s2_band_mean"])  # (B, 10)
            s2_band_stds  = np.array(batch_data["s2_band_std"])   # (B, 10)
            s1_band_means = np.array(batch_data["s1_band_mean"])
            s1_band_stds  = np.array(batch_data["s1_band_std"])

            B = s2_bands_batch.shape[0]
            sum_repr = None
            for _ in range(config["repeat_times"]):
                s2_input_np = sample_s2_batch(
                    s2_bands_batch, s2_masks_batch, s2_doys_batch,
                    band_means=s2_band_means, band_stds=s2_band_stds,
                    sample_size_s2=config["sample_size_s2"],
                    standardize=True
                )
                s1_input_np = sample_s1_batch(
                    s1_asc_bands_batch, s1_asc_doys_batch,
                    s1_desc_bands_batch, s1_desc_doys_batch,
                    band_means=s1_band_means, band_stds=s1_band_stds,
                    sample_size_s1=config["sample_size_s1"],
                    standardize=True
                )
                s2_input = torch.tensor(s2_input_np, dtype=torch.float32, device=device)
                s1_input = torch.tensor(s1_input_np, dtype=torch.float32, device=device)
                z = model(s2_input, s1_input)  # (B, latent_dim)
                if sum_repr is None:
                    sum_repr = z
                else:
                    sum_repr += z
            avg_repr = sum_repr / float(config["repeat_times"])
            avg_repr_np = avg_repr.cpu().numpy()  # (B, latent_dim)
            for i in range(B):
                pid = patch_ids[i]
                gidx = global_idxs[i]
                rep_vec = avg_repr_np[i]
                if pid not in results_dict:
                    results_dict[pid] = []
                results_dict[pid].append((gidx, rep_vec))
            logging.info(f"Processed batch {batch_idx+1}/{len(data_loader)}")
    # 对每个 patch 的结果进行整理，并保存为 npy 文件和 PNG 可视化图像（如果需要）
    os.makedirs(config["output_dir"], exist_ok=True)
    for pid, items in results_dict.items():
        H, W = patch_info[pid]
        latent_dim = items[0][1].shape[0]
        out_array = np.full((H * W, latent_dim), 0, dtype=np.float32)
        for gidx, rep_vec in items:
            out_array[gidx] = rep_vec
        out_array = out_array.reshape(H, W, latent_dim)
        out_path = os.path.join(config["output_dir"], f"representation_{pid}.npy")
        np.save(out_path, out_array)
        logging.info(f"Patch {pid} => {out_path}, shape={out_array.shape}")
        if config.get("visualize", False):
            flat_repr = out_array.reshape(-1, latent_dim)
            pca = PCA(n_components=3)
            flat_rgb = pca.fit_transform(flat_repr)
            rgb_image = flat_rgb.reshape(H, W, 3)
            min_val = rgb_image.min()
            max_val = rgb_image.max()
            if max_val > min_val:
                rgb_norm = (rgb_image - min_val) / (max_val - min_val)
            else:
                rgb_norm = np.zeros_like(rgb_image)
            rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
            img_out_path = os.path.join(config["output_dir"], f"representation_{pid}.png")
            im = Image.fromarray(rgb_uint8)
            im.save(img_out_path)
            logging.info(f"Visualization saved to {img_out_path}")
        logging.info(f"Saved {pid} to {out_path}")

def main():
    args = parse_args()

    # 1. 加载配置
    config_module = load_config_module(args.config)
    config = config_module.config

    # 2. 覆盖配置
    if args.patch_root is not None:
        config["patch_root"] = args.patch_root
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.repeat_times is not None:
        config["repeat_times"] = args.repeat_times
    if args.sample_size_s2 is not None:
        config["sample_size_s2"] = args.sample_size_s2
    if args.sample_size_s1 is not None:
        config["sample_size_s1"] = args.sample_size_s1
    config["visualize"] = args.visualize
    config["max_patch_batch_size"] = args.max_patch_batch_size

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("===== Batch Inference for PASTIS patches =====")
    for k, v in config.items():
        logging.info(f"  {k}: {v}")

    # 3. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_model = build_ssl_model(config, device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    state_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    
    ################### 用于处理FSDP ###################
    state_dict = checkpoint[state_key]
    # 创建新的state_dict，移除"_orig_mod."前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]  # 移除前缀
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    # 使用处理后的state_dict加载模型
    ssl_model.load_state_dict(new_state_dict, strict=True)
    #####################################################
    
    # ssl_model.load_state_dict(checkpoint[state_key], strict=True)

    for param in ssl_model.s2_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.s1_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.dim_reducer.parameters():
        param.requires_grad = False

    model = MultimodalBTInferenceModel(
        s2_backbone=ssl_model.s2_backbone,
        s1_backbone=ssl_model.s1_backbone,
        fusion_method=config["fusion_method"],
        dim_reducer=ssl_model.dim_reducer,
    ).to(device)
    model.eval()

    # 4. 先扫描所有有效 patch 的路径（不加载数据），再分批加载并推理
    patch_root = config["patch_root"]
    subdirs = sorted(os.listdir(patch_root))
    valid_patch_paths = []
    for subdir in subdirs:
        patch_path = os.path.join(patch_root, subdir)
        if not os.path.isdir(patch_path):
            continue
        required_files = [
            "bands.npy",
            "masks.npy",
            "doys.npy",
            "sar_ascending.npy",
            "sar_ascending_doy.npy",
            "sar_descending.npy",
            "sar_descending_doy.npy"
        ]
        if not all(os.path.isfile(os.path.join(patch_path, rf)) for rf in required_files):
            logging.debug(f"Skip {subdir}: required files not found.")
            continue
        valid_patch_paths.append(patch_path)

    total_patches = len(valid_patch_paths)
    if total_patches == 0:
        logging.error("No valid patches found.")
        return

    max_batch = config["max_patch_batch_size"]
    num_batches = (total_patches + max_batch - 1) // max_batch
    logging.info(f"Total valid patches: {total_patches}. Will process in {num_batches} batch(es) (max {max_batch} patches per batch).")

    start_time = time.time()
    processed = 0
    # 按批次加载和推理
    for batch_idx in range(num_batches):
        batch_patch_paths = valid_patch_paths[batch_idx * max_batch : (batch_idx + 1) * max_batch]
        patch_datasets = []
        patch_info = {}  # 用于记录当前批次中各个 patch 的 (H, W)
        for patch_path in batch_patch_paths:
            dataset_wrapper = PatchDatasetWrapper(patch_path, config["min_valid_timesteps"])
            patch_datasets.append(dataset_wrapper)
            patch_info[dataset_wrapper.patch_id] = dataset_wrapper.shape
        combined_dataset = ConcatDataset(patch_datasets)
        logging.info(f"Processing batch {batch_idx+1}/{num_batches} with {len(patch_datasets)} patches.")
        inference_on_all_patches(combined_dataset, patch_info, model, config, device)
        processed += len(patch_datasets)
    logging.info(f"Processed {processed} patches in total. Time used = {time.time() - start_time:.1f}s.")
    logging.info("All done.")

if __name__ == "__main__":
    main()
