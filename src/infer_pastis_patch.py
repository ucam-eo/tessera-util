#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_pastis_patch_intel.py

多 Intel XPU (oneCCL) 并行推理版本，每个进程分配到若干 patch，
对每个 patch 单独构建 Dataset + DataLoader 并推理，结果保存在 output_dir/representation_{patchid}.npy。
这样可以避免在 ConcatDataset 里出现索引偏移导致的错误映射，也能保证 patch_id 对应正确的输出文件。

用法示例（在 Slurm/mpi/mpirun 作业脚本里）：
   mpirun -np 8 python infer_pastis_patch_intel.py --config configs/pastis_patch_infer_config.py --visualize
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch

import torch.distributed as dist
import intel_extension_for_pytorch as ipex  # noqa: F401 (用于 HPC 场景)
import oneccl_bindings_for_pytorch  # noqa: F401 (提供 'ccl' backend 给 PyTorch)

from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from PIL import Image
import importlib.util

# 如果你的项目根目录在上一层，可以这样添加:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.ssl_dataset import PastisPatchInferenceDataset
from models.builder import build_ssl_model
from models.ssl_model import MultimodalBTInferenceModel


def load_config_module(config_file_path):
    """动态加载 Python config 文件 (如 pastis_patch_infer_config.py)，并返回其中的 config 字典对象。"""
    spec = importlib.util.spec_from_file_location("my_dynamic_config", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["my_dynamic_config"] = config_module
    spec.loader.exec_module(config_module)
    return config_module


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for multiple PASTIS patches with Intel XPU parallelization.")
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
                        help="Max number of patches to process at once (per rank).")
    parser.add_argument('--verbose', action='store_true',
                        help="Set logging level to DEBUG for more detailed output.")
    return parser.parse_args()


def init_distributed_ccl():
    """
    使用 oneCCL (ccl backend) 初始化分布式。
    根据常见的环境变量获取 rank/world_size，如果 HPC 不设置这些，需要自行调整。
    """
    rank = int(os.environ.get("PMI_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    init_method = f"tcp://{master_addr}:{master_port}"

    dist.init_process_group(backend="ccl",
                            init_method=init_method,
                            rank=rank,
                            world_size=world_size)

    # 获取每个节点上有多少个 XPU
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
    # 若 ZE_FLAT_DEVICE_HIERARCHY=FLAT，则可能拆分为更多设备
    if (os.environ.get("ZE_FLAT_DEVICE_HIERARCHY", "N/A") == "FLAT") and (gpus_per_node != 8):
        gpus_per_node *= 2
    # local_rank = rank % gpus_per_node
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    return rank, world_size, local_rank, gpus_per_node


class PastisPatchDatasetSingle(torch.utils.data.Dataset):
    """
    针对单个 Patch 的推理 Dataset。与原 PastisPatchInferenceDataset 类似，但整合在一起，
    并使用私有属性 _patch_id 来避免与 @property patch_id 冲突。
    """
    def __init__(self, patch_path, min_valid_timesteps=4, standardize=True):
        self._patch_path = patch_path
        self._patch_id = os.path.basename(os.path.normpath(patch_path))

        self.bands = np.load(os.path.join(patch_path, "bands.npy"))        # (T_s2, H, W, 10)
        self.masks = np.load(os.path.join(patch_path, "masks.npy"))       # (T_s2, H, W)
        self.doys  = np.load(os.path.join(patch_path, "doys.npy"))        # (T_s2,)

        self.sar_asc_bands = np.load(os.path.join(patch_path, "sar_ascending.npy"))       # (T_a, H, W, 2)
        self.sar_asc_doys  = np.load(os.path.join(patch_path, "sar_ascending_doy.npy"))   # (T_a,)

        self.sar_desc_bands = np.load(os.path.join(patch_path, "sar_descending.npy"))     # (T_d, H, W, 2)
        self.sar_desc_doys  = np.load(os.path.join(patch_path, "sar_descending_doy.npy")) # (T_d,)

        self.T_s2, self.H, self.W, _ = self.bands.shape

        # 筛选出有效像素：在 T_s2 中有效时刻 >= min_valid_timesteps
        valid_counts = self.masks.sum(axis=0)  # shape (H, W)
        valid_mask2d = (valid_counts >= min_valid_timesteps)
        self.valid_coords = np.nonzero(valid_mask2d)  # (row_indices, col_indices)
        self.pixel_indices = self.valid_coords[0] * self.W + self.valid_coords[1]

        # 计算每个 pixel 的 S2 mean/std
        bands_2d = self.bands.reshape(self.T_s2, self.H*self.W, 10)
        masks_2d = self.masks.reshape(self.T_s2, self.H*self.W)
        s2_band_mean = []
        s2_band_std  = []
        for pix_id in self.pixel_indices:
            pix_bands = bands_2d[:, pix_id, :]  # (T_s2, 10)
            pix_mask  = masks_2d[:, pix_id]     # (T_s2,)
            valid_bands = pix_bands[pix_mask == 1]
            if len(valid_bands) == 0:
                mean_ = np.zeros((10,), dtype=np.float32)
                std_  = np.ones((10,), dtype=np.float32)
            else:
                mean_ = valid_bands.mean(axis=0)
                std_  = valid_bands.std(axis=0)
                std_[std_ < 1e-8] = 1.0
            s2_band_mean.append(mean_)
            s2_band_std.append(std_)
        self.s2_band_mean = np.array(s2_band_mean, dtype=np.float32)
        self.s2_band_std  = np.array(s2_band_std, dtype=np.float32)

        # 计算 S1 asc+desc mean/std
        T_a, _, _, _ = self.sar_asc_bands.shape
        T_d, _, _, _ = self.sar_desc_bands.shape
        asc_bands_2d = self.sar_asc_bands.reshape(T_a, self.H*self.W, 2)
        desc_bands_2d= self.sar_desc_bands.reshape(T_d, self.H*self.W, 2)

        s1_band_mean = []
        s1_band_std  = []
        for pix_id in self.pixel_indices:
            asc_pix = asc_bands_2d[:, pix_id, :]   # (T_a,2)
            desc_pix= desc_bands_2d[:, pix_id, :]  # (T_d,2)
            asc_valid_mask = np.any(asc_pix != 0, axis=-1)
            desc_valid_mask= np.any(desc_pix!= 0, axis=-1)
            asc_valid_bands= asc_pix[asc_valid_mask]
            desc_valid_bands=desc_pix[desc_valid_mask]
            merged = np.concatenate([asc_valid_bands, desc_valid_bands], axis=0) if (len(asc_valid_bands)>0 or len(desc_valid_bands)>0) else np.zeros((0,2), dtype=np.float32)
            if len(merged)==0:
                mean_ = np.zeros((2,), dtype=np.float32)
                std_  = np.ones((2,), dtype=np.float32)
            else:
                mean_ = merged.mean(axis=0)
                std_  = merged.std(axis=0)
                std_[std_ < 1e-8] = 1.0
            s1_band_mean.append(mean_)
            s1_band_std.append(std_)
        self.s1_band_mean = np.array(s1_band_mean, dtype=np.float32)
        self.s1_band_std  = np.array(s1_band_std, dtype=np.float32)

        self.standardize = standardize

    @property
    def patch_id(self):
        return self._patch_id

    @property
    def patch_path(self):
        return self._patch_path

    @property
    def shape(self):
        return (self.H, self.W)

    def __len__(self):
        return len(self.pixel_indices)

    def __getitem__(self, idx):
        pix_id = self.pixel_indices[idx]
        row = pix_id // self.W
        col = pix_id % self.W
        s2_bands = self.bands[:, row, col, :]  # (T_s2, 10)
        s2_masks = self.masks[:, row, col]     # (T_s2,)
        s2_doys  = self.doys                  # (T_s2,)

        asc_bands = self.sar_asc_bands[:, row, col, :]  # (T_a,2)
        asc_doys  = self.sar_asc_doys                  # (T_a,)

        desc_bands= self.sar_desc_bands[:, row, col, :] # (T_d,2)
        desc_doys = self.sar_desc_doys                  # (T_d,)

        return {
            "global_idx": pix_id,
            "s2_bands": s2_bands.astype(np.float32),
            "s2_masks": s2_masks.astype(np.uint8),
            "s2_doys" : s2_doys.astype(np.float32),
            "s1_asc_bands": asc_bands.astype(np.float32),
            "s1_asc_doys" : asc_doys.astype(np.float32),
            "s1_desc_bands": desc_bands.astype(np.float32),
            "s1_desc_doys" : desc_doys.astype(np.float32),
            "s2_band_mean": self.s2_band_mean[idx],
            "s2_band_std" : self.s2_band_std[idx],
            "s1_band_mean": self.s1_band_mean[idx],
            "s1_band_std" : self.s1_band_std[idx],
        }


def inference_single_patch(patch_path, model, config, device, logger):
    """
    对单个 patch 进行推理，并将结果保存到 representation_{patchid}.npy / .png
    """
    dataset = PastisPatchDatasetSingle(
        patch_path=patch_path,
        min_valid_timesteps=config["min_valid_timesteps"],
        standardize=True
    )
    H, W = dataset.shape
    patch_id = dataset.patch_id

    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=False,
        drop_last=False
    )

    latent_dim = config["latent_dim"]  # 如果在你的 model config 中指定
    out_array = np.zeros((H * W, latent_dim), dtype=np.float32)

    # 开始推理
    with torch.no_grad():
        for batch_data in data_loader:
            global_idxs = batch_data["global_idx"].numpy()
            s2_bands_batch = batch_data["s2_bands"].numpy()     # (B, T_s2, 10)
            s2_masks_batch = batch_data["s2_masks"].numpy()     # (B, T_s2)
            s2_doys_batch  = batch_data["s2_doys"].numpy()      # (B, T_s2)
            s1_asc_bands_batch = batch_data["s1_asc_bands"].numpy()
            s1_asc_doys_batch  = batch_data["s1_asc_doys"].numpy()
            s1_desc_bands_batch = batch_data["s1_desc_bands"].numpy()
            s1_desc_doys_batch  = batch_data["s1_desc_doys"].numpy()

            s2_band_means = batch_data["s2_band_mean"].numpy()  # (B, 10)
            s2_band_stds  = batch_data["s2_band_std"].numpy()   # (B, 10)
            s1_band_means = batch_data["s1_band_mean"].numpy()  # (B, 2)
            s1_band_stds  = batch_data["s1_band_std"].numpy()   # (B, 2)

            B = s2_bands_batch.shape[0]
            sum_repr = torch.zeros((B, latent_dim), dtype=torch.float32, device=device)

            # repeat_times：对同一个 pixel 多次随机采样再平均
            for _ in range(config["repeat_times"]):
                # 对 S2 随机抽样 => (B, sample_size_s2, 12)
                s2_input_list = []
                for b_i in range(B):
                    valid_idx = np.nonzero(s2_masks_batch[b_i])[0]
                    if len(valid_idx) == 0:
                        valid_idx = np.arange(s2_masks_batch.shape[1])
                    if len(valid_idx) < config["sample_size_s2"]:
                        idx_chosen = np.random.choice(valid_idx, config["sample_size_s2"], replace=True)
                    else:
                        idx_chosen = np.random.choice(valid_idx, config["sample_size_s2"], replace=False)
                    idx_chosen = np.sort(idx_chosen)
                    sub_bands = s2_bands_batch[b_i, idx_chosen, :]
                    sub_doys  = s2_doys_batch[b_i, idx_chosen]
                    # 标准化
                    sub_bands = (sub_bands - s2_band_means[b_i]) / (s2_band_stds[b_i] + 1e-9)
                    doys_norm = sub_doys / 365.0
                    sin_doy = np.sin(2*np.pi*doys_norm).reshape(-1,1)
                    cos_doy = np.cos(2*np.pi*doys_norm).reshape(-1,1)
                    arr_ = np.hstack([sub_bands, sin_doy, cos_doy])
                    s2_input_list.append(arr_.astype(np.float32))
                s2_input_np = np.stack(s2_input_list, axis=0)

                # 对 S1 (asc+desc) 合并时序随机抽样 => (B, sample_size_s1, 4)
                s1_input_list = []
                for b_i in range(B):
                    asc_bands_all = s1_asc_bands_batch[b_i]
                    asc_doys_all  = s1_asc_doys_batch[b_i]
                    desc_bands_all= s1_desc_bands_batch[b_i]
                    desc_doys_all = s1_desc_doys_batch[b_i]

                    merged_bands = np.concatenate([asc_bands_all, desc_bands_all], axis=0)
                    merged_doys  = np.concatenate([asc_doys_all,  desc_doys_all], axis=0)
                    valid_mask = np.any(merged_bands != 0, axis=-1)
                    valid_idx = np.nonzero(valid_mask)[0]
                    if len(valid_idx)==0:
                        valid_idx = np.arange(merged_bands.shape[0])
                    if len(valid_idx) < config["sample_size_s1"]:
                        idx_chosen = np.random.choice(valid_idx, config["sample_size_s1"], replace=True)
                    else:
                        idx_chosen = np.random.choice(valid_idx, config["sample_size_s1"], replace=False)
                    idx_chosen = np.sort(idx_chosen)
                    sub_bands = merged_bands[idx_chosen,:]
                    sub_doys  = merged_doys[idx_chosen]
                    sub_bands = (sub_bands - s1_band_means[b_i]) / (s1_band_stds[b_i] + 1e-9)
                    doys_norm = sub_doys / 365.0
                    sin_doy = np.sin(2*np.pi*doys_norm).reshape(-1,1)
                    cos_doy = np.cos(2*np.pi*doys_norm).reshape(-1,1)
                    arr_ = np.hstack([sub_bands, sin_doy, cos_doy])
                    s1_input_list.append(arr_.astype(np.float32))
                s1_input_np = np.stack(s1_input_list, axis=0)

                s2_input = torch.tensor(s2_input_np, dtype=torch.float32, device=device)
                s1_input = torch.tensor(s1_input_np, dtype=torch.float32, device=device)

                z = model(s2_input, s1_input)  # (B, latent_dim)
                sum_repr += z

            avg_repr = sum_repr / float(config["repeat_times"])
            avg_repr_np = avg_repr.cpu().numpy()
            # 写入 out_array
            for i in range(B):
                gidx = global_idxs[i]
                out_array[gidx] = avg_repr_np[i]

    # 推理完成，保存 .npy
    out_array = out_array.reshape(H, W, latent_dim)
    out_path = os.path.join(config["output_dir"], f"representation_{patch_id}.npy")
    os.makedirs(config["output_dir"], exist_ok=True)
    np.save(out_path, out_array)
    logger.info(f"Patch {patch_id} => saved to {out_path}, shape={out_array.shape}")

    # 如果要可视化
    if config["visualize"]:
        flat_repr = out_array.reshape(-1, latent_dim)
        pca = PCA(n_components=3)
        flat_rgb = pca.fit_transform(flat_repr)
        rgb_image = flat_rgb.reshape(H, W, 3)
        min_val, max_val = rgb_image.min(), rgb_image.max()
        if max_val > min_val:
            rgb_norm = (rgb_image - min_val)/(max_val - min_val)
        else:
            rgb_norm = np.zeros_like(rgb_image)
        rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
        img_out_path = os.path.join(config["output_dir"], f"representation_{patch_id}.png")
        Image.fromarray(rgb_uint8).save(img_out_path)
        logger.info(f"Patch {patch_id} => visualization saved to {img_out_path}")


def main():
    args = parse_args()

    # ========== 1. 加载配置并合并命令行覆盖 ==========
    config_module = load_config_module(args.config)
    config = config_module.config

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

    # ========== 2. 初始化分布式 (CCL backend) ==========
    rank, world_size, local_rank, gpus_per_node = init_distributed_ccl()

    # 根据命令行 verbose 情况设定日志
    log_level = logging.DEBUG if (args.verbose and rank == 0) else (logging.INFO if rank == 0 else logging.WARN)
    logging.basicConfig(level=log_level, format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    logger.info(f"===== Starting Inference (Intel XPU / oneCCL) =====")
    logger.info(f"rank={rank}, world_size={world_size}, local_rank={local_rank}, gpus_per_node={gpus_per_node}")
    for k, v in config.items():
        logger.info(f"{k}: {v}")

    # ========== 3. 设置XPU设备 ==========
    current_xpu = f"xpu:{local_rank}"
    torch.xpu.set_device(current_xpu)
    device = torch.device(current_xpu)
    logger.info(f"Use device: {device}")

    # ========== 4. 构建模型并加载checkpoint ==========
    ssl_model = build_ssl_model(config, device)
    if rank == 0:
        logger.info("Before loading checkpoint, s2_backbone weights sample:")
        logger.info(ssl_model.s2_backbone.fc_out.weight[:1])
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    state_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    ssl_model.load_state_dict(checkpoint[state_key], strict=True)
    if rank == 0:
        logger.info("After loading checkpoint, s2_backbone weights sample:")
        logger.info(ssl_model.s2_backbone.fc_out.weight[:1])

    # 冻结参数
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
        dim_reducer=ssl_model.dim_reducer
    ).to(device)
    model.eval()

    # ========== 5. 扫描patch目录，过滤出有效patch ==========
    patch_root = config["patch_root"]
    if not os.path.isdir(patch_root):
        logger.error(f"patch_root does not exist or is not a directory: {patch_root}")
        dist.barrier()
        dist.destroy_process_group()
        return

    required_files = [
        "bands.npy",
        "masks.npy",
        "doys.npy",
        "sar_ascending.npy",
        "sar_ascending_doy.npy",
        "sar_descending.npy",
        "sar_descending_doy.npy"
    ]
    subdirs = sorted(os.listdir(patch_root))
    valid_patch_paths = []
    for subdir in subdirs:
        patch_path = os.path.join(patch_root, subdir)
        if not os.path.isdir(patch_path):
            continue
        if all(os.path.isfile(os.path.join(patch_path, f)) for f in required_files):
            valid_patch_paths.append(patch_path)

    total_patches = len(valid_patch_paths)
    if total_patches == 0:
        logger.error("No valid patches found. Exit.")
        dist.barrier()
        dist.destroy_process_group()
        return

    # ========== 6. 分配patch到各rank，行间隔方式 ==========
    patches_for_this_rank = valid_patch_paths[rank::world_size]
    local_count = len(patches_for_this_rank)
    logger.info(f"Total valid patches={total_patches}, rank={rank} => local_count={local_count}")

    # ========== 7. 分批次处理，防止一次加载太多导致OOM ==========
    max_batch = config["max_patch_batch_size"]
    num_batches = (local_count + max_batch - 1)//max_batch
    logger.info(f"Will process local patches in {num_batches} batch(es), each batch up to {max_batch} patches.")

    start_time = time.time()
    processed_local = 0

    for batch_i in range(num_batches):
        batch_patch_paths = patches_for_this_rank[batch_i*max_batch : (batch_i+1)*max_batch]
        logger.info(f"Batch {batch_i+1}/{num_batches}, patch_count={len(batch_patch_paths)}")

        # 逐个patch处理
        for patch_path in batch_patch_paths:
            patch_id = os.path.basename(patch_path)
            t0 = time.time()
            inference_single_patch(patch_path, model, config, device, logger)
            processed_local += 1
            logger.info(f"Patch {patch_id} done. Elapsed={time.time()-t0:.2f}s")

    end_time = time.time()
    logger.info(f"Rank={rank} finished. Processed {processed_local} patches. Time used={end_time - start_time:.2f}s")

    # ========== 8. 结束分布式 ==========
    dist.barrier()
    if rank == 0:
        logger.info("All ranks finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
