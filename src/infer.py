#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import argparse
import logging
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from models.modules import TransformerEncoder, ProjectionHead
from models.ssl_model import MultimodalBTModel, MultimodalBTInferenceModel
from models.builder import build_ssl_model
from datasets.ssl_dataset import SingleTileInferenceDataset
import importlib.util
import sys
# 将项目根目录添加到 sys.path 中（假设 src 和 configs 在同一目录下）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_config_module(config_file_path):
    spec = importlib.util.spec_from_file_location("my_dynamic_config", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["my_dynamic_config"] = config_module
    spec.loader.exec_module(config_module)
    return config_module

def parse_args():
    parser = argparse.ArgumentParser(description="Downstream Classification Training")
    parser.add_argument('--config', type=str, required=True, help="Path to config file (e.g. configs/downstream_config.py)")
    return parser.parse_args()

def main():
    args = parse_args()
    config_module = load_config_module(args.config)
    config = config_module.config  # 直接从模块拿 config 字典

    # Step2: 日志 & 设备设置（单 GPU）
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    # 打印config
    logging.info("Configurations:")
    for k, v in config.items():
        logging.info(f"  {k}: {v}")

    # Step3: 构建数据集 & DataLoader（不使用分布式采样）
    dataset = SingleTileInferenceDataset(
        tile_path=config["tile_path"],
        min_valid_timesteps=config["min_valid_timesteps"],
        standardize=False  # 注意：在 sample 时才做标准化
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # Step4: 构建ssl模型
    ssl_model = build_ssl_model(config, device)

    logging.info("After loading checkpoint, s2_backbone weights:")
    logging.info(ssl_model.s2_backbone.fc_out.weight)
    logging.info(f"Loading SSL checkpoint from {config['checkpoint_path']}")
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
    ssl_model.load_state_dict(checkpoint[state_key], strict=True)
    logging.info("SSL backbone weights after loading checkpoint (s2 backbone):")
    logging.info(ssl_model.s2_backbone.fc_out.weight)
    
    # 冻结 SSL 骨干参数
    for param in ssl_model.s2_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.s1_backbone.parameters():
        param.requires_grad = False
    
        # ------------------------------
    # 构建推理模型
    # ------------------------------
    model = MultimodalBTInferenceModel(
        s2_backbone=ssl_model.s2_backbone,
        s1_backbone=ssl_model.s1_backbone,
        fusion_method=config["fusion_method"]
    ).to(device)
    logging.info("Inference model constructed. s2 backbone weights:")
    logging.info(model.s2_backbone.fc_out.weight)
    
    model.eval()

    # ========== 新增的两个批量采样辅助函数 ==========
    def sample_s2_batch(s2_bands_batch, s2_masks_batch, s2_doys_batch,
                        band_mean, band_std, sample_size_s2, standardize=True):
        """
        针对同一个批次的像素一次性做 S2 的随机采样。
          s2_bands_batch.shape = (B, T_s2, 10)
          s2_masks_batch.shape = (B, T_s2)
          s2_doys_batch.shape  = (B, T_s2)
        返回: np.array, shape=(B, sample_size_s2, 12), dtype float32
        """
        B = s2_bands_batch.shape[0]
        out_list = []
        for b in range(B):
            valid_idx = np.nonzero(s2_masks_batch[b])[0]
            if len(valid_idx) < sample_size_s2:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=True)
            else:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=False)
            idx_chosen = np.sort(idx_chosen)

            sub_bands = s2_bands_batch[b, idx_chosen, :]  # (sample_size_s2, 10)
            sub_doys  = s2_doys_batch[b, idx_chosen]      # (sample_size_s2,)
            if standardize:
                sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

            doys_norm = sub_doys / 365.0
            sin_doy = np.sin(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
            cos_doy = np.cos(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)

            out_arr = np.hstack([sub_bands, sin_doy, cos_doy])  # (sample_size_s2, 12)
            out_list.append(out_arr.astype(np.float32))

        return np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s2, 12)

    def sample_s1_batch(s1_asc_bands_batch, s1_asc_doys_batch,
                        s1_desc_bands_batch, s1_desc_doys_batch,
                        band_mean, band_std, sample_size_s1, standardize=True):
        """
        针对同一个批次的像素一次性做 S1 的随机采样 (asc + desc 合并)。
          s1_asc_bands_batch.shape = (B, t_s1a, 2)
          s1_asc_doys_batch.shape  = (B, t_s1a)
          s1_desc_bands_batch.shape= (B, t_s1d, 2)
          s1_desc_doys_batch.shape = (B, t_s1d)
        返回: np.array, shape=(B, sample_size_s1, 4), dtype float32
        """
        B = s1_asc_bands_batch.shape[0]
        out_list = []
        for b in range(B):
            s1_bands_all = np.concatenate([s1_asc_bands_batch[b], s1_desc_bands_batch[b]], axis=0)  # shape (t_s1a+t_s1d, 2)
            s1_doys_all  = np.concatenate([s1_asc_doys_batch[b], s1_desc_doys_batch[b]], axis=0)

            valid_mask = np.any(s1_bands_all != 0, axis=-1)
            valid_idx = np.nonzero(valid_mask)[0]
            if len(valid_idx) == 0:
                # 如果该像素所有时刻均为 0，则使用所有时刻的索引（即采样到的结果仍为全 0）
                valid_idx = np.arange(s1_bands_all.shape[0])
            if len(valid_idx) < sample_size_s1:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=True)
            else:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=False)
            idx_chosen = np.sort(idx_chosen)

            sub_bands = s1_bands_all[idx_chosen, :]  # (sample_size_s1, 2)
            sub_doys  = s1_doys_all[idx_chosen]

            if standardize:
                sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

            doys_norm = sub_doys / 365.0
            sin_doy = np.sin(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
            cos_doy = np.cos(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)

            out_arr = np.hstack([sub_bands, sin_doy, cos_doy])  # (sample_size_s1, 4)
            out_list.append(out_arr.astype(np.float32))

        return np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s1, 4)

    # Step7: 推理循环 (只需要 1 个 epoch)
    local_results = []
    start_time = time.time()

    logging.info(f"[Inference] loader length: {len(loader)}")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            global_idxs = batch_data["global_idx"]  # shape=(B,)

            # ---- 将 batch_data 中的 numpy 数据取出来 (B, T, dim)  ----
            s2_bands_batch = batch_data["s2_bands"].numpy()  # (B, t_s2, 10)
            s2_masks_batch = batch_data["s2_masks"].numpy()  # (B, t_s2)
            s2_doys_batch  = batch_data["s2_doys"].numpy()   # (B, t_s2)

            s1_asc_bands_batch = batch_data["s1_asc_bands"].numpy()   # (B, t_s1a, 2)
            s1_asc_doys_batch  = batch_data["s1_asc_doys"].numpy()    # (B, t_s1a)
            s1_desc_bands_batch= batch_data["s1_desc_bands"].numpy()   # (B, t_s1d, 2)
            s1_desc_doys_batch = batch_data["s1_desc_doys"].numpy()    # (B, t_s1d)

            B = s2_bands_batch.shape[0]
            sum_repr = None

            for r in range(config["repeat_times"]):
                # S2
                s2_input_np = sample_s2_batch(
                    s2_bands_batch, s2_masks_batch, s2_doys_batch,
                    band_mean=dataset.s2_band_mean,
                    band_std=dataset.s2_band_std,
                    sample_size_s2=config["sample_size_s2"],
                    standardize=True
                )  # (B, sample_size_s2, 12) float32

                # S1
                s1_input_np = sample_s1_batch(
                    s1_asc_bands_batch, s1_asc_doys_batch,
                    s1_desc_bands_batch, s1_desc_doys_batch,
                    band_mean=dataset.s1_band_mean,
                    band_std=dataset.s1_band_std,
                    sample_size_s1=config["sample_size_s1"],
                    standardize=True
                )  # (B, sample_size_s1, 4) float32

                # 转成 Tensor => [B, sample_size_s2, 12]
                s2_input = torch.tensor(s2_input_np, dtype=torch.float32, device=device)
                s1_input = torch.tensor(s1_input_np, dtype=torch.float32, device=device)

                # 前向传播
                z = model(s2_input, s1_input)  # shape: (B, latent_dim) 或 (B, 2*latent_dim)

                if sum_repr is None:
                    sum_repr = z
                else:
                    sum_repr += z

            avg_repr = sum_repr / float(config["repeat_times"])

            # ---- 保存到 local_results 中
            avg_repr_np = avg_repr.cpu().numpy()  # (B, latent_dim)
            global_idxs_list = global_idxs.tolist()

            for b in range(B):
                gidx = global_idxs_list[b]
                local_results.append((gidx, avg_repr_np[b]))

            if batch_idx % 2 == 0:
                logging.info(f"[Inference] batch_idx={batch_idx}, local_results.size={len(local_results)}")

    # Step8: 结果汇总并写入 npy 文件（单 GPU，无需 all_gather）
    final_gidx_np = np.array([item[0] for item in local_results], dtype=np.int64)
    final_vecs_np = np.array([item[1] for item in local_results], dtype=np.float32)

    logging.info(f"final_gidx_np.shape={final_gidx_np.shape}, final_vecs_np.shape={final_vecs_np.shape}")

    H, W = dataset.H, dataset.W
    latent_dim = final_vecs_np.shape[1]

    out_array = np.full((H * W, latent_dim), 0, dtype=np.float32)
    out_array[final_gidx_np] = final_vecs_np
    out_array = out_array.reshape(H, W, latent_dim)

    np.save(config["output_npy"], out_array)
    logging.info(f"Saved final representation to {config["output_npy"]}, shape={out_array.shape}")

if __name__ == "__main__":
    main()
