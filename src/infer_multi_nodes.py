#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import argparse
import logging
import gc
import socket
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch

from models.modules import TransformerEncoder, ProjectionHead
from models.ssl_model import MultimodalBTModel, MultimodalBTInferenceModel
from models.builder import build_ssl_model

# 从 datasets.ssl_dataset 导入 SingleTileInferenceDataset（请确保该版本支持预切块后的文件）
from datasets.ssl_dataset import SingleTileInferenceDataset

import importlib.util
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
    parser = argparse.ArgumentParser(description="Multi-node Multi-XPU Inference with Per-Node Pre-Chucking")
    parser.add_argument('--config', type=str, default="configs/infer_config.py", 
                        help="Path to the config file")
    return parser.parse_args()

def init_distributed():
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    rank = int(os.environ.get("PMI_RANK", "-1"))
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    dist.init_process_group(
        backend="ccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    return rank, world_size, local_rank, gpus_per_node

def create_chunk_folder(base_folder, timestamp):
    chunk_folder = os.path.join(base_folder, f"{timestamp}_chunk_file")
    os.makedirs(chunk_folder, exist_ok=True)
    return chunk_folder

def pre_chunk_files(tile_path, chunk_folder, max_file_size):
    """
    对 tile_path 下的 bands.npy、masks.npy、sar_ascending.npy 和 sar_descending.npy
    按 H 方向切块，切块数根据 bands.npy 的文件大小和 max_file_size（单位：GB）确定。
    切块文件保存为 chunk_{i}_bands.npy 等，放入 chunk_folder 中。
    返回切块的边界列表。
    """
    logger = logging.getLogger()
    bands_file = os.path.join(tile_path, "bands.npy")
    masks_file = os.path.join(tile_path, "masks.npy")
    sar_asc_file = os.path.join(tile_path, "sar_ascending.npy")
    sar_desc_file = os.path.join(tile_path, "sar_descending.npy")
    # 使用 np.load 的 mmap_mode 读取大文件（仅获取头信息和切片）
    bands_arr = np.load(bands_file, mmap_mode='r')
    masks_arr = np.load(masks_file, mmap_mode='r')
    sar_asc_arr = np.load(sar_asc_file, mmap_mode='r')
    sar_desc_arr = np.load(sar_desc_file, mmap_mode='r')
    # 假设所有文件在 H 方向尺寸一致
    _, H, W, _ = bands_arr.shape
    total_size_gb = os.path.getsize(bands_file) / (1024**3)
    num_chunks = math.ceil(total_size_gb / max_file_size)
    num_chunks = max(1, num_chunks)
    logger.info(f"[Pre-chunk] bands.npy size: {total_size_gb:.2f} GB, max_file_size: {max_file_size} GB, num_chunks: {num_chunks}")
    chunk_height = math.ceil(H / num_chunks)
    chunk_boundaries = []
    for i in range(num_chunks):
        start = i * chunk_height
        end = min((i+1) * chunk_height, H)
        chunk_boundaries.append((start, end))
    logger.info(f"[Pre-chunk] Chunk boundaries (H axis): {chunk_boundaries}")
    for chunk_id, (start, end) in enumerate(chunk_boundaries):
        logger.info(f"[Pre-chunk] Processing chunk {chunk_id}: rows {start} to {end}")
        bands_chunk = bands_arr[:, start:end, :, :].copy()
        masks_chunk = masks_arr[:, start:end, :].copy()
        sar_asc_chunk = sar_asc_arr[:, start:end, :, :].copy()
        sar_desc_chunk = sar_desc_arr[:, start:end, :, :].copy()
        np.save(os.path.join(chunk_folder, f"chunk_{chunk_id}_bands.npy"), bands_chunk)
        np.save(os.path.join(chunk_folder, f"chunk_{chunk_id}_masks.npy"), masks_chunk)
        np.save(os.path.join(chunk_folder, f"chunk_{chunk_id}_sar_ascending.npy"), sar_asc_chunk)
        np.save(os.path.join(chunk_folder, f"chunk_{chunk_id}_sar_descending.npy"), sar_desc_chunk)
        logger.info(f"[Pre-chunk] Saved chunk {chunk_id} files.")
    return chunk_boundaries

def main():
    args = parse_args()
    config_module = load_config_module(args.config)
    config = config_module.config

    rank, world_size, local_rank, gpus_per_node = init_distributed()
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO if rank==0 else logging.WARN,
                        format=f"%(asctime)s [Rank {rank}] %(levelname)s: %(message)s")

    # 输出每个进程所在节点信息
    hostname = socket.gethostname()
    logger.info(f"Process rank {rank}, local_rank {local_rank} running on hostname: {hostname}")

    # 设置当前设备
    current_xpu = f"xpu:{local_rank}"
    torch.xpu.set_device(current_xpu)
    device = torch.device(current_xpu)

    tile_path = config["tile_path"]
    base_temp_folder = "/local/zf281"
    # 全局生成统一的 timestamp（由全局 rank0生成，然后广播给所有进程）
    if rank == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = "0"
    timestamp_tensor = torch.tensor(bytearray(timestamp, 'utf-8'), dtype=torch.uint8, device=device)
    max_len = 64
    pad = torch.zeros(max_len - timestamp_tensor.numel(), dtype=torch.uint8, device=device)
    timestamp_tensor = torch.cat([timestamp_tensor, pad])
    dist.broadcast(timestamp_tensor, src=0)
    timestamp = bytearray(timestamp_tensor.cpu().numpy()).decode('utf-8').strip('\x00')
    logger.info(f"Using common timestamp: {timestamp}")

    # 每个节点在自己的 /local 下创建临时 chunk 文件夹（目录名一致，但位于各自节点）
    node_chunk_folder = create_chunk_folder(base_temp_folder, timestamp)
    logger.info(f"Node ({hostname}) chunk folder: {node_chunk_folder}")

    # 每个节点选择 local_rank==0 进程进行预切块，其它进程等待本节点中切块完成
    boundaries_file = os.path.join(node_chunk_folder, "chunk_boundaries.npy")
    if local_rank == 0:
        logger.info("Starting pre-chunking on this node.")
        max_file_size = config.get("max_file_size", 10)  # 单位：GB
        chunk_boundaries = pre_chunk_files(tile_path, node_chunk_folder, max_file_size)
        np.save(boundaries_file, np.array(chunk_boundaries))
        logger.info("Pre-chunking finished on this node.")
    else:
        logger.info("Waiting for pre-chunking to complete on this node...")
        while not os.path.exists(boundaries_file):
            time.sleep(1)
    # 本节点所有进程加载本节点的 chunk boundaries
    chunk_boundaries = np.load(boundaries_file, allow_pickle=True)
    num_chunks = len(chunk_boundaries)
    logger.info(f"Loaded chunk boundaries on node ({hostname}): {chunk_boundaries}")

    # 构建模型并加载 checkpoint（所有节点共用同一模型）
    logger.info("Building model and loading checkpoint.")
    ssl_model = build_ssl_model(config, device)
    logger.info("Before loading checkpoint, s2_backbone weights:")
    logger.info(ssl_model.s2_backbone.fc_out.weight)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
    ssl_model.load_state_dict(checkpoint[state_key], strict=True)
    logger.info("After loading checkpoint, s2_backbone weights:")
    logger.info(ssl_model.s2_backbone.fc_out.weight)
    for p in ssl_model.s2_backbone.parameters():
        p.requires_grad = False
    for p in ssl_model.s1_backbone.parameters():
        p.requires_grad = False
    model = MultimodalBTInferenceModel(
        s2_backbone=ssl_model.s2_backbone,
        s1_backbone=ssl_model.s1_backbone,
        fusion_method=config["fusion_method"]
    ).to(device)
    model.eval()

    # 用于保存各个 chunk 推理结果的 representation chunk 文件（输出目录在 config 中指定）
    representation_chunk_files = []

    # 对每个切块依次处理
    for chunk_id, (start_row, end_row) in enumerate(chunk_boundaries):
        logger.info(f"Processing chunk {chunk_id+1}/{num_chunks}, rows {start_row} to {end_row} on node ({hostname})")
        # 构造数据集时传入本节点的 chunk_folder、chunk_id 和起始行号
        dataset = SingleTileInferenceDataset(
            tile_path=tile_path,
            chunk_folder=node_chunk_folder,
            chunk_id=chunk_id,
            start_row=start_row,
            min_valid_timesteps=config["min_valid_timesteps"],
            standardize=False
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            sampler=sampler,
            num_workers=config.get("num_workers", 0),
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"[Chunk {chunk_id}] DataLoader length: {len(loader)}")
        gidx_list = []
        vecs_list = []
        chunk_start_time = time.time()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                global_idxs = batch_data["global_idx"].numpy()  # (B,)
                s2_bands_batch = batch_data["s2_bands"].numpy()
                s2_masks_batch = batch_data["s2_masks"].numpy()
                s2_doys_batch  = batch_data["s2_doys"].numpy()
                s1_asc_bands_batch  = batch_data["s1_asc_bands"].numpy()
                s1_asc_doys_batch   = batch_data["s1_asc_doys"].numpy()
                s1_desc_bands_batch = batch_data["s1_desc_bands"].numpy()
                s1_desc_doys_batch  = batch_data["s1_desc_doys"].numpy()
                B = s2_bands_batch.shape[0]
                sum_repr = None
                # 定义随机采样函数（保持原有逻辑）
                def sample_s2_batch(s2_bands_batch, s2_masks_batch, s2_doys_batch,
                                    band_mean, band_std, sample_size_s2, standardize=True):
                    B_local = s2_bands_batch.shape[0]
                    out_list = []
                    for b in range(B_local):
                        valid_idx = np.nonzero(s2_masks_batch[b])[0]
                        if len(valid_idx) == 0:
                            valid_idx = np.arange(s2_masks_batch.shape[1])
                        if len(valid_idx) < sample_size_s2:
                            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=True)
                        else:
                            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=False)
                        idx_chosen = np.sort(idx_chosen)
                        sub_bands = s2_bands_batch[b, idx_chosen, :]
                        sub_doys  = s2_doys_batch[b, idx_chosen]
                        if standardize:
                            sub_bands = (sub_bands - dataset.s2_band_mean) / (dataset.s2_band_std + 1e-9)
                        doys_norm = sub_doys / 365.0
                        sin_doy = np.sin(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
                        cos_doy = np.cos(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
                        out_arr = np.hstack([sub_bands, sin_doy, cos_doy])
                        out_list.append(out_arr.astype(np.float32))
                    return np.stack(out_list, axis=0).astype(np.float32)

                def sample_s1_batch(s1_asc_bands_batch, s1_asc_doys_batch,
                                    s1_desc_bands_batch, s1_desc_doys_batch,
                                    band_mean, band_std, sample_size_s1, standardize=True):
                    B_local = s1_asc_bands_batch.shape[0]
                    out_list = []
                    for b in range(B_local):
                        s1_bands_all = np.concatenate([s1_asc_bands_batch[b], s1_desc_bands_batch[b]], axis=0)
                        s1_doys_all  = np.concatenate([s1_asc_doys_batch[b], s1_desc_doys_batch[b]], axis=0)
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
                            sub_bands = (sub_bands - dataset.s1_band_mean) / (dataset.s1_band_std + 1e-9)
                        doys_norm = sub_doys / 365.0
                        sin_doy = np.sin(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
                        cos_doy = np.cos(2 * np.pi * doys_norm).astype(np.float32).reshape(-1, 1)
                        out_arr = np.hstack([sub_bands, sin_doy, cos_doy])
                        out_list.append(out_arr.astype(np.float32))
                    return np.stack(out_list, axis=0).astype(np.float32)

                s2_in_np = sample_s2_batch(
                    s2_bands_batch, s2_masks_batch, s2_doys_batch,
                    dataset.s2_band_mean, dataset.s2_band_std,
                    config["sample_size_s2"], True
                )
                s1_in_np = sample_s1_batch(
                    s1_asc_bands_batch, s1_asc_doys_batch,
                    s1_desc_bands_batch, s1_desc_doys_batch,
                    dataset.s1_band_mean, dataset.s1_band_std,
                    config["sample_size_s1"], True
                )
                s2_input = torch.tensor(s2_in_np, dtype=torch.float32, device=device)
                s1_input = torch.tensor(s1_in_np, dtype=torch.float32, device=device)
                z = model(s2_input, s1_input)
                if sum_repr is None:
                    sum_repr = z
                else:
                    sum_repr += z
                avg_repr = sum_repr / float(config["repeat_times"])
                avg_repr_np = avg_repr.cpu().numpy()
                gidx_list.append(global_idxs)
                vecs_list.append(avg_repr_np)
                if rank == 0 and batch_idx % 10 == 0:
                    logger.info(f"[Chunk {chunk_id}] Batch {batch_idx}, accumulated batches: {len(gidx_list)}")
        chunk_end_time = time.time()
        logger.info(f"[Rank {rank}] Finished inference for chunk {chunk_id} in {chunk_end_time - chunk_start_time:.2f} seconds")
        if len(gidx_list) > 0:
            local_gidx_np = np.concatenate(gidx_list, axis=0)
            local_vecs_np = np.concatenate(vecs_list, axis=0)
        else:
            local_gidx_np = np.zeros((0,), dtype=np.int64)
            local_vecs_np = np.zeros((0, config["latent_dim"]), dtype=np.float32)
        local_size = local_gidx_np.shape[0]
        local_dim = local_vecs_np.shape[1] if local_size > 0 else config["latent_dim"]
        logger.info(f"[Chunk {chunk_id}] After accumulation, local_gidx_np.shape={local_gidx_np.shape}, local_vecs_np.shape={local_vecs_np.shape}")
        local_size_t = torch.tensor([local_size], dtype=torch.long, device=device)
        size_list_t = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        logger.info(f"[Chunk {chunk_id}] Starting all_gather for local sizes")
        dist.all_gather(size_list_t, local_size_t)
        size_list_cpu = [int(t.item()) for t in size_list_t]
        max_size_chunk = max(size_list_cpu)
        logger.info(f"[Chunk {chunk_id}] All_gather sizes: {size_list_cpu}, max_size_chunk: {max_size_chunk}")
        if local_size < max_size_chunk:
            pad_len = max_size_chunk - local_size
            pad_idx = np.full((pad_len,), -1, dtype=np.int64)
            pad_vec = np.zeros((pad_len, local_dim), dtype=np.float32)
            local_gidx_np = np.concatenate([local_gidx_np, pad_idx], axis=0)
            local_vecs_np = np.concatenate([local_vecs_np, pad_vec], axis=0)
        local_gidx_t = torch.from_numpy(local_gidx_np).to(device)
        local_vecs_t = torch.from_numpy(local_vecs_np).to(device)
        gather_gidx_list = [torch.zeros((max_size_chunk,), dtype=torch.long, device=device) for _ in range(world_size)]
        gather_vecs_list = [torch.zeros((max_size_chunk, local_dim), dtype=torch.float32, device=device) for _ in range(world_size)]
        logger.info(f"[Chunk {chunk_id}] Starting all_gather for gidx and vecs")
        dist.all_gather(gather_gidx_list, local_gidx_t)
        dist.all_gather(gather_vecs_list, local_vecs_t)
        logger.info(f"[Chunk {chunk_id}] Finished all_gather for gidx and vecs")
        if rank == 0:
            final_gidx_list = []
            final_vecs_list = []
            for r in range(world_size):
                real_sz = size_list_cpu[r]
                if real_sz > 0:
                    rank_gidx_np = gather_gidx_list[r][:real_sz].cpu().numpy()
                    rank_vecs_np = gather_vecs_list[r][:real_sz, :local_dim].cpu().numpy()
                    final_gidx_list.append(rank_gidx_np)
                    final_vecs_list.append(rank_vecs_np)
                    logger.info(f"[Chunk {chunk_id}] Gather from rank {r}, real_sz={real_sz}, rank_vecs_np.shape={rank_vecs_np.shape}")
                else:
                    logger.info(f"[Chunk {chunk_id}] Gather from rank {r}, real_sz=0, skip")
            if len(final_gidx_list) == 0:
                logger.info(f"[Chunk {chunk_id}] All ranks have zero data, nothing to save for this chunk.")
                chunk_output = np.zeros((end_row - start_row, dataset.W, local_dim), dtype=np.float32)
            else:
                final_gidx = np.concatenate(final_gidx_list, axis=0)
                final_vecs = np.concatenate(final_vecs_list, axis=0)
                chunk_out_array = np.full(((end_row - start_row) * dataset.W, local_dim), 0, dtype=np.float32)
                # chunk_out_array[final_gidx] = final_vecs
                final_gidx = final_gidx - (start_row * dataset.W)
                chunk_out_array[final_gidx] = final_vecs
                chunk_output = chunk_out_array.reshape((end_row - start_row, dataset.W, local_dim))
            chunk_file = f"{os.path.splitext(config['output_npy'])[0]}_chunk_{chunk_id}.npy"
            np.save(chunk_file, chunk_output)
            logger.info(f"[Chunk {chunk_id}] Saved chunk output to {chunk_file} with shape {chunk_output.shape}")
            representation_chunk_files.append(chunk_file)
        del dataset, loader, gidx_list, vecs_list, local_gidx_np, local_vecs_np
        gc.collect()
        dist.barrier()

    # 所有 chunk 完成后，由全局 rank0 合并各节点的 representation chunk 文件（输出目录为共享目录）
    if rank == 0:
        final_chunks = []
        for chunk_id in range(num_chunks):
            chunk_file = f"{os.path.splitext(config['output_npy'])[0]}_chunk_{chunk_id}.npy"
            logger.info(f"Loading representation chunk file {chunk_file}")
            chunk_data = np.load(chunk_file)
            final_chunks.append(chunk_data)
        final_representation = np.vstack(final_chunks)
        logger.info(f"Final representation shape: {final_representation.shape}")
        np.save(config["output_npy"], final_representation)
        logger.info(f"Saved final representation to {config['output_npy']}")
        for chunk_id in range(num_chunks):
            chunk_file = f"{os.path.splitext(config['output_npy'])[0]}_chunk_{chunk_id}.npy"
            os.remove(chunk_file)
            logger.info(f"Removed temporary chunk file {chunk_file}")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
