#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Run command:
# torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 src/infer_multi_gpu.py

import os
import sys
# Add the workspace directory to Python's path
sys.path.insert(0, os.getcwd())
import math
import time
import argparse
import logging
import numpy as np
from datetime import timedelta

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from models.modules import TransformerEncoder, ProjectionHead
from models.ssl_model import MultimodalBTModel, MultimodalBTInferenceModel
from models.builder import build_ssl_model
from datasets.ssl_dataset import SingleTileInferenceDataset
import importlib.util

# Set environment variables to avoid issues with AMD GPUs
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def load_config_module(config_file_path):
    spec = importlib.util.spec_from_file_location("my_dynamic_config", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["my_dynamic_config"] = config_module
    spec.loader.exec_module(config_module)
    return config_module

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Inference")
    parser.add_argument('--config', type=str, default="configs/infer_multi_gpu_config.py", 
                        help="Path to config file (e.g. configs/infer_multi_gpu_config.py)")
    return parser.parse_args()

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.3f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.3f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.3f}s"

def main():
    # Track total execution time
    total_start_time = time.time()
    
    # Initialize distributed environment
    dist_start_time = time.time()
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist_time = time.time() - dist_start_time

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Configure logging
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"⏱️ Distributed initialization completed in {format_time(dist_time)}")
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse arguments and load config
    config_start_time = time.time()
    args = parse_args()
    config_module = load_config_module(args.config)
    config = config_module.config
    config_time = time.time() - config_start_time
    
    # Print config (only on rank 0)
    if local_rank == 0:
        logging.info(f"⏱️ Config loading completed in {format_time(config_time)}")
        logging.info(f"Running on {world_size} GPU(s). LocalRank={local_rank}")
        logging.info("Configurations:")
        for k, v in config.items():
            logging.info(f"  {k}: {v}")

    # Measure dataset loading time
    dataset_start_time = time.time()
    
    # Construct dataset
    dataset = SingleTileInferenceDataset(
        tile_path=config["tile_path"],
        min_valid_timesteps=config["min_valid_timesteps"],
        standardize=False  # Standardize during sampling
    )

    # Use DistributedSampler to partition the data
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
        drop_last=False
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False
    )
    
    dataset_time = time.time() - dataset_start_time
    if local_rank == 0:
        logging.info(f"⏱️ Dataset preparation completed in {format_time(dataset_time)}")

    # Track model loading time
    model_start_time = time.time()
    
    # Build and load SSL model
    ssl_model = build_ssl_model(config, device)

    if local_rank == 0:
        logging.info("Before loading checkpoint, s2_backbone weights:")
        logging.info(ssl_model.s2_backbone.embedding[0].weight)
        logging.info(f"Loading SSL checkpoint from {config['checkpoint_path']}")
    
    checkpoint_start_time = time.time()
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    checkpoint_load_time = time.time() - checkpoint_start_time
    
    if local_rank == 0:
        logging.info(f"⏱️ Checkpoint loading completed in {format_time(checkpoint_load_time)}")
    
    state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
    
    # Handle FSDP state dict
    state_dict = checkpoint[state_key]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    state_dict_start_time = time.time()
    ssl_model.load_state_dict(new_state_dict, strict=True)
    state_dict_time = time.time() - state_dict_start_time
    
    if local_rank == 0:
        logging.info(f"⏱️ State dict loading completed in {format_time(state_dict_time)}")
        logging.info("SSL backbone weights after loading checkpoint (s2 backbone):")
        logging.info(ssl_model.s2_backbone.embedding[0].weight)
    
    # Freeze model parameters
    for param in ssl_model.s2_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.s1_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.dim_reducer.parameters():
        param.requires_grad = False
    
    # Construct inference model
    model = MultimodalBTInferenceModel(
        s2_backbone=ssl_model.s2_backbone,
        s1_backbone=ssl_model.s1_backbone,
        fusion_method=config["fusion_method"],
        dim_reducer=ssl_model.dim_reducer,
    ).to(device)
    
    if local_rank == 0:
        logging.info("Inference model constructed. s2 backbone weights:")
        logging.info(model.s2_backbone.embedding[0].weight)
    
    model.eval()
    
    model_time = time.time() - model_start_time
    if local_rank == 0:
        logging.info(f"⏱️ Model preparation completed in {format_time(model_time)}")

    # Vectorized batch sampling helper functions
    def sample_s2_batch(s2_bands_batch, s2_masks_batch, s2_doys_batch,
                        band_mean, band_std, sample_size_s2, standardize=True):
        """
        Perform S2 random sampling for a batch of pixels.
        Vectorized implementation for improved performance.
        """
        B = s2_bands_batch.shape[0]
        C = s2_bands_batch.shape[2]  # Number of channels/bands
        
        # Pre-allocate output array
        out_array = np.zeros((B, sample_size_s2, C + 1), dtype=np.float32)
        
        for b in range(B):
            valid_idx = np.nonzero(s2_masks_batch[b])[0]
            
            if len(valid_idx) == 0:
                valid_idx = np.arange(s2_bands_batch.shape[1])
            
            if len(valid_idx) < sample_size_s2:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=True)
            else:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=False)
            idx_chosen = np.sort(idx_chosen)

            # Directly assign to pre-allocated array
            out_array[b, :, :C] = s2_bands_batch[b, idx_chosen, :]
            out_array[b, :, C] = s2_doys_batch[b, idx_chosen]
        
        # Standardize entire batch at once
        if standardize:
            out_array[:, :, :C] = (out_array[:, :, :C] - band_mean) / (band_std + 1e-9)
        
        return out_array

    def sample_s1_batch(s1_asc_bands_batch, s1_asc_doys_batch,
                        s1_desc_bands_batch, s1_desc_doys_batch,
                        band_mean, band_std, sample_size_s1, standardize=True):
        """
        Perform S1 random sampling for a batch of pixels (combining asc + desc).
        Vectorized implementation for improved performance.
        """
        B = s1_asc_bands_batch.shape[0]
        C = s1_asc_bands_batch.shape[2]  # Number of channels/bands
        
        # Pre-allocate output array
        out_array = np.zeros((B, sample_size_s1, C + 1), dtype=np.float32)
        
        for b in range(B):
            # Concatenate ascending and descending data
            s1_bands_all = np.concatenate([s1_asc_bands_batch[b], s1_desc_bands_batch[b]], axis=0)
            s1_doys_all = np.concatenate([s1_asc_doys_batch[b], s1_desc_doys_batch[b]], axis=0)

            valid_mask = np.any(s1_bands_all != 0, axis=-1)
            valid_idx = np.nonzero(valid_mask)[0]
            
            if len(valid_idx) == 0:
                valid_idx = np.arange(s1_bands_all.shape[0])
            
            if len(valid_idx) < sample_size_s1:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=True)
            else:
                idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=False)
            idx_chosen = np.sort(idx_chosen)

            # Directly assign to pre-allocated array
            out_array[b, :, :C] = s1_bands_all[idx_chosen, :]
            out_array[b, :, C] = s1_doys_all[idx_chosen]
        
        # Standardize entire batch at once
        if standardize:
            out_array[:, :, :C] = (out_array[:, :, :C] - band_mean) / (band_std + 1e-9)
        
        return out_array

    # Inference loop
    local_results = []
    inference_start_time = time.time()
    
    # Track timing for different stages
    total_data_loading_time = 0
    total_s2_sampling_time = 0
    total_s1_sampling_time = 0
    total_tensor_conversion_time = 0
    total_inference_time = 0
    total_postprocessing_time = 0
    total_batch_time = 0
    
    batch_count = 0
    batch_end_time = time.time()  # Initialize for first batch

    if local_rank == 0:
        logging.info(f"[Inference] Starting inference with loader length: {len(loader)}")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            batch_start_time = time.time()
            
            # Track data loading time (from last batch to current batch data arrival)
            if batch_idx > 0:
                data_loading_time = batch_start_time - batch_end_time
                total_data_loading_time += data_loading_time
                if local_rank == 0 and batch_idx % 10 == 0:
                    logging.info(f"⏱️ Batch {batch_idx-1} → {batch_idx} data loading time: {format_time(data_loading_time)}")
            
            global_idxs = batch_data["global_idx"]  # shape=(B,)

            # Extract numpy data from batch
            extract_start_time = time.time()
            s2_bands_batch = batch_data["s2_bands"].numpy()
            s2_masks_batch = batch_data["s2_masks"].numpy()
            s2_doys_batch = batch_data["s2_doys"].numpy()

            s1_asc_bands_batch = batch_data["s1_asc_bands"].numpy()
            s1_asc_doys_batch = batch_data["s1_asc_doys"].numpy()
            s1_desc_bands_batch = batch_data["s1_desc_bands"].numpy()
            s1_desc_doys_batch = batch_data["s1_desc_doys"].numpy()
            extract_time = time.time() - extract_start_time

            B = s2_bands_batch.shape[0]
            sum_repr = None

            for r in range(config["repeat_times"]):
                # Sample S2 data
                s2_sampling_start = time.time()
                s2_input_np = sample_s2_batch(
                    s2_bands_batch, s2_masks_batch, s2_doys_batch,
                    band_mean=dataset.s2_band_mean,
                    band_std=dataset.s2_band_std,
                    sample_size_s2=config["sample_size_s2"],
                    standardize=True
                )
                s2_sampling_time = time.time() - s2_sampling_start
                total_s2_sampling_time += s2_sampling_time

                # Sample S1 data
                s1_sampling_start = time.time()
                s1_input_np = sample_s1_batch(
                    s1_asc_bands_batch, s1_asc_doys_batch,
                    s1_desc_bands_batch, s1_desc_doys_batch,
                    band_mean=dataset.s1_band_mean,
                    band_std=dataset.s1_band_std,
                    sample_size_s1=config["sample_size_s1"],
                    standardize=True
                )
                s1_sampling_time = time.time() - s1_sampling_start
                total_s1_sampling_time += s1_sampling_time

                # Convert to tensors
                tensor_start_time = time.time()
                s2_input = torch.tensor(s2_input_np, dtype=torch.float32, device=device)
                s1_input = torch.tensor(s1_input_np, dtype=torch.float32, device=device)
                tensor_time = time.time() - tensor_start_time
                total_tensor_conversion_time += tensor_time

                # Forward pass
                inference_start = time.time()
                z = model(s2_input, s1_input)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                if sum_repr is None:
                    sum_repr = z
                else:
                    sum_repr += z

            # Postprocessing (averaging)
            postprocess_start = time.time()
            avg_repr = sum_repr / float(config["repeat_times"])
            avg_repr_np = avg_repr.cpu().numpy()
            global_idxs_list = global_idxs.tolist()

            for b in range(B):
                gidx = global_idxs_list[b]
                local_results.append((gidx, avg_repr_np[b]))
            postprocess_time = time.time() - postprocess_start
            total_postprocessing_time += postprocess_time
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            total_batch_time += batch_time
            batch_count += 1

            if batch_idx % 10 == 0:
                if local_rank == 0:
                    logging.info(f"[Inference] batch_idx={batch_idx}, local_results.size={len(local_results)}")
                    logging.info(f"⏱️ Batch {batch_idx} timing breakdown:")
                    logging.info(f"    Extract data:  {format_time(extract_time)}")
                    logging.info(f"    S2 sampling:   {format_time(s2_sampling_time)}")
                    logging.info(f"    S1 sampling:   {format_time(s1_sampling_time)}")
                    logging.info(f"    To tensor:     {format_time(tensor_time)}")
                    logging.info(f"    Inference:     {format_time(inference_time)}")
                    logging.info(f"    Postprocess:   {format_time(postprocess_time)}")
                    logging.info(f"    Total batch:   {format_time(batch_time)}")
                    
                    # Calculate and log average timing every 10 batches
                    if batch_count > 0:
                        avg_batch_time = total_batch_time / batch_count
                        avg_data_loading_time = total_data_loading_time / max(1, batch_count - 1)
                        avg_s2_sampling_time = total_s2_sampling_time / batch_count
                        avg_s1_sampling_time = total_s1_sampling_time / batch_count
                        avg_tensor_conversion_time = total_tensor_conversion_time / batch_count
                        avg_inference_time = total_inference_time / batch_count
                        avg_postprocessing_time = total_postprocessing_time / batch_count
                        
                        logging.info(f"⏱️ Average timing after {batch_count} batches:")
                        logging.info(f"    Data loading:      {format_time(avg_data_loading_time)} ({avg_data_loading_time/avg_batch_time*100:.1f}%)")
                        logging.info(f"    S2 sampling:       {format_time(avg_s2_sampling_time)} ({avg_s2_sampling_time/avg_batch_time*100:.1f}%)")
                        logging.info(f"    S1 sampling:       {format_time(avg_s1_sampling_time)} ({avg_s1_sampling_time/avg_batch_time*100:.1f}%)")
                        logging.info(f"    Tensor conversion: {format_time(avg_tensor_conversion_time)} ({avg_tensor_conversion_time/avg_batch_time*100:.1f}%)")
                        logging.info(f"    Inference:         {format_time(avg_inference_time)} ({avg_inference_time/avg_batch_time*100:.1f}%)")
                        logging.info(f"    Postprocessing:    {format_time(avg_postprocessing_time)} ({avg_postprocessing_time/avg_batch_time*100:.1f}%)")
                        logging.info(f"    Total batch:       {format_time(avg_batch_time)} (100%)")
                        logging.info(f"    Processed {batch_count} batches with batch size {config['batch_size']}")
                else:
                    logging.debug(f"[Rank {local_rank}] batch_idx={batch_idx}, local_results.size={len(local_results)}")

    inference_time = time.time() - inference_start_time
    
    # Final summary statistics
    if local_rank == 0 and batch_count > 0:
        avg_batch_time = total_batch_time / batch_count
        avg_data_loading_time = total_data_loading_time / max(1, batch_count - 1)
        avg_s2_sampling_time = total_s2_sampling_time / batch_count
        avg_s1_sampling_time = total_s1_sampling_time / batch_count
        avg_tensor_conversion_time = total_tensor_conversion_time / batch_count
        avg_inference_time = total_inference_time / batch_count
        avg_postprocessing_time = total_postprocessing_time / batch_count
        
        logging.info(f"⏱️ Inference completed in {format_time(inference_time)}")
        logging.info(f"⏱️ Final average timing per batch:")
        logging.info(f"    Data loading:      {format_time(avg_data_loading_time)} ({avg_data_loading_time/avg_batch_time*100:.1f}%)")
        logging.info(f"    S2 sampling:       {format_time(avg_s2_sampling_time)} ({avg_s2_sampling_time/avg_batch_time*100:.1f}%)")
        logging.info(f"    S1 sampling:       {format_time(avg_s1_sampling_time)} ({avg_s1_sampling_time/avg_batch_time*100:.1f}%)")
        logging.info(f"    Tensor conversion: {format_time(avg_tensor_conversion_time)} ({avg_tensor_conversion_time/avg_batch_time*100:.1f}%)")
        logging.info(f"    Inference:         {format_time(avg_inference_time)} ({avg_inference_time/avg_batch_time*100:.1f}%)")
        logging.info(f"    Postprocessing:    {format_time(avg_postprocessing_time)} ({avg_postprocessing_time/avg_batch_time*100:.1f}%)")
        logging.info(f"    Total batch:       {format_time(avg_batch_time)} (100%)")
        logging.info(f"    Processed {batch_count} batches with batch size {config['batch_size']}")

    # Collect results from all processes
    if local_rank == 0:
        logging.info(f"Rank {local_rank} completed inference with {len(local_results)} results")
    
    # Start measuring result collection time
    result_collection_start = time.time()
    
    # Convert local results to numpy arrays
    local_gidx = np.array([item[0] for item in local_results], dtype=np.int64)
    local_vecs = np.array([item[1] for item in local_results], dtype=np.float32)
    
    # Get counts from all processes
    count_start_time = time.time()
    local_count = torch.tensor([len(local_results)], dtype=torch.int64, device=device)
    all_counts = [torch.tensor([0], dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)
    count_time = time.time() - count_start_time
    
    if local_rank == 0:
        logging.info(f"⏱️ Count gathering time: {format_time(count_time)}")
        logging.info(f"Result counts per rank: {[c.item() for c in all_counts]}")
    
    # Gather all results to rank 0
    if local_rank == 0:
        # Initialize storage for gathered results
        total_count = sum(count.item() for count in all_counts)
        all_gidx = np.zeros(total_count, dtype=np.int64)
        all_vecs = np.zeros((total_count, local_vecs.shape[1]), dtype=np.float32)
        
        # Copy local results first
        copy_start_time = time.time()
        all_gidx[:len(local_results)] = local_gidx
        all_vecs[:len(local_results)] = local_vecs
        copy_time = time.time() - copy_start_time
        logging.info(f"⏱️ Local copy time: {format_time(copy_time)}")
        
        # Receive data from other ranks
        offset = len(local_results)
        total_recv_time = 0
        for rank in range(1, world_size):
            count = all_counts[rank].item()
            if count > 0:
                # Create tensor to receive data
                gidx_buffer = torch.zeros(count, dtype=torch.int64, device=device)
                vecs_buffer = torch.zeros((count, local_vecs.shape[1]), dtype=torch.float32, device=device)
                
                # Receive data
                recv_start = time.time()
                dist.recv(gidx_buffer, src=rank, tag=rank*2)
                dist.recv(vecs_buffer, src=rank, tag=rank*2+1)
                recv_time = time.time() - recv_start
                total_recv_time += recv_time
                
                logging.info(f"⏱️ Receiving data from rank {rank}: {format_time(recv_time)} for {count} items")
                
                # Copy to the final arrays
                copy_remote_start = time.time()
                all_gidx[offset:offset+count] = gidx_buffer.cpu().numpy()
                all_vecs[offset:offset+count] = vecs_buffer.cpu().numpy()
                copy_remote_time = time.time() - copy_remote_start
                logging.info(f"⏱️ Remote copy time (rank {rank}): {format_time(copy_remote_time)}")
                offset += count
        
        logging.info(f"⏱️ Total receive time: {format_time(total_recv_time)}")
        
        # Create the final output array
        H, W = dataset.H, dataset.W
        latent_dim = all_vecs.shape[1]
        
        logging.info(f"Assembling final output with shape: H={H}, W={W}, latent_dim={latent_dim}")
        assembly_start = time.time()
        out_array = np.full((H * W, latent_dim), 0, dtype=np.float32)
        out_array[all_gidx] = all_vecs
        out_array = out_array.reshape(H, W, latent_dim)
        assembly_time = time.time() - assembly_start
        logging.info(f"⏱️ Assembly time: {format_time(assembly_time)}")
        
        # Save output
        save_start = time.time()
        np.save(config["output_npy"], out_array)
        save_time = time.time() - save_start
        logging.info(f"⏱️ Save time: {format_time(save_time)}")
        logging.info(f"Saved final representation to {config['output_npy']}, shape={out_array.shape}")
    else:
        # Send results to rank 0
        if len(local_results) > 0:
            # Convert to tensors
            send_prep_start = time.time()
            gidx_tensor = torch.tensor(local_gidx, dtype=torch.int64, device=device)
            vecs_tensor = torch.tensor(local_vecs, dtype=torch.float32, device=device)
            send_prep_time = time.time() - send_prep_start
            
            # Send data
            send_start = time.time()
            dist.send(gidx_tensor, dst=0, tag=local_rank*2)
            dist.send(vecs_tensor, dst=0, tag=local_rank*2+1)
            send_time = time.time() - send_start
            
            logging.info(f"⏱️ [Rank {local_rank}] Send preparation: {format_time(send_prep_time)}")
            logging.info(f"⏱️ [Rank {local_rank}] Sending data to rank 0: {format_time(send_time)} for {len(local_results)} items")
    
    result_collection_time = time.time() - result_collection_start
    if local_rank == 0:
        logging.info(f"⏱️ Result collection and processing time: {format_time(result_collection_time)}")
    
    # Clean up
    dist.barrier()
    cleanup_start = time.time()
    dist.destroy_process_group()
    cleanup_time = time.time() - cleanup_start
    
    # Total time
    total_time = time.time() - total_start_time
    if local_rank == 0:
        logging.info(f"⏱️ Cleanup time: {format_time(cleanup_time)}")
        logging.info(f"⏱️ Total execution time: {format_time(total_time)}")
        logging.info(f"⏱️ Performance summary:")
        logging.info(f"    Distributed init:      {format_time(dist_time)} ({dist_time/total_time*100:.1f}%)")
        logging.info(f"    Config loading:        {format_time(config_time)} ({config_time/total_time*100:.1f}%)")
        logging.info(f"    Dataset preparation:   {format_time(dataset_time)} ({dataset_time/total_time*100:.1f}%)")
        logging.info(f"    Model preparation:     {format_time(model_time)} ({model_time/total_time*100:.1f}%)")
        logging.info(f"    Inference:             {format_time(inference_time)} ({inference_time/total_time*100:.1f}%)")
        logging.info(f"    Result collection:     {format_time(result_collection_time)} ({result_collection_time/total_time*100:.1f}%)")
        logging.info(f"    Cleanup:               {format_time(cleanup_time)} ({cleanup_time/total_time*100:.1f}%)")
        logging.info(f"    Total:                 {format_time(total_time)} (100%)")

if __name__ == "__main__":
    main()