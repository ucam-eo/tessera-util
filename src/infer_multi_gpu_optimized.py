#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Run command:
# torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 src/infer_multi_gpu_optimized.py

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
import gc
from functools import partial
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.modules import TransformerEncoder, ProjectionHead
from models.ssl_model import MultimodalBTModel, MultimodalBTInferenceModel
from models.builder import build_ssl_model
from datasets.ssl_dataset import OptimizedTileInferenceDataset
import importlib.util

# Set environment variables to avoid issues with AMD GPUs
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"  # Increased for better CPU utilization
os.environ["MKL_NUM_THREADS"] = "4"  # Increased for better CPU utilization

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
    parser.add_argument('--profile', action='store_true', help="Enable profiling")
    parser.add_argument('--cache-dir', type=str, default="data/downstream/austrian_crop/infer_cache", 
                        help="Directory to store/load dataset preprocessing cache")
    parser.add_argument('--use-ddp', action='store_true', help="Use DistributedDataParallel")
    return parser.parse_args()

def validate_tensor_for_miopen(tensor, name="tensor"):
    """
    Validates tensors for MIOpen compatibility, ensuring no NaNs, Infs, or all-zero rows.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Input tensor to validate
    name : str
        Name of the tensor for logging purposes
        
    Returns:
    --------
    torch.Tensor
        Validated tensor safe for MIOpen operations
    """
    B, T, C = tensor.shape
    device = tensor.device
    
    # Make a copy to avoid modifying the original
    validated = tensor.clone()
    
    # Check for NaNs or Infs
    has_nan = torch.isnan(validated).any()
    has_inf = torch.isinf(validated).any()
    
    if has_nan or has_inf:
        logging.warning(f"Found NaN or Inf in {name}")
        # Replace NaNs and Infs with small random values
        nan_mask = torch.isnan(validated)
        inf_mask = torch.isinf(validated)
        problem_mask = nan_mask | inf_mask
        validated[problem_mask] = torch.randn_like(validated[problem_mask]) * 0.01
    
    # Check for all-zero rows (entire feature vectors)
    zero_rows = torch.all(validated == 0, dim=2)
    
    if torch.any(zero_rows):
        logging.warning(f"Found all-zero feature vectors in {name}")
        for b in range(B):
            if torch.any(zero_rows[b]):
                # Replace all-zero rows with small random values
                zero_indices = torch.nonzero(zero_rows[b], as_tuple=True)[0]
                validated[b, zero_indices] = torch.randn(len(zero_indices), C, device=device) * 0.01
    
    # Check if any batch dimension has all zero rows
    if torch.any(torch.all(zero_rows, dim=1)):
        logging.warning(f"Found batches with all zero rows in {name}")
        all_zero_batches = torch.nonzero(torch.all(zero_rows, dim=1), as_tuple=True)[0]
        for b in all_zero_batches:
            # Fill the entire batch with small random values
            validated[b] = torch.randn(T, C, device=device) * 0.01
    
    return validated

def ensure_valid_input_for_gru(input_tensor, min_size=2):
    """
    Ensures tensor has valid dimensions for GRU processing.
    Particularly important for AMD GPUs with MIOpen.
    
    Parameters:
    -----------
    input_tensor : torch.Tensor
        Input tensor of shape (B, T, C)
    min_size : int
        Minimum sequence length required
        
    Returns:
    --------
    torch.Tensor
        Tensor with valid dimensions for GRU processing
    """
    B, T, C = input_tensor.shape
    device = input_tensor.device
    
    # For AMD GPUs with MIOpen, ensure minimum dimensions
    # MIOpen requires at least 4 timesteps for some configurations
    min_size = max(4, min_size)  
    
    # Check if sequence length is sufficient
    if T < min_size:
        # Create a new tensor with valid sequence length
        new_tensor = torch.zeros((B, min_size, C), dtype=input_tensor.dtype, device=device)
        
        # Fill the tensor with valid data
        if T > 0:
            # Copy existing data
            new_tensor[:, :T] = input_tensor
            
            # For remaining positions, repeat the first timestep or use small random values
            for t in range(T, min_size):
                if torch.all(input_tensor[:, 0] == 0):
                    # If first timestep is all zeros, use small random values
                    new_tensor[:, t] = torch.randn(B, C, device=device) * 0.01
                else:
                    # Otherwise repeat the first timestep
                    new_tensor[:, t] = input_tensor[:, 0]
        else:
            # If no valid timesteps, use small random values
            new_tensor = torch.randn((B, min_size, C), dtype=input_tensor.dtype, device=device) * 0.01
        
        # Ensure no all-zero rows that could cause MIOpen issues
        zero_rows = torch.all(new_tensor == 0, dim=2)
        for b in range(B):
            if torch.any(zero_rows[b]):
                # Replace all-zero rows with small random values
                zero_indices = torch.nonzero(zero_rows[b], as_tuple=True)[0]
                new_tensor[b, zero_indices] = torch.randn(len(zero_indices), C, device=device) * 0.01
        
        return new_tensor
    
    # Even if T >= min_size, still check for all-zero rows
    zero_rows = torch.all(input_tensor == 0, dim=2)
    if torch.any(zero_rows):
        # Create a copy to avoid modifying the input tensor in-place
        new_tensor = input_tensor.clone()
        for b in range(B):
            if torch.any(zero_rows[b]):
                # Replace all-zero rows with small random values
                zero_indices = torch.nonzero(zero_rows[b], as_tuple=True)[0]
                new_tensor[b, zero_indices] = torch.randn(len(zero_indices), C, device=device) * 0.01
        return new_tensor
    
    return input_tensor

def sample_s2_batch(s2_bands_batch, s2_masks_batch, s2_doys_batch,
                    band_mean, band_std, sample_size_s2, standardize=True):
    """
    Perform S2 random sampling for a batch of pixels.
    Vectorized implementation for better performance.
    With additional safety checks to prevent empty sequences.
    Enhanced for AMD GPU compatibility.
    """
    B = s2_bands_batch.shape[0]
    device = s2_bands_batch.device
    
    # Create arrays to store results
    out_tensor = torch.zeros((B, sample_size_s2, s2_bands_batch.shape[2] + 1), 
                            dtype=torch.float32, device=device)
    
    # Ensure minimum sequence length is at least 1
    min_seq_length = max(1, sample_size_s2)
    
    for b in range(B):
        valid_idx = torch.nonzero(s2_masks_batch[b], as_tuple=True)[0]
        
        # Always ensure we have at least some indices
        if len(valid_idx) == 0:
            # If no valid timesteps, create artificial indices
            valid_idx = torch.arange(min(s2_bands_batch.shape[1], min_seq_length), device=device)
            
            # If still no indices (empty tensor), create fake indices
            if len(valid_idx) == 0:
                valid_idx = torch.zeros(min_seq_length, dtype=torch.long, device=device)
        
        # Make sure valid_idx is not empty and has appropriate shape
        if len(valid_idx) < min_seq_length:
            # Repeat indices to ensure we have enough
            repeat_factor = (min_seq_length + len(valid_idx) - 1) // max(1, len(valid_idx))
            valid_idx = valid_idx.repeat(repeat_factor)[:min_seq_length]
            idx_chosen = valid_idx
        else:
            # Randomly select indices
            perm = torch.randperm(len(valid_idx), device=device)
            if len(perm) < min_seq_length:
                # Safety check if randperm fails (rare case)
                perm = torch.arange(len(valid_idx), device=device)
                if len(perm) < min_seq_length:
                    # If still not enough, repeat
                    perm = perm.repeat((min_seq_length + len(perm) - 1) // max(1, len(perm)))[:min_seq_length]
            
            idx_chosen = valid_idx[perm[:min_seq_length]]
        
        # Sort indices for better memory access patterns
        idx_chosen, _ = torch.sort(idx_chosen)
        
        # Ensure idx_chosen is within valid range
        max_idx = max(0, s2_bands_batch.shape[1] - 1)  # Avoid negative index if empty
        idx_chosen = torch.clamp(idx_chosen, 0, max_idx)
        
        # Get data for chosen indices
        sub_bands = s2_bands_batch[b, idx_chosen]
        sub_doys = s2_doys_batch[b, idx_chosen].unsqueeze(-1)
        
        # Check if all bands are zero
        all_zero = torch.all(sub_bands == 0)
        
        if all_zero:
            # If all bands are zero, fill with small random values
            sub_bands = torch.randn_like(sub_bands) * 0.01
        
        # Standardize if required
        if standardize:
            sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)
        
        # Concatenate and store
        try:
            out_tensor[b] = torch.cat([sub_bands, sub_doys], dim=-1)
        except RuntimeError as e:
            # Handle any concatenation errors
            logging.warning(f"Error in concat: {e}, using random values")
            out_tensor[b] = torch.randn_like(out_tensor[b]) * 0.01
    
    # Ensure no all-zero rows that could cause MIOpen issues
    zero_rows = torch.all(out_tensor[:, :, :-1] == 0, dim=2)  # Check all columns except DOY
    if torch.any(zero_rows):
        for b in range(B):
            if torch.any(zero_rows[b]):
                # Replace all-zero rows with small random values
                zero_indices = torch.nonzero(zero_rows[b], as_tuple=True)[0]
                out_tensor[b, zero_indices, :-1] = torch.randn(len(zero_indices), out_tensor.shape[2]-1, device=device) * 0.01
    
    return out_tensor

def sample_s1_batch(s1_asc_bands_batch, s1_asc_doys_batch,
                    s1_desc_bands_batch, s1_desc_doys_batch,
                    band_mean, band_std, sample_size_s1, standardize=True):
    """
    Perform S1 random sampling for a batch of pixels (combining asc + desc).
    Vectorized implementation for better performance.
    Enhanced with additional safety checks for AMD GPUs.
    """
    B = s1_asc_bands_batch.shape[0]
    device = s1_asc_bands_batch.device
    
    # Create arrays to store results
    out_tensor = torch.zeros((B, sample_size_s1, s1_asc_bands_batch.shape[2] + 1), 
                            dtype=torch.float32, device=device)
    
    # Ensure minimum sequence length is at least 1
    min_seq_length = max(1, sample_size_s1)
    
    for b in range(B):
        # Safety check for empty tensors
        if (s1_asc_bands_batch.shape[0] == 0 or torch.all(s1_asc_bands_batch[b] == 0)) and \
           (s1_desc_bands_batch.shape[0] == 0 or torch.all(s1_desc_bands_batch[b] == 0)):
            # Create synthetic data if both are empty or all zeros
            s1_bands_all = torch.randn(min_seq_length, s1_asc_bands_batch.shape[2], device=device) * 0.01
            s1_doys_all = torch.randint(1, 366, (min_seq_length,), dtype=torch.int32, device=device)
        else:
            # Handle the case where one might be empty
            if s1_asc_bands_batch.shape[0] == 0 or torch.all(s1_asc_bands_batch[b] == 0):
                s1_bands_all = s1_desc_bands_batch[b]
                s1_doys_all = s1_desc_doys_batch[b]
            elif s1_desc_bands_batch.shape[0] == 0 or torch.all(s1_desc_bands_batch[b] == 0):
                s1_bands_all = s1_asc_bands_batch[b]
                s1_doys_all = s1_asc_doys_batch[b]
            else:
                # Concatenate ascending and descending data
                s1_bands_all = torch.cat([s1_asc_bands_batch[b], s1_desc_bands_batch[b]], dim=0)
                s1_doys_all = torch.cat([s1_asc_doys_batch[b], s1_desc_doys_batch[b]], dim=0)
        
        # Find valid indices (any non-zero band value)
        valid_mask = torch.any(s1_bands_all != 0, dim=-1)
        valid_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
        
        # Ensure we have at least some indices
        if len(valid_idx) == 0:
            # If no valid data, create synthetic indices
            valid_idx = torch.arange(min(s1_bands_all.shape[0], min_seq_length), device=device)
            
            # If still empty, create fake data
            if len(valid_idx) == 0:
                # Create fake data
                s1_bands_all = torch.randn(min_seq_length, s1_asc_bands_batch.shape[2], device=device) * 0.01
                s1_doys_all = torch.randint(1, 366, (min_seq_length,), dtype=torch.int32, device=device)
                valid_idx = torch.arange(min_seq_length, device=device)
        
        # Make sure valid_idx is not empty and has appropriate length
        if len(valid_idx) < min_seq_length:
            # Repeat indices to ensure we have enough
            repeat_factor = (min_seq_length + len(valid_idx) - 1) // max(1, len(valid_idx))
            valid_idx = valid_idx.repeat(repeat_factor)[:min_seq_length]
            idx_chosen = valid_idx
        else:
            # Randomly choose indices
            perm = torch.randperm(len(valid_idx), device=device)
            if len(perm) < min_seq_length:
                # Safety check if randperm fails
                perm = torch.arange(len(valid_idx), device=device)
                if len(perm) < min_seq_length:
                    # If still not enough, repeat
                    perm = perm.repeat((min_seq_length + len(perm) - 1) // max(1, len(perm)))[:min_seq_length]
            
            idx_chosen = valid_idx[perm[:min_seq_length]]
        
        # Sort indices for better memory access patterns
        idx_chosen, _ = torch.sort(idx_chosen)
        
        # Ensure idx_chosen is within valid range
        max_idx = max(0, s1_bands_all.shape[0] - 1)  # Avoid negative index
        idx_chosen = torch.clamp(idx_chosen, 0, max_idx)
        
        # Get data for chosen indices
        try:
            sub_bands = s1_bands_all[idx_chosen]
            sub_doys = s1_doys_all[idx_chosen].unsqueeze(-1)
            
            # Check if all bands are zero
            all_zero = torch.all(sub_bands == 0)
            
            if all_zero:
                # If all bands are zero, fill with small random values
                sub_bands = torch.randn_like(sub_bands) * 0.01
            
            # Standardize if required
            if standardize:
                sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)
            
            # Concatenate and store
            out_tensor[b] = torch.cat([sub_bands, sub_doys], dim=-1)
        except Exception as e:
            # Handle any indexing or concatenation errors
            logging.warning(f"Error in S1 sampling: {e}, using random values")
            out_tensor[b, :, :-1] = torch.randn(min_seq_length, s1_asc_bands_batch.shape[2], device=device) * 0.01
            out_tensor[b, :, -1] = torch.randint(1, 366, (min_seq_length,), dtype=torch.float32, device=device)
    
    # Final check for all-zero features
    zero_rows = torch.all(out_tensor[:, :, :-1] == 0, dim=2)  # Check all columns except DOY
    if torch.any(zero_rows):
        for b in range(B):
            if torch.any(zero_rows[b]):
                # Replace all-zero rows with small random values
                zero_indices = torch.nonzero(zero_rows[b], as_tuple=True)[0]
                out_tensor[b, zero_indices, :-1] = torch.randn(len(zero_indices), out_tensor.shape[2]-1, device=device) * 0.01
    
    return out_tensor

def prefetch_to_gpu(batch_data, device):
    """Prefetch batch data to the specified GPU device."""
    gpu_batch = {}
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            gpu_batch[key] = value.to(device, non_blocking=True)
        else:
            gpu_batch[key] = value
    return gpu_batch

def main():
    start_total_time = time.time()
    
    # Initialize distributed environment
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=120))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Debug info for MIOpen issues
    if local_rank == 0:
        logging.info(f"PYTORCH_SDP_DISABLE_FLASH_ATTENTION: {os.environ.get('PYTORCH_SDP_DISABLE_FLASH_ATTENTION', 'Not Set')}")
        logging.info(f"PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION: {os.environ.get('PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION', 'Not Set')}")

    # Configure logging
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse arguments and load config
    args = parse_args()
    config_module = load_config_module(args.config)
    config = config_module.config
    
    # Create cache directory if needed
    cache_dir = args.cache_dir
    if cache_dir and not os.path.exists(cache_dir):
        if local_rank == 0:
            os.makedirs(cache_dir, exist_ok=True)
    
    # Print config (only on rank 0)
    if local_rank == 0:
        logging.info(f"Running on {world_size} GPU(s). LocalRank={local_rank}")
        logging.info("Configurations:")
        for k, v in config.items():
            logging.info(f"  {k}: {v}")

    # Enable CUDA profiling if requested
    if args.profile and local_rank == 0:
        logging.info("Profiling enabled - measuring start-to-end time segments")
    
    # Construct dataset with caching
    dataset_start_time = time.time()
    cache_file = None
    if cache_dir:
        dataset_name = os.path.basename(config["tile_path"])
        cache_file = os.path.join(cache_dir, f"{dataset_name}_valid_pixels.npy")
    
    dataset = OptimizedTileInferenceDataset(
        tile_path=config["tile_path"],
        min_valid_timesteps=config["min_valid_timesteps"],
        standardize=False,  # Standardize during sampling
        cache_file=cache_file
    )
    
    if local_rank == 0:
        logging.info(f"Dataset initialization took {time.time() - dataset_start_time:.2f} seconds")
        logging.info(f"Dataset size: {len(dataset)} valid pixels")
        logging.info(f"Dataset shape: H={dataset.H}, W={dataset.W}")

    # Use DistributedSampler to partition the data
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
        drop_last=False
    )

    # Create dataloader with prefetch
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if config["num_workers"] > 0 else False
    )

    # Build and load SSL model
    model_start_time = time.time()
    ssl_model = build_ssl_model(config, device)

    if local_rank == 0:
        logging.info(f"Loading SSL checkpoint from {config['checkpoint_path']}")
    
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
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
    
    ssl_model.load_state_dict(new_state_dict, strict=True)
    
    # Freeze model parameters but keep one parameter with requires_grad=True to make DDP happy
    # We're not doing backprop so this doesn't affect runtime
    requires_grad_set = False
    
    for param in ssl_model.s2_backbone.parameters():
        if not requires_grad_set and args.use_ddp:
            param.requires_grad = True
            requires_grad_set = True
        else:
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
    
    # Enable DDP for more efficient inference if requested
    if args.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                    find_unused_parameters=False, static_graph=True)
    
    model.eval()
    
    if local_rank == 0:
        logging.info(f"Model loading and setup took {time.time() - model_start_time:.2f} seconds")
    
    # Remove automatic mixed precision as requested
    
    # Convert dataset stats to torch tensors on GPU
    s2_band_mean = torch.tensor(dataset.s2_band_mean, dtype=torch.float32, device=device)
    s2_band_std = torch.tensor(dataset.s2_band_std, dtype=torch.float32, device=device)
    s1_band_mean = torch.tensor(dataset.s1_band_mean, dtype=torch.float32, device=device)
    s1_band_std = torch.tensor(dataset.s1_band_std, dtype=torch.float32, device=device)
    
    # Create storage for results
    # We'll use sharded output files rather than collecting all on rank 0
    local_results = []
    
    # Starting inference
    inference_start_time = time.time()
    if local_rank == 0:
        logging.info(f"[Inference] Starting inference on {len(loader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            batch_start_time = time.time()
            
            # Record batch load time
            data_load_time = time.time() - batch_start_time
            
            # Move data to GPU (non-blocking)
            gpu_data_start = time.time()
            global_idxs = batch_data["global_idx"].to(device, non_blocking=True)  # shape=(B,)
            
            # Handle data that could be either tensor or numpy array
            def to_device(data):
                if isinstance(data, torch.Tensor):
                    return data.to(device, non_blocking=True)
                else:
                    return torch.from_numpy(data).to(device, non_blocking=True)
            
            # Move all data to GPU
            s2_bands_batch = to_device(batch_data["s2_bands"])
            s2_masks_batch = to_device(batch_data["s2_masks"])
            s2_doys_batch = to_device(batch_data["s2_doys"])
            
            s1_asc_bands_batch = to_device(batch_data["s1_asc_bands"])
            s1_asc_doys_batch = to_device(batch_data["s1_asc_doys"])
            s1_desc_bands_batch = to_device(batch_data["s1_desc_bands"])
            s1_desc_doys_batch = to_device(batch_data["s1_desc_doys"])
            
            gpu_transfer_time = time.time() - gpu_data_start
            
            B = s2_bands_batch.shape[0]
            sum_repr = None
            
            process_start_time = time.time()
            
            # Add try/except for error handling
            try:
                for r in range(config["repeat_times"]):
                    # Sample S2 data - directly on GPU
                    s2_input = sample_s2_batch(
                        s2_bands_batch, s2_masks_batch, s2_doys_batch,
                        band_mean=s2_band_mean,
                        band_std=s2_band_std,
                        sample_size_s2=config["sample_size_s2"],
                        standardize=True
                    )
                    
                    # Ensure valid dimensions for GRU processing with enhanced function
                    s2_input = ensure_valid_input_for_gru(s2_input, min_size=4)
                    
                    # Validate tensor for MIOpen compatibility
                    s2_input = validate_tensor_for_miopen(s2_input, "s2_input")

                    # Sample S1 data - directly on GPU
                    s1_input = sample_s1_batch(
                        s1_asc_bands_batch, s1_asc_doys_batch,
                        s1_desc_bands_batch, s1_desc_doys_batch,
                        band_mean=s1_band_mean,
                        band_std=s1_band_std,
                        sample_size_s1=config["sample_size_s1"],
                        standardize=True
                    )
                    
                    # Ensure valid dimensions for GRU processing with enhanced function
                    s1_input = ensure_valid_input_for_gru(s1_input, min_size=4)
                    
                    # Validate tensor for MIOpen compatibility
                    s1_input = validate_tensor_for_miopen(s1_input, "s1_input")

                    # Extra logging for debugging when issues occur
                    if batch_idx % 20 == 0 and local_rank == 0 and r == 0:
                        logging.info(f"S2 input shape: {s2_input.shape}, "
                                    f"min: {s2_input.min().item():.4f}, "
                                    f"max: {s2_input.max().item():.4f}, "
                                    f"has_nan: {torch.isnan(s2_input).any().item()}, "
                                    f"has_inf: {torch.isinf(s2_input).any().item()}")
                        logging.info(f"S1 input shape: {s1_input.shape}, "
                                    f"min: {s1_input.min().item():.4f}, "
                                    f"max: {s1_input.max().item():.4f}, "
                                    f"has_nan: {torch.isnan(s1_input).any().item()}, "
                                    f"has_inf: {torch.isinf(s1_input).any().item()}")

                    # Forward pass
                    z = model(s2_input, s1_input)

                    if sum_repr is None:
                        sum_repr = z
                    else:
                        sum_repr += z

                avg_repr = sum_repr / float(config["repeat_times"])
                
                # Store local results
                process_time = time.time() - process_start_time
                
                # Move results back to CPU and store
                store_start_time = time.time()
                avg_repr_np = avg_repr.cpu().numpy()
                global_idxs_np = global_idxs.cpu().numpy()
                
                for b in range(B):
                    gidx = global_idxs_np[b]
                    local_results.append((gidx, avg_repr_np[b]))
                
                store_time = time.time() - store_start_time
                
            except Exception as e:
                # Handle any errors during processing
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                # Create dummy representation
                dummy_repr = torch.zeros((B, config["latent_dim"] if "latent_dim" in config else 128), 
                                        dtype=torch.float32, device=device)
                
                # Move results back to CPU and store
                store_start_time = time.time()
                dummy_repr_np = dummy_repr.cpu().numpy()
                global_idxs_np = global_idxs.cpu().numpy()
                
                for b in range(B):
                    gidx = global_idxs_np[b]
                    local_results.append((gidx, dummy_repr_np[b]))
                    
                process_time = time.time() - process_start_time
                store_time = time.time() - store_start_time
            
            # Report progress
            if batch_idx % 5 == 0:
                batch_total_time = time.time() - batch_start_time
                if local_rank == 0:
                    logging.info(f"[Inference] batch {batch_idx}/{len(loader)} | "
                                 f"size={B} | "
                                 f"total_time={batch_total_time:.3f}s | "
                                 f"data_load={data_load_time:.3f}s | "
                                 f"gpu_transfer={gpu_transfer_time:.3f}s | "
                                 f"processing={process_time:.3f}s | "
                                 f"store={store_time:.3f}s")
    
    inference_time = time.time() - inference_start_time
    if local_rank == 0:
        logging.info(f"Inference completed in {inference_time:.2f} seconds")
        logging.info(f"Processed {len(local_results)} samples on rank {local_rank}")
    
    # Save results using sharded files for better efficiency
    save_start_time = time.time()
    
    # Convert local results to numpy arrays
    local_gidx = np.array([item[0] for item in local_results], dtype=np.int64)
    local_vecs = np.array([item[1] for item in local_results], dtype=np.float32)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(config["output_npy"])
    if local_rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save shard
    shard_output_path = f"{os.path.splitext(config['output_npy'])[0]}_shard_{global_rank}.npz"
    np.savez(shard_output_path, indices=local_gidx, vectors=local_vecs)
    
    # Make sure all processes have saved their shards
    dist.barrier()
    
    # On rank 0, combine the shards
    if local_rank == 0:
        logging.info("Combining shards...")
        H, W = dataset.H, dataset.W
        latent_dim = local_vecs.shape[1] if len(local_vecs) > 0 else config["latent_dim"]
        
        # Create the final output array
        out_array = np.full((H * W, latent_dim), 0, dtype=np.float32)
        
        # Load and combine all shards
        for rank in range(world_size):
            shard_path = f"{os.path.splitext(config['output_npy'])[0]}_shard_{rank}.npz"
            if os.path.exists(shard_path):
                shard_data = np.load(shard_path)
                shard_indices = shard_data['indices']
                shard_vectors = shard_data['vectors']
                
                # Add to the combined array
                out_array[shard_indices] = shard_vectors
                
                # Clean up shard file if desired
                if config.get("cleanup_shards", True):
                    os.remove(shard_path)
        
        # Reshape and save the final result
        out_array = out_array.reshape(H, W, latent_dim)
        np.save(config["output_npy"], out_array)
        logging.info(f"Saved final representation to {config['output_npy']}, shape={out_array.shape}")
    
    save_time = time.time() - save_start_time
    if local_rank == 0:
        logging.info(f"Saving results took {save_time:.2f} seconds")
    
    # Report total time
    total_time = time.time() - start_total_time
    if local_rank == 0:
        logging.info(f"Total execution time: {total_time:.2f} seconds")
    
    # Clean up
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()