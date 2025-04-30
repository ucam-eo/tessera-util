#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Run command:
# torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 src/train_multi_gpu_FSDP_64_fixed.py

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NUMEXPR_MAX_THREADS"] = "24"
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time
import math
import gc
import subprocess
import argparse
import logging
import json
from datetime import datetime
from contextlib import nullcontext

import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import wandb
import matplotlib.pyplot as plt

# Set matrix multiplication precision if needed
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Avoid torch.compile errors

# FSDP related imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.api import FullOptimStateDictConfig, ShardedOptimStateDictConfig

# Import standard dataset for validation
from datasets.ssl_dataset import AustrianCropValidation_64_Fixed
from models.modules import TransformerEncoder_64_Fixed, ProjectionHead
from models.ssl_model import MultimodalBTModel_64_Fixed, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate
from utils.metrics import linear_probe_evaluate_64_fixed, rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr

import torch.nn.attention as attn


# FSDP helper function: Get GPU memory usage
def get_gpu_memory_usage():
    """Return current GPU memory usage (in MB)"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

# FSDP helper function: Print model information

def print_model_info(model, name="Model"):
    """Print model structure and parameter information"""
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model structure tree (max 20 levels of recursion)
    def get_model_structure(model, depth=0, max_depth=20):
        if depth > max_depth:
            return "    " * depth + "...\n"
            
        result = ""
        for name, child in model.named_children():
            result += "    " * depth + f"({name}): {child.__class__.__name__}\n"
            result += get_model_structure(child, depth + 1, max_depth)
        return result
    
    model_structure = get_model_structure(model)
    
    return (
        f"\n{name} Information:\n"
        f"Model Class: {model.__class__.__name__}\n"
        f"Total Parameters: {total_params:,}\n"
        f"Trainable Parameters: {trainable_params:,}\n"
        f"Model Structure:\n{model_structure}"
    )

# Single group learning rate adjustment function
def adjust_single_group_lr(iter_count, total_iters, learning_rate, warmup_ratio=0.1, plateau_ratio=0.7):
    """Adjust learning rate for a single parameter group"""
    warmup_iters = int(total_iters * warmup_ratio)
    plateau_iters = int(total_iters * plateau_ratio)
    
    if iter_count < warmup_iters:
        # Warmup phase, linearly increase learning rate
        return learning_rate * (iter_count / warmup_iters)
    elif iter_count < plateau_iters:
        # Plateau phase
        return learning_rate
    else:
        # Cosine decay phase
        decay_ratio = (iter_count - plateau_iters) / (total_iters - plateau_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return learning_rate * cosine_decay

##########################################################################
# ChunkDataset class for FSDP-compatible data loading
##########################################################################
class ChunkDataset(Dataset):
    """
    Dataset class compatible with FSDP that loads data from s1 and s2 directories
    and applies augmentations on-the-fly.
    """
    def __init__(self, 
                 data_root,
                 file_names,
                 min_valid_timesteps=0,
                 sample_size_s2=64,
                 sample_size_s1=64,
                 shuffle_tiles=False):
        super().__init__()
        self.data_root = data_root
        self.min_valid_timesteps = min_valid_timesteps
        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1
        self.shuffle_tiles = shuffle_tiles

        # Build data paths
        self.s2_dir = os.path.join(data_root, "s2")
        self.s1_dir = os.path.join(data_root, "s1")
        
        for d in [self.s2_dir, self.s1_dir]:
            if not os.path.exists(d):
                raise RuntimeError(f"Directory {d} not found!")
        
        # Load all data into memory
        self.all_samples = []
        for fn in file_names:
            s2_path = os.path.join(self.s2_dir, fn)
            s1_path = os.path.join(self.s1_dir, fn)
            
            if os.path.exists(s1_path) and os.path.exists(s2_path):
                s2_array = np.load(s2_path)
                s1_array = np.load(s1_path)
                
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
                    
                    self.all_samples.append({
                        "s2_sample": s2_sample,
                        "s1_sample": s1_sample,
                        "s2_valid_mask": s2_valid_mask,
                        "s1_valid_mask": s1_valid_mask
                    })
        
        if len(self.all_samples) == 0:
            raise RuntimeError("No valid samples found in dataset!")
        
        if shuffle_tiles:
            np.random.shuffle(self.all_samples)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        sample_data = self.all_samples[index]
        s2_sample = sample_data["s2_sample"]
        s1_sample = sample_data["s1_sample"]
        s2_valid_mask = sample_data["s2_valid_mask"]
        s1_valid_mask = sample_data["s1_valid_mask"]
        
        # Apply augmentations
        s2_aug1, s2_aug2 = self._augment_pair(s2_sample, s2_valid_mask, is_s2=True)
        s1_aug1, s1_aug2 = self._augment_pair(s1_sample, s1_valid_mask, is_s2=False)
        
        return {
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

def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training with 64 timesteps (FSDP)")
    parser.add_argument('--config', type=str, default="configs/ssl_config_64_fixed.py", help="Path to config file")
    return parser.parse_args()

def main():
    # Set random seed
    torch.manual_seed(3407)
    np.random.seed(3407)
    
    ##########################################################################
    # Initialize distributed environment (PyTorch's torchrun + FSDP)
    ##########################################################################
    dist.init_process_group(backend='nccl')
    
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Local process index from torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set current device
    torch.cuda.set_device(local_rank)
    
    # Set log level: rank=0 outputs info logs, other ranks only warnings
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Disable W&B for non-zero ranks
        os.environ['WANDB_MODE'] = 'disabled'

    args_cli = parse_args()
    # Load configuration
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    # Check if AMP should be applied
    apply_amp = config.get('apply_amp', False)
    if local_rank == 0:
        logging.info(f"apply_amp = {apply_amp}")

    # Use CUDA device
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        logging.info(f"Running on {world_size} GPU(s). LocalRank={local_rank}, device={device}")
        logging.info(f"Initial GPU memory usage: {get_gpu_memory_usage():.2f} MB")
    
    # Set W&B API key if provided in config
    if local_rank == 0 and 'wandb_api_key' in config:
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']

    # Disable W&B git info if requested
    if config.get("disable_wandb_git", False):
        os.environ['WANDB_DISABLE_GIT'] = 'true'

    # Read chunk_batch size
    chunk_batch = config.get('chunk_batch', 25)
    if local_rank == 0:
        logging.info(f"chunk_batch = {chunk_batch}")

    # Initialize W&B
    run_name = f"FSDP_BT_64_Fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if local_rank == 0:
        wandb_run = wandb.init(project="btfm-64-fixed", name=run_name, config=config)
        
        # Save source code
        artifact = wandb.Artifact('source-code', type='code')
        artifact.add_file('src/train_multi_gpu_FSDP_64_fixed.py')
        artifact.add_file('src/datasets/ssl_dataset.py')
        artifact.add_file('src/models/modules.py')
        artifact.add_file('src/models/ssl_model.py')
        artifact.add_file('src/utils/lr_scheduler.py')
        artifact.add_file('src/utils/metrics.py')
        artifact.add_file('src/utils/misc.py')
        artifact.add_file('configs/ssl_config_64_fixed.py')
        wandb.log_artifact(artifact)
    else:
        wandb_run = None

    # Calculate total training steps
    total_steps = config['epochs'] * config['total_samples'] // config['batch_size'] // world_size
    if local_rank == 0:
        logging.info(f"Total steps per rank (approx) = {total_steps}")

    # Build model components
    s2_num_heads = config.get('s2_num_heads', 8)
    s2_num_layers = config.get('s2_num_layers', 8)
    s2_dim_feedforward = config.get('s2_dim_feedforward', 1024)
    s1_num_heads = config.get('s1_num_heads', 8)
    s1_num_layers = config.get('s1_num_layers', 8)
    s1_dim_feedforward = config.get('s1_dim_feedforward', 1024)
    
    # Sync to wandb config if needed
    if local_rank == 0:
        wandb.config.update({
            "s2_num_heads": s2_num_heads,
            "s2_num_layers": s2_num_layers,
            "s2_dim_feedforward": s2_dim_feedforward,
            "s1_num_heads": s1_num_heads,
            "s1_num_layers": s1_num_layers,
            "s1_dim_feedforward": s1_dim_feedforward
        })
    
    # Create model components
    s2_enc = TransformerEncoder_64_Fixed(
        band_num=10,
        latent_dim=config['latent_dim'],
        nhead=s2_num_heads,
        num_encoder_layers=s2_num_layers,
        dim_feedforward=s2_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s2']
    ).to(device)
    
    s1_enc = TransformerEncoder_64_Fixed(
        band_num=2,
        latent_dim=config['latent_dim'],
        nhead=s1_num_heads,
        num_encoder_layers=s1_num_layers,
        dim_feedforward=s1_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s1']
    ).to(device)
    
    # Choose projector input dimension based on fusion method
    if config['fusion_method'] == 'concat':
        proj_in_dim = config['latent_dim']
    else:
        proj_in_dim = config['latent_dim']
        
    projector = ProjectionHead(proj_in_dim, config['projector_hidden_dim'], config['projector_out_dim']).to(device)
    
    # Create the full model
    model = MultimodalBTModel_64_Fixed(
        s2_enc, s1_enc, projector, 
        fusion_method=config['fusion_method'], 
        return_repr=True, 
        latent_dim=config['latent_dim']
    ).to(device)
    
    # Print FSDP pre-wrapping model info
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Before FSDP wrapping - Total model parameters: {total_params:,}")
        logging.info(f"GPU memory usage before FSDP: {get_gpu_memory_usage():.2f} MB")
    
    # Set loss function
    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])
    
    # Define modules to be wrapped by FSDP
    module_classes_to_wrap = [TransformerEncoder_64_Fixed]
    
    # Define FSDP mixed precision config (if using AMP)
    if apply_amp:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            # Keep main computation in float16, but accumulate in float32 to avoid numerical issues
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float16,
        )
    else:
        mixed_precision_policy = None
    
    # Create custom FSDP wrap policy
    def custom_auto_wrap_policy(module, recurse=True, **kwargs):
        """Custom FSDP wrapping policy to wrap specified module types"""
        if isinstance(module, tuple(module_classes_to_wrap)):
            return True
        return False
    
    # Define FSDP configuration parameters
    fsdp_config = {
        "auto_wrap_policy": custom_auto_wrap_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": local_rank,
        "sync_module_states": True,
        "forward_prefetch": True,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "cpu_offload": CPUOffload(offload_params=False),
        "use_orig_params": True,  # Required for torch.compile compatibility
    }
    
    # Add mixed precision config if enabled
    if mixed_precision_policy:
        fsdp_config["mixed_precision"] = mixed_precision_policy
    
    # Print FSDP configuration
    if local_rank == 0:
        logging.info("FSDP Configuration:")
        config_str = json.dumps({k: str(v) for k, v in fsdp_config.items()}, indent=2)
        logging.info(config_str)
    
    # Wrap model with FSDP
    with (attn.sdpa_kernel(attn.SDPBackend.MATH) if config.get('use_torch_compile', False) else nullcontext()):
        # Wrap model with FSDP
        model = FSDP(model, **fsdp_config)
        
        # Print wrapped model structure (only on rank=0)
        if local_rank == 0:
            logging.info(f"GPU memory usage after FSDP: {get_gpu_memory_usage():.2f} MB")
            
            # Get parameter count on each rank
            fsdp_params = sum(p.numel() for p in model.parameters())
            logging.info(f"After FSDP wrapping - Parameters on rank {local_rank}: {fsdp_params:,}")
            
            # Calculate sharding ratio
            sharded_ratio = fsdp_params / total_params
            logging.info(f"Sharding ratio: {sharded_ratio:.4f} (lower means better distribution)")
            
            # Calculate ideal sharding ratio based on world_size
            ideal_ratio = 1.0 / world_size
            logging.info(f"Ideal sharding ratio: {ideal_ratio:.4f}")
        
        # Use torch.compile if configured
        if config.get('use_torch_compile', False):
            # model = torch.compile(model, mode="max-autotune", dynamic=True)
            model = torch.compile(model, mode="default", dynamic=True)
                
            if local_rank == 0:
                logging.info("Using torch.compile to optimize the model...")
        
        # Collect parameter counts across all GPUs (to verify even distribution)
        local_param_count = sum(p.numel() for p in model.parameters())
        all_ranks_param_counts = [torch.tensor([0], device=device) for _ in range(world_size)]
        local_count_tensor = torch.tensor([local_param_count], device=device)
        
        # Gather parameter counts from all ranks
        dist.all_gather(all_ranks_param_counts, local_count_tensor)
        
        if local_rank == 0:
            rank_counts = [int(t.item()) for t in all_ranks_param_counts]
            logging.info(f"Parameter counts across all ranks: {rank_counts}")
            
            # Check for distribution differences (<0.1% is acceptable)
            max_count = max(rank_counts)
            min_count = min(rank_counts)
            diff_percentage = (max_count - min_count) / max_count * 100
            
            if diff_percentage < 0.1:
                logging.info(f"✅ Parameters are evenly distributed across GPUs (difference < 0.1%: {diff_percentage:.4f}%)")
            else:
                logging.info(f"⚠️ Parameters are NOT evenly distributed across GPUs (difference: {diff_percentage:.4f}%)")
        
        # Create optimizer with parameter groups
        all_params = list(model.parameters())
        weight_params = []
        bias_params = []
        
        # Group parameters by shape instead of name
        for param in all_params:
            if param.requires_grad:
                if len(param.shape) > 1:
                    weight_params.append(param)
                else:
                    bias_params.append(param)
        
        if local_rank == 0:
            logging.info(f"Number of weight parameters: {len(weight_params)}")
            logging.info(f"Number of bias parameters: {len(bias_params)}")
            
            # If no weight parameters, use all parameters in one group
            if len(weight_params) == 0:
                logging.warning("No weight parameters found! Using all parameters in one group.")
                weight_params = all_params
                bias_params = []
        
        # Create optimizer (one or two parameter groups)
        if len(weight_params) == 0:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'], 
                weight_decay=1e-6
            )
        else:
            optimizer = torch.optim.AdamW(
                [{'params': weight_params}, {'params': bias_params}],
                lr=config['learning_rate'], 
                weight_decay=1e-6
            )
        
        # Setup AMP if configured
        scaler = torch.cuda.amp.GradScaler() if apply_amp else None
        
        if local_rank == 0 and wandb_run is not None:
            # Note: FSDP may limit what wandb.watch can record
            wandb.watch(model, log="gradients", log_freq=400)
        
        ########################################################################
        # Initialize global variables for training
        ########################################################################
        step = 0
        examples = 0
        last_time = time.time()
        last_examples = 0
        rolling_loss = []
        best_val_acc = 0.0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_fsdp_{timestamp}.pt")
        
        # Ensure checkpoints directory exists
        if local_rank == 0:
            os.makedirs(os.path.join("checkpoints", "ssl"), exist_ok=True)
        
        ########################################################################
        # Start training epochs
        ########################################################################
        for epoch in range(config['epochs']):
            # If data needs to be generated via rust script, only do it on rank 0
            # if local_rank == 0 and 'rust_cmd' in config:
            #     logging.info(f"Epoch {epoch} started. Generating training data via rust_cmd...")
            #     subprocess.run(config['rust_cmd'], shell=True, check=True)
            #     logging.info(f"Data generation finished for epoch {epoch}.")
            
            # Wait for rank 0 to finish data generation
            dist.barrier()
            
            # Get list of data files
            s2_dir = os.path.join(config['data_root'], 's2')
            file_names = sorted(os.listdir(s2_dir))
            np.random.shuffle(file_names)
            
            # Split files into chunks for memory-efficient processing
            total_files = len(file_names)
            if local_rank == 0:
                logging.info(f"Epoch {epoch}: total files = {total_files}")
            
            # Process data in chunks to avoid OOM
            for chunk_start in range(0, total_files, chunk_batch):
                chunk_end = min(chunk_start + chunk_batch, total_files)
                chunk_files = file_names[chunk_start:chunk_end]
                
                if local_rank == 0:
                    logging.info(f"Epoch {epoch}, processing chunk {chunk_start}:{chunk_end} ({len(chunk_files)} files)")
                
                # Create dataset for this chunk
                chunk_dataset = ChunkDataset(
                    data_root=config['data_root'],
                    file_names=chunk_files,
                    min_valid_timesteps=config['min_valid_timesteps'],
                    sample_size_s2=config['sample_size_s2'],
                    sample_size_s1=config['sample_size_s1'],
                    shuffle_tiles=config.get('shuffle_tiles', False)
                )
                
                # Create distributed sampler
                train_sampler = DistributedSampler(
                    chunk_dataset,
                    num_replicas=world_size,
                    rank=global_rank,
                    shuffle=True,
                    drop_last=True
                )
                
                # Create dataloader
                train_loader = DataLoader(
                    chunk_dataset,
                    batch_size=config['batch_size'],
                    sampler=train_sampler,
                    num_workers=config['num_workers'],
                    pin_memory=True,
                    drop_last=True
                )
                
                # Set epoch for the sampler
                train_sampler.set_epoch(epoch)
                
                if local_rank == 0:
                    logging.info(f"Chunk dataset has {len(chunk_dataset)} samples, {len(train_loader)} batches")
                
                # Train on this chunk
                model.train()
                
                for batch_idx, batch_data in enumerate(train_loader):
                    s2_aug1 = batch_data['s2_aug1'].to(device, non_blocking=True)
                    s2_aug2 = batch_data['s2_aug2'].to(device, non_blocking=True)
                    s1_aug1 = batch_data['s1_aug1'].to(device, non_blocking=True)
                    s1_aug2 = batch_data['s1_aug2'].to(device, non_blocking=True)
                    s2_mask = batch_data['s2_mask'].to(device, non_blocking=True)
                    s1_mask = batch_data['s1_mask'].to(device, non_blocking=True)
                    
                    # Learning rate scheduling
                    if len(optimizer.param_groups) > 1:
                        # Original scheduler for two parameter groups
                        adjust_learning_rate(
                            optimizer,
                            step,
                            total_steps,
                            config['learning_rate'],
                            config['warmup_ratio'],
                            config['plateau_ratio']
                        )
                    else:
                        # Single parameter group scheduling
                        current_lr = adjust_single_group_lr(
                            step, 
                            total_steps,
                            config['learning_rate'],
                            config['warmup_ratio'],
                            config['plateau_ratio']
                        )
                        optimizer.param_groups[0]['lr'] = current_lr
                    
                    optimizer.zero_grad()
                    
                    # Forward pass with AMP if enabled
                    if apply_amp:
                        with torch.cuda.amp.autocast():
                            z1, repr1 = model(s2_aug1, s1_aug1, s2_mask, s1_mask)
                            z2, repr2 = model(s2_aug2, s1_aug2, s2_mask, s1_mask)
                            loss_main, bar_main, off_main = criterion(z1, z2)
                            
                            # Apply mixup if configured
                            loss_mix = 0.0
                            if config.get('apply_mixup', False):
                                B = s2_aug1.size(0)
                                idxs = torch.randperm(B, device=device)
                                alpha = torch.distributions.Beta(
                                    config['beta_alpha'],
                                    config['beta_beta']
                                ).sample().to(device)
                                y_m_s2 = alpha * s2_aug1 + (1 - alpha) * s2_aug2[idxs, :]
                                y_m_s1 = alpha * s1_aug1 + (1 - alpha) * s1_aug2[idxs, :]
                                
                                z_m, _ = model(y_m_s2, y_m_s1, s2_mask, s1_mask)
                                
                                # Use gather to replace advanced indexing z2[idxs]
                                z2_perm = torch.gather(z2, 0, idxs.unsqueeze(1).expand(-1, z2.size(1)))
                                
                                cc_m_a = compute_cross_correlation(z_m, z1)
                                cc_m_b = compute_cross_correlation(z_m, z2)
                                
                                cc_z1_z1 = compute_cross_correlation(z1, z1)
                                cc_z2idx_z1 = compute_cross_correlation(z2_perm, z1)
                                cc_z1_z2 = compute_cross_correlation(z1, z2)
                                cc_z2idx_z2 = compute_cross_correlation(z2_perm, z2)
                                
                                cc_m_a_gt = alpha * cc_z1_z1 + (1 - alpha) * cc_z2idx_z1
                                cc_m_b_gt = alpha * cc_z1_z2 + (1 - alpha) * cc_z2idx_z2
                                
                                diff_a = (cc_m_a - cc_m_a_gt).pow(2).sum()
                                diff_b = (cc_m_b - cc_m_b_gt).pow(2).sum()
                                loss_mix = config['mixup_lambda'] * config['barlow_lambda'] * (diff_a + diff_b)
                            
                            total_loss = loss_main + loss_mix
                        
                        # Backward pass with AMP
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard forward pass without AMP
                        z1, repr1 = model(s2_aug1, s1_aug1, s2_mask, s1_mask)
                        z2, repr2 = model(s2_aug2, s1_aug2, s2_mask, s1_mask)
                        loss_main, bar_main, off_main = criterion(z1, z2)
                        
                        # Apply mixup if configured
                        loss_mix = 0.0
                        if config.get('apply_mixup', False):
                            B = s2_aug1.size(0)
                            idxs = torch.randperm(B, device=device)
                            alpha = torch.distributions.Beta(
                                config['beta_alpha'],
                                config['beta_beta']
                            ).sample().to(device)
                            y_m_s2 = alpha * s2_aug1 + (1 - alpha) * s2_aug2[idxs, :]
                            y_m_s1 = alpha * s1_aug1 + (1 - alpha) * s1_aug2[idxs, :]
                            
                            z_m, _ = model(y_m_s2, y_m_s1, s2_mask, s1_mask)
                            
                            # Use gather to replace advanced indexing z2[idxs]
                            z2_perm = torch.gather(z2, 0, idxs.unsqueeze(1).expand(-1, z2.size(1)))
                            
                            cc_m_a = compute_cross_correlation(z_m, z1)
                            cc_m_b = compute_cross_correlation(z_m, z2)
                            
                            cc_z1_z1 = compute_cross_correlation(z1, z1)
                            cc_z2idx_z1 = compute_cross_correlation(z2_perm, z1)
                            cc_z1_z2 = compute_cross_correlation(z1, z2)
                            cc_z2idx_z2 = compute_cross_correlation(z2_perm, z2)
                            
                            cc_m_a_gt = alpha * cc_z1_z1 + (1 - alpha) * cc_z2idx_z1
                            cc_m_b_gt = alpha * cc_z1_z2 + (1 - alpha) * cc_z2idx_z2
                            
                            diff_a = (cc_m_a - cc_m_a_gt).pow(2).sum()
                            diff_b = (cc_m_b - cc_m_b_gt).pow(2).sum()
                            loss_mix = config['mixup_lambda'] * config['barlow_lambda'] * (diff_a + diff_b)
                        
                        total_loss = loss_main + loss_mix
                        total_loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                        optimizer.step()
                    
                    batch_sz = s2_aug1.size(0)
                    examples += batch_sz
                    
                    # Logging
                    if (step % config['log_interval_steps'] == 0) and (local_rank == 0):
                        current_time = time.time()
                        exps = (examples - last_examples) / (current_time - last_time)
                        last_time = current_time
                        last_examples = examples
                        
                        rolling_loss.append(loss_main.item())
                        if len(rolling_loss) > 40:
                            rolling_loss = rolling_loss[-40:]
                        avg_loss = sum(rolling_loss) / len(rolling_loss)
                        
                        current_lr = optimizer.param_groups[0]['lr']
                        try:
                            erank_z = rankme(z1)
                            erank_repr = rankme(repr1)
                        except:
                            erank_z = 0.0
                            erank_repr = 0.0
                            logging.warning("Rank computation failed.")
                        
                        # Record current GPU memory usage
                        current_gpu_mem = get_gpu_memory_usage()
                        
                        logging.info(
                            f"[Epoch={epoch}, Step={step}] "
                            f"Loss={loss_main.item():.2f}, MixLoss={loss_mix:.2f}, AvgLoss={avg_loss:.2f}, "
                            f"LR={current_lr:.4f}, batchsize={batch_sz}, Examples/sec={exps:.2f}, "
                            f"Rank(z)={erank_z:.4f}, Rank(repr)={erank_repr:.4f}, "
                            f"GPU_Mem={current_gpu_mem:.2f}MB"
                        )
                        
                        wandb_dict = {
                            "epoch": epoch,
                            "loss_main": loss_main.item(),
                            "mix_loss": loss_mix,
                            "avg_loss": avg_loss,
                            "lr": current_lr,
                            "examples/sec": exps,
                            "total_loss": total_loss.item(),
                            "rank_z": erank_z,
                            "rank_repr": erank_repr,
                            # "gpu_memory_usage_mb": current_gpu_mem,
                            # "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                        }
                        
                        # Add training progress visualization
                        progress_percentage = (step / total_steps) * 100
                        progress_bar_html = f"""
                        <div style="border: 1px solid #ccc; width: 100%; height: 20px; border-radius: 3px; overflow: hidden;">
                        <div style="background-color: #4CAF50; width: {progress_percentage}%; height: 100%; text-align: center; line-height: 20px; color: white;">
                            {progress_percentage:.1f}%
                        </div>
                        </div>
                        """
                        wandb_dict["training_progress"] = wandb.Html(progress_bar_html)
                        
                        wandb.log(wandb_dict, step=step)
                    
                    # Validation
                    # if (config['val_interval_steps'] > 0) and (step > 0) and (step % config['val_interval_steps'] == 0) and (local_rank == 0):
                    if (config['val_interval_steps'] > 0) and (step % config['val_interval_steps'] == 0) and (local_rank == 0):
                        # If validation dataset paths are configured
                        if all(config.get(k) for k in [
                            'val_s2_bands_file_path',
                            'val_s2_masks_file_path',
                            'val_s2_doy_file_path',
                            'val_s1_asc_bands_file_path',
                            'val_s1_asc_doy_file_path',
                            'val_s1_desc_bands_file_path',
                            'val_s1_desc_doy_file_path',
                            'val_labels_path'
                        ]):
                            # Create validation dataset
                            val_dataset = AustrianCropValidation_64_Fixed(
                                s2_bands_file_path=config['val_s2_bands_file_path'],
                                s2_masks_file_path=config['val_s2_masks_file_path'],
                                s2_doy_file_path=config['val_s2_doy_file_path'],
                                s1_asc_bands_file_path=config['val_s1_asc_bands_file_path'],
                                s1_asc_doy_file_path=config['val_s1_asc_doy_file_path'],
                                s1_desc_bands_file_path=config['val_s1_desc_bands_file_path'],
                                s1_desc_doy_file_path=config['val_s1_desc_doy_file_path'],
                                labels_path=config['val_labels_path'],
                                field_id_path=config.get('field_id_path'),
                                sample_size_s2=config['sample_size_s2'],
                                sample_size_s1=config['sample_size_s1'],
                                min_valid_timesteps=0,
                                standardize=True
                            )
                            
                            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
                            model.eval()
                            
                            # Field IDs for validation if provided
                            field_ids = None
                            if 'field_id_path' in config and os.path.exists(config['field_id_path']):
                                field_ids = np.load(config['field_id_path'])
                            
                            # Evaluate with linear probe
                            val_acc, val_f1, val_cm, val_cr = linear_probe_evaluate_64_fixed(
                                model, 
                                val_loader, 
                                field_ids=field_ids,
                                field_data_path=config.get('fielddata_csv_path'),
                                training_ratio=config.get('training_ratio', 0.3),
                                val_test_split_ratio=config.get('val_test_split_ratio', 1/7.0),
                                classifier_type=config.get('classifier_type', 'lr'),
                                num_inference=config.get('num_inference', 1),
                                device=device
                            )
                            
                            wandb.log({
                                "val_acc": val_acc, 
                                "val_f1": val_f1,
                                "val_cr": wandb.Html(f"<pre>{val_cr}</pre>"),
                            }, step=step)
                            
                            logging.info(f"Validation at step {step}: val_acc={val_acc:.4f}, F1 Score={val_f1:.4f}")
                            logging.info(f"Confusion Matrix:\n{val_cm}")
                            logging.info(f"Classification Report:\n{val_cr}")
                            
                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(val_cm, interpolation='nearest', cmap=plt.cm.Blues)
                            ax.figure.colorbar(im, ax=ax)
                            ax.set(xticks=range(val_cm.shape[1]),
                                  yticks=range(val_cm.shape[0]),
                                  xticklabels=range(val_cm.shape[1]),
                                  yticklabels=range(val_cm.shape[0]),
                                  title='Confusion Matrix',
                                  ylabel='True label',
                                  xlabel='Predicted label')
                            thresh = val_cm.max() / 2.
                            for i in range(val_cm.shape[0]):
                                for j in range(val_cm.shape[1]):
                                    ax.text(j, i, format(val_cm[i, j], 'd'),
                                          ha="center", va="center",
                                          color="white" if val_cm[i, j] > thresh else "black")
                            fig.tight_layout()
                            wandb.log({"val_confusion_matrix": wandb.Image(fig)}, step=step)
                            plt.close(fig)
                            
                            # Save best model with FSDP
                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                
                                # Use FSDP state_dict_type to save model correctly
                                full_state_dict_config = FullStateDictConfig(
                                    offload_to_cpu=True,
                                    rank0_only=True
                                )
                                
                                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                                    state_dict = model.state_dict()
                                    if local_rank == 0:
                                        logging.info(f"Saving best model with val_acc={val_acc:.4f} to {best_ckpt_path}")
                                        
                                        # Create checkpoint directory if it doesn't exist
                                        os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                                        
                                        torch.save({
                                            'epoch': epoch,
                                            'step': step,
                                            'model_state_dict': state_dict,
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'best_val_acc': best_val_acc,
                                        }, best_ckpt_path)
                            
                            model.train()
                    
                    step += 1
                
                # Clean up after processing this chunk
                del train_loader
                del train_sampler
                del chunk_dataset
                gc.collect()
                torch.cuda.empty_cache()
            
            if local_rank == 0:
                logging.info(f"Epoch {epoch} finished, current step = {step}")
                # Record GPU memory at end of epoch
                logging.info(f"GPU memory at end of epoch {epoch}: {get_gpu_memory_usage():.2f} MB")
        
        if local_rank == 0:
            logging.info("Training completed.")
            if wandb_run is not None:
                wandb_run.finish()
        
        dist.destroy_process_group()

if __name__ == "__main__":
    main()