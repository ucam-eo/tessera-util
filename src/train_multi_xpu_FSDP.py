#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NUMEXPR_MAX_THREADS"] = "64"

import math
import time
import gc
import argparse
import logging
import subprocess
from datetime import datetime

import numpy as np
import torch
import torch.amp
import functools

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

# Intel GPU environment initialization
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch

# plotting / WANDB
import wandb
import matplotlib.pyplot as plt

# ======== Your existing modules (imports only!) ========
from models.modules import TransformerEncoder, ProjectionHead, SpectralTemporalTransformer
from models.ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate, LARS
from utils.metrics import linear_probe_evaluate, rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr
# =======================================================

##############################################################################
# 1) Dataset for chunk-based loading
##############################################################################
class ChunkDataset(torch.utils.data.Dataset):
    """
    Loads partial chunked .npy files for augmented data, merges, shuffles in-memory.
    Each item -> (s2_aug1, s2_aug2, s1_aug1, s1_aug2).
    """
    def __init__(self, aug1_s2_files, aug2_s2_files, aug1_s1_files, aug2_s1_files):
        super().__init__()
        self.all_samples = []

        for idx_file in range(len(aug1_s2_files)):
            arr_aug1_s2 = np.load(aug1_s2_files[idx_file])
            arr_aug2_s2 = np.load(aug2_s2_files[idx_file])
            arr_aug1_s1 = np.load(aug1_s1_files[idx_file])
            arr_aug2_s1 = np.load(aug2_s1_files[idx_file])

            n_samples = min(
                arr_aug1_s2.shape[0],
                arr_aug2_s2.shape[0],
                arr_aug1_s1.shape[0],
                arr_aug2_s1.shape[0]
            )
            for i in range(n_samples):
                self.all_samples.append((
                    arr_aug1_s2[i],
                    arr_aug2_s2[i],
                    arr_aug1_s1[i],
                    arr_aug2_s1[i]
                ))

        np.random.shuffle(self.all_samples)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        s2_aug1_np, s2_aug2_np, s1_aug1_np, s1_aug2_np = self.all_samples[idx]
        s2_aug1 = torch.tensor(s2_aug1_np, dtype=torch.float32)
        s2_aug2 = torch.tensor(s2_aug2_np, dtype=torch.float32)
        s1_aug1 = torch.tensor(s1_aug1_np, dtype=torch.float32)
        s1_aug2 = torch.tensor(s1_aug2_np, dtype=torch.float32)
        return {
            "s2_aug1": s2_aug1,
            "s2_aug2": s2_aug2,
            "s1_aug1": s1_aug1,
            "s1_aug2": s1_aug2
        }

##############################################################################
# 2) Parse config
##############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training (Intel GPU + chunk-based FSDP)")
    parser.add_argument('--config', type=str, default="configs/ssl_config.py",
                        help="Path to config file (e.g. configs/ssl_config.py)")
    return parser.parse_args()

##############################################################################
# 3) Diagnostics / Logging Helpers for FSDP
##############################################################################
def analyze_model_structure(model_instance, rank):
    """Analyze model structure *before* FSDP wrapping/sharding."""
    if rank != 0:
        return
    logging.info("=== Model structure analysis (pre-FSDP) ===")
    total_params = sum(p.numel() for p in model_instance.parameters())
    logging.info(f"Total param count: {total_params:,}")

    for name, module in model_instance.named_children():
        sub_params = sum(p.numel() for p in module.parameters())
        pct = sub_params / total_params * 100 if total_params > 0 else 0
        logging.info(f"  {name}: {sub_params:,} params ({pct:.2f}%)")
        
        # Log second-level modules for more detail
        for sub_name, sub_module in module.named_children():
            sub_sub_params = sum(p.numel() for p in sub_module.parameters())
            sub_pct = sub_sub_params / total_params * 100 if total_params > 0 else 0
            logging.info(f"    {name}.{sub_name}: {sub_sub_params:,} params ({sub_pct:.2f}%)")

def check_xpu_memory_usage(rank, world_size, device):
    """Gather memory usage from each rank and log."""
    try:
        stats = torch.xpu.memory_stats(device)
        allocated = stats.get("allocated_bytes", 0) / (1024 ** 2)
    except:
        try:
            allocated = torch.xpu.memory_allocated(device) / (1024 ** 2)
        except:
            allocated = -1

    # Synchronize before memory check
    torch.xpu.synchronize(device)
    dist.barrier()
    
    mem_tensor = torch.tensor([allocated], device=device)
    mem_list = [torch.zeros_like(mem_tensor) for _ in range(world_size)]
    dist.all_gather(mem_list, mem_tensor)

    if rank == 0:
        logging.info("=== XPU memory usage by rank (MB) ===")
        values = []
        for i, t in enumerate(mem_list):
            val = t.item()
            values.append(val)
            logging.info(f"Rank {i}: {val:.2f} MB")
        total = sum(values)
        logging.info(f"Total across ranks: {total:.2f} MB")
        if len(values) > 1 and min(values) > 0:
            imb = (max(values) - min(values)) / min(values) * 100
            logging.info(f"Memory imbalance: {imb:.2f}%")

def check_model_device_placement(model, rank):
    """Check device placement after FSDP wrapping."""
    if rank != 0:
        return
    logging.info("=== Checking device placement of parameters ===")
    device_map = {}
    for n, p in model.named_parameters():
        dev_str = str(p.device)
        device_map[dev_str] = device_map.get(dev_str, 0) + 1
    for d, cnt in device_map.items():
        logging.info(f"  Device {d}: {cnt} parameters")
    if len(device_map) == 1:
        logging.warning("⚠ All parameters appear on a single device -> no real sharding?")

def print_model_parameter_stats(model, rank, world_size, device):
    """Show how many parameters each rank physically holds after FSDP shards."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if rank == 0:
        logging.info("=== FSDP parameter stats ===")
        logging.info(f"Logical total: {total_params:,}")
        logging.info(f"Trainable: {trainable_params:,}")

    # Count only actual parameters stored on this rank (flat params)
    local_params = 0
    for module in model.modules():  # Fix: modules() returns an iterator, not name-value pairs
        if isinstance(module, FSDP):
            # Access flat param if available
            if hasattr(module, '_flat_param') and module._flat_param is not None:
                local_params += module._flat_param.numel()

    # If we couldn't find flat params, try this alternate approach
    if local_params == 0:
        for param in model.parameters():
            if hasattr(param, '_is_sharded') and param._is_sharded:
                local_params += param.numel()
    
    # Last resort fallback - just count all params (this would indicate sharding isn't working)
    if local_params == 0:
        for param in model.parameters():
            local_params += param.numel()

    # Force synchronization before gathering
    torch.xpu.synchronize(device)
    dist.barrier()
    
    local_tensor = torch.tensor([local_params], device=device)
    all_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors, local_tensor)

    if rank == 0:
        logging.info("=== Per-rank local param counts ===")
        vals = [t.item() for t in all_tensors]
        for i, v in enumerate(vals):
            pct = (v / total_params * 100) if total_params > 0 else 0
            logging.info(f"Rank {i}: {int(v):,} ({pct:.2f}%)")
        
        # Check sharding status
        unique_counts = len(set([int(v) for v in vals]))
        if unique_counts == 1 and vals[0] >= 0.95 * total_params:
            logging.warning("⚠ All ranks have nearly identical param counts - FSDP might not be sharding.")
            ideal = total_params // world_size
            logging.info(f"Ideal shard size: ~{ideal:,}")
        else:
            logging.info("✅ Parameters appear to be properly sharded across ranks.")

def diagnose_fsdp_implementation(rank):
    """Check environment + versions for potential issues."""
    if rank != 0:
        return
    logging.info("=== Diagnosing FSDP environment ===")
    logging.info(f"PyTorch version: {torch.__version__}")
    ipex_version = getattr(intel_extension_for_pytorch, '__version__', 'unknown')
    logging.info(f"IPEX version: {ipex_version}")
    env_vars = ['ZE_FLAT_DEVICE_HIERARCHY', 'ZE_AFFINITY_MASK', 'CCL_WORKER_COUNT', 
                'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 
                'LOCAL_RANK', 'CCL_WORKER_COUNT', 'CCL_LOG_LEVEL']
    for v in env_vars:
        logging.info(f"{v} = {os.environ.get(v, 'Not set')}")

##############################################################################
# 4) FSDP Wrapping Helpers 
##############################################################################
def get_transformer_layer_cls():
    """Return the transformer layer class for auto-wrapping."""
    # Import the correct layer class from your model
    import torch.nn as nn
    return nn.TransformerEncoderLayer  # Using the standard PyTorch implementation

def create_fsdp_model(model_instance, device, rank, world_size):
    """Create properly wrapped FSDP model with comprehensive logging."""
    if rank == 0:
        logging.info("Starting FSDP model wrapping process...")
    
    # Size based policy is more reliable in this case
    size_policy = functools.partial(
        size_based_auto_wrap_policy, 
        min_num_params=10000,  # Lower threshold to catch more modules
    )
    
    if rank == 0:
        logging.info(f"Using size-based FSDP wrapping policy with min_num_params=10000")
    
    # Try with FSDP wrapping via context manager
    with enable_wrap(
        wrapper_cls=FSDP,
        auto_wrap_policy=size_policy,
        device_id=device,
    ):
        wrapped_model = wrap(model_instance)
    
    # Now apply top-level FSDP wrapping with tuned options
    model = FSDP(
        wrapped_model,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Most aggressive sharding
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        sync_module_states=True,  # Important for consistent init
    )
    
    if rank == 0:
        # Count how many FSDP-wrapped modules we have
        fsdp_modules = [m for m in model.modules() if isinstance(m, FSDP)]
        logging.info(f"Created FSDP model with {len(fsdp_modules)} FSDP-wrapped submodules")
        
    return model

##############################################################################
# 5) Main FSDP Training Function
##############################################################################
def fsdp_main():
    # Fix random seed
    torch.manual_seed(3407)
    np.random.seed(3407)

    # 0) Init Dist for oneCCL with better error detection
    mpi_world_size = int(os.environ.get('PMI_SIZE', '-1'))
    mpi_rank = int(os.environ.get('PMI_RANK', '-1'))
    
    if mpi_world_size > 0:
        # MPI environment detected
        os.environ['WORLD_SIZE'] = str(mpi_world_size)
        os.environ['RANK'] = str(mpi_rank)
        os.environ['LOCAL_RANK'] = str(mpi_rank % torch.xpu.device_count())
    else:
        # Fallback to env vars
        os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '1')
        os.environ['RANK'] = os.environ.get('RANK', '0')
        os.environ['LOCAL_RANK'] = os.environ.get('LOCAL_RANK', '0')

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    # Added CCL-specific settings
    os.environ['CCL_WORKER_COUNT'] = os.environ.get('CCL_WORKER_COUNT', '1')
    os.environ['CCL_LOG_LEVEL'] = os.environ.get('CCL_LOG_LEVEL', 'ERROR')
    
    # Initialize process group with clearer error handling
    try:
        dist.init_process_group(backend='ccl')
        print(f"Process group initialized successfully with backend 'ccl'")
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        raise

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))

    device = f"xpu:{local_rank}"
    torch.xpu.set_device(device)

    # 1) Parse config
    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    apply_amp = config.get('apply_amp', False)

    # Logging setup
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"apply_amp = {apply_amp}")
        logging.info(f"Running on {world_size} XPU(s). local_rank={local_rank}")
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        os.environ['WANDB_MODE'] = 'disabled'

    chunk_batch = config.get('chunk_batch', 40)
    if local_rank == 0:
        logging.info(f"chunk_batch = {chunk_batch}")

    # Check environment
    diagnose_fsdp_implementation(local_rank)

    # 2) Setup W&B on rank0
    run_name = f"BT_FSDP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if local_rank == 0:
        wandb_run = wandb.init(project="btfm-iterable", name=run_name, config=config)
    else:
        wandb_run = None

    # total steps approximate
    total_steps = config['epochs'] * config['total_samples'] // config['batch_size'] // world_size
    if local_rank == 0:
        logging.info(f"Total steps = {total_steps} (approx)")

    # 3) Build model
    s2_num_heads = 16
    s2_num_layers = 32
    s2_dim_feedforward = 1024
    s1_num_heads = 8
    s1_num_layers = 16
    s1_dim_feedforward = 512

    if local_rank == 0:
        wandb.config.update({
            "s2_num_heads": s2_num_heads,
            "s2_num_layers": s2_num_layers,
            "s2_dim_feedforward": s2_dim_feedforward,
            "s1_num_heads": s1_num_heads,
            "s1_num_layers": s1_num_layers,
            "s1_dim_feedforward": s1_dim_feedforward
        })

    # Create your submodules
    s2_enc = TransformerEncoder(
        band_num=12,
        latent_dim=config['latent_dim'],
        nhead=s2_num_heads,
        num_encoder_layers=s2_num_layers,
        dim_feedforward=s2_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s2']
    )
    s1_enc = TransformerEncoder(
        band_num=4,
        latent_dim=config['latent_dim'],
        nhead=s1_num_heads,
        num_encoder_layers=s1_num_layers,
        dim_feedforward=s1_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s1']
    )

    if config['fusion_method'] == 'concat':
        proj_in_dim = config['latent_dim'] * 2
    else:
        proj_in_dim = config['latent_dim']
    projector = ProjectionHead(
        proj_in_dim,
        config['projector_hidden_dim'],
        config['projector_out_dim']
    )

    if config['fusion_method'] == 'transformer':
        model_instance = MultimodalBTModel(
            s2_enc, s1_enc, projector,
            fusion_method=config['fusion_method'],
            return_repr=True,
            latent_dim=config['latent_dim']
        )
    else:
        model_instance = MultimodalBTModel(
            s2_enc, s1_enc, projector,
            fusion_method=config['fusion_method'],
            return_repr=True
        )

    if local_rank == 0:
        orig_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        logging.info(f"Original model parameters (pre-FSDP): ~{orig_params:,}")

    analyze_model_structure(model_instance, local_rank)
    
    # Move model to device before FSDP wrapping for cleaner tracing
    model_instance = model_instance.to(device)

    dist.barrier()
    if local_rank == 0:
        logging.info("=== Memory usage before FSDP wrap ===")
    check_xpu_memory_usage(local_rank, world_size, device)

    # FSDP wrapping with improved approach
    model = create_fsdp_model(model_instance, device, local_rank, world_size)

    if local_rank == 0:
        logging.info("FSDP wrap complete (FULL_SHARD). Checking device placement...")

    check_model_device_placement(model, local_rank)
    dist.barrier()
    if local_rank == 0:
        logging.info("=== Memory usage after FSDP wrap ===")
    check_xpu_memory_usage(local_rank, world_size, device)

    print_model_parameter_stats(model, local_rank, world_size, device)

    # Barlow Twins Loss
    barlow_lambda = config['barlow_lambda']
    if barlow_lambda > 0.05:
        barlow_lambda = 0.05
    criterion = BarlowTwinsLoss(lambda_coeff=barlow_lambda)
    if local_rank == 0:
        logging.info(f"Using BarlowTwinsLoss with lambda_coeff={barlow_lambda} (orig: {config['barlow_lambda']})")

    # Optimizer
    weight_params = [p for n, p in model.named_parameters() if p.ndim > 1]
    bias_params = [p for n, p in model.named_parameters() if p.ndim == 1]
    base_lr = config['learning_rate'] * 0.1  # lower for stability
    optimizer = torch.optim.AdamW(
        [{'params': weight_params}, {'params': bias_params}],
        lr=base_lr,
        weight_decay=1e-6,
        eps=1e-5
    )
    if local_rank == 0:
        logging.info(f"Optimizer created with LR={base_lr:.6f}, eps=1e-5")

    scaler = torch.amp.GradScaler() if apply_amp else None

    ########################################################################
    # Training logic on chunked data
    ########################################################################
    def train_one_chunk(chunk_files, epoch, step, examples, last_time, last_examples, rolling_loss):
        """
        - Build ChunkDataset from chunk_files
        - DistributedSampler
        - BarlowTwins training step
        - Return updated step, examples, times, rolling_loss
        """
        aug1_s2_dir = os.path.join(config['data_root'], 'aug1', 's2')
        aug2_s2_dir = os.path.join(config['data_root'], 'aug2', 's2')
        aug1_s1_dir = os.path.join(config['data_root'], 'aug1', 's1')
        aug2_s1_dir = os.path.join(config['data_root'], 'aug2', 's1')

        aug1_s2_paths = [os.path.join(aug1_s2_dir, fn) for fn in chunk_files]
        aug2_s2_paths = [os.path.join(aug2_s2_dir, fn) for fn in chunk_files]
        aug1_s1_paths = [os.path.join(aug1_s1_dir, fn) for fn in chunk_files]
        aug2_s1_paths = [os.path.join(aug2_s1_dir, fn) for fn in chunk_files]

        ds = ChunkDataset(aug1_s2_paths, aug2_s2_paths, aug1_s1_paths, aug2_s1_paths)
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False, drop_last=True)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=config['batch_size'],
            sampler=sampler,
            num_workers=config['num_workers'],
            drop_last=True,
            pin_memory=True,
        )
        sampler.set_epoch(epoch)

        if local_rank == 0:
            logging.info(f"   => chunk has {len(ds)} samples, steps in loader = {len(loader)}")

        model.train()
        for batch_data in loader:
            s2_aug1 = batch_data['s2_aug1'].to(device, non_blocking=True)
            s2_aug2 = batch_data['s2_aug2'].to(device, non_blocking=True)
            s1_aug1 = batch_data['s1_aug1'].to(device, non_blocking=True)
            s1_aug2 = batch_data['s1_aug2'].to(device, non_blocking=True)

            # Adjust LR
            from utils.lr_scheduler import adjust_learning_rate
            adjust_learning_rate(optimizer, step, total_steps,
                                 config['learning_rate'],
                                 config['warmup_ratio'],
                                 config['plateau_ratio'])
            optimizer.zero_grad()

            if apply_amp:
                with torch.amp.autocast(device_type="xpu"):
                    z1, repr1 = model(s2_aug1, s1_aug1)
                    z2, repr2 = model(s2_aug2, s1_aug2)
                    loss_main, bar_main, off_main = criterion(z1, z2)

                    # optional mixup
                    loss_mix = 0.0
                    if config.get('apply_mixup', False):
                        B = s2_aug1.size(0)
                        idxs = torch.randperm(B, device=device)
                        alpha = torch.distributions.Beta(config['beta_alpha'], config['beta_beta']).sample().to(device)
                        y_m_s2 = alpha * s2_aug1 + (1 - alpha) * s2_aug2[idxs, :]
                        y_m_s1 = alpha * s1_aug1 + (1 - alpha) * s1_aug2[idxs, :]
                        z_m, _ = model(y_m_s2, y_m_s1)
                        cc_m_a = compute_cross_correlation(z_m, z1)
                        cc_m_b = compute_cross_correlation(z_m, z2)
                        cc_z1_z1 = compute_cross_correlation(z1, z1)
                        cc_z2idx_z1 = compute_cross_correlation(z2[idxs], z1)
                        cc_z1_z2 = compute_cross_correlation(z1, z2)
                        cc_z2idx_z2 = compute_cross_correlation(z2[idxs], z2)
                        cc_m_a_gt = alpha * cc_z1_z1 + (1 - alpha) * cc_z2idx_z1
                        cc_m_b_gt = alpha * cc_z1_z2 + (1 - alpha) * cc_z2idx_z2
                        diff_a = (cc_m_a - cc_m_a_gt).pow(2).sum()
                        diff_b = (cc_m_b - cc_m_b_gt).pow(2).sum()
                        loss_mix = config['mixup_lambda'] * barlow_lambda * (diff_a + diff_b)

                    total_loss = loss_main + loss_mix

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                z1, repr1 = model(s2_aug1, s1_aug1)
                z2, repr2 = model(s2_aug2, s1_aug2)
                loss_main, bar_main, off_main = criterion(z1, z2)

                loss_mix = 0.0
                if config.get('apply_mixup', False):
                    B = s2_aug1.size(0)
                    idxs = torch.randperm(B, device=device)
                    alpha = torch.distributions.Beta(config['beta_alpha'], config['beta_beta']).sample().to(device)
                    y_m_s2 = alpha * s2_aug1 + (1 - alpha) * s2_aug2[idxs, :]
                    y_m_s1 = alpha * s1_aug1 + (1 - alpha) * s1_aug2[idxs, :]
                    z_m, _ = model(y_m_s2, y_m_s1)
                    # cross correlation checks
                    cc_m_a = compute_cross_correlation(z_m, z1)
                    cc_m_b = compute_cross_correlation(z_m, z2)
                    cc_z1_z1 = compute_cross_correlation(z1, z1)
                    cc_z2idx_z1 = compute_cross_correlation(z2[idxs], z1)
                    cc_z1_z2 = compute_cross_correlation(z1, z2)
                    cc_z2idx_z2 = compute_cross_correlation(z2[idxs], z2)
                    cc_m_a_gt = alpha * cc_z1_z1 + (1 - alpha) * cc_z2idx_z1
                    cc_m_b_gt = alpha * cc_z1_z2 + (1 - alpha) * cc_z2idx_z2
                    diff_a = (cc_m_a - cc_m_a_gt).pow(2).sum()
                    diff_b = (cc_m_b - cc_m_b_gt).pow(2).sum()
                    loss_mix = config['mixup_lambda'] * barlow_lambda * (diff_a + diff_b)
                
                total_loss = loss_main + loss_mix
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # skip if NaNs
                skip_step = False
                for p in model.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        skip_step = True
                        break
                if skip_step:
                    if local_rank == 0:
                        logging.warning(f"NaN gradient at step {step}, skipping optimizer step.")
                else:
                    optimizer.step()

            # logging
            batch_sz = s2_aug1.size(0)
            examples += batch_sz

            if (step % config['log_interval_steps'] == 0) and (local_rank == 0):
                current_time = time.time()
                speed = (examples - last_examples) / (current_time - last_time + 1e-8)
                last_time = current_time
                last_examples = examples

                # rolling average
                rolling_loss.append(loss_main.item())
                if len(rolling_loss) > 40:
                    rolling_loss = rolling_loss[-40:]
                avg_loss = sum(rolling_loss) / len(rolling_loss)
                lr_curr = optimizer.param_groups[0]['lr']

                # rankme measure
                try:
                    erank_z = rankme(z1)
                    erank_repr = rankme(repr1)
                except:
                    erank_z, erank_repr = 0.0, 0.0
                    logging.warning("Rank computation failed (SVD fallback).")

                logging.info(
                    f"[Epoch={epoch}, Step={step}] "
                    f"Loss={loss_main.item():.2f}, MixLoss={loss_mix:.2f}, AvgLoss={avg_loss:.2f}, "
                    f"LR={lr_curr:.6f}, batch_size={batch_sz}, Examples/s={speed:.2f}, "
                    f"Rank(z)={erank_z:.4f}, Rank(repr)={erank_repr:.4f}"
                )
                wandb_dict = {
                    "epoch": epoch,
                    "loss_main": loss_main.item(),
                    "loss_mix": loss_mix,
                    "avg_loss": avg_loss,
                    "lr": lr_curr,
                    "examples_sec": speed,
                    "rank_z": erank_z,
                    "rank_repr": erank_repr,
                    "total_loss": total_loss.item()
                }

                # periodic CC plots
                if step % (10 * config['log_interval_steps']) == 0:
                    try:
                        fig_cc = plot_cross_corr(z1, z2)
                        wandb_dict["cross_corr"] = wandb.Image(fig_cc)
                        plt.close(fig_cc)
                    except:
                        pass
                    try:
                        fig_repr = plot_cross_corr(repr1, repr2)
                        wandb_dict["cross_corr_repr"] = wandb.Image(fig_repr)
                        plt.close(fig_repr)
                    except:
                        pass

                wandb.log(wandb_dict, step=step)

            # validation & checkpoint
            if (config['val_interval_steps'] > 0) and (step > 0) \
               and (step % config['val_interval_steps'] == 0) and (local_rank == 0):
                # optional validation
                if all(config.get(k) for k in [
                    'val_s2_bands_file_path', 'val_s2_masks_file_path', 'val_s2_doy_file_path',
                    'val_s1_asc_bands_file_path', 'val_s1_asc_doy_file_path',
                    'val_s1_desc_bands_file_path', 'val_s1_desc_doy_file_path',
                    'val_labels_path'
                ]):
                    from datasets.ssl_dataset import AustrianCropValidation
                    val_dataset = AustrianCropValidation(
                        s2_bands_file_path=config['val_s2_bands_file_path'],
                        s2_masks_file_path=config['val_s2_masks_file_path'],
                        s2_doy_file_path=config['val_s2_doy_file_path'],
                        s1_asc_bands_file_path=config['val_s1_asc_bands_file_path'],
                        s1_asc_doy_file_path=config['val_s1_asc_doy_file_path'],
                        s1_desc_bands_file_path=config['val_s1_desc_bands_file_path'],
                        s1_desc_doy_file_path=config['val_s1_desc_doy_file_path'],
                        labels_path=config['val_labels_path'],
                        sample_size_s2=config['sample_size_s2'],
                        sample_size_s1=config['sample_size_s1'],
                        min_valid_timesteps=0,
                        standardize=True
                    )
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
                    model.eval()

                    val_acc, val_f1, val_cm = linear_probe_evaluate(model, val_loader, device=device)
                    wandb.log({"val_acc": val_acc, "val_f1": val_f1}, step=step)
                    logging.info(f"Validation at step={step}: val_acc={val_acc:.4f}, F1={val_f1:.4f}")
                    logging.info(f"Confusion matrix:\n{val_cm}")

                    # cm plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(val_cm, interpolation='nearest', cmap=plt.cm.Blues)
                    ax.figure.colorbar(im, ax=ax)
                    ax.set(xticks=range(val_cm.shape[1]),
                           yticks=range(val_cm.shape[0]),
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

                    # Save best checkpoint
                    global best_val_acc
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_fsdp_checkpoint(model, optimizer, epoch, step, best_val_acc, best_ckpt_path)
                    model.train()

            step += 1

        # cleanup
        try:
            loader._iterator._shutdown_workers()
        except:
            pass
        del loader
        del sampler
        del ds
        gc.collect()

        return step, examples, last_time, last_examples, rolling_loss
    
    ########################################################################
    # Save checkpoint helpers
    ########################################################################
    def save_fsdp_checkpoint(model, optimizer, epoch, step, val_acc, filename):
        """Save FSDP checkpoint from rank 0 with proper consolidation."""
        if local_rank == 0:
            logging.info(f"Saving checkpoint to {filename}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Ensure model is in eval mode for saving
        model.eval()
        
        # Use FSDP state_dict utilities
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        try:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                model_state = model.state_dict()
                
                if local_rank == 0:
                    # Only save on rank 0
                    checkpoint = {
                        'epoch': epoch,
                        'step': step,
                        'val_acc': val_acc,
                        'model': model_state,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, filename)
                    logging.info(f"✅ Checkpoint saved successfully to {filename}")
        except Exception as e:
            logging.error(f"Error during checkpoint saving: {e}")
            # Fallback: try to save just the local state dict
            if local_rank == 0:
                try:
                    checkpoint = {
                        'epoch': epoch,
                        'step': step,
                        'val_acc': val_acc,
                        'model': model.state_dict(),  # This will be a sharded state dict
                        'optimizer': optimizer.state_dict(),
                        'is_sharded': True,
                    }
                    torch.save(checkpoint, filename + '.sharded')
                    logging.info(f"⚠️ Saved sharded checkpoint as fallback")
                except:
                    logging.error("Failed to save even the sharded checkpoint")
        
        # Return to train mode
        model.train()
        
        # Ensure all processes are synchronized after saving
        dist.barrier()

    ########################################################################
    # 6) Global Training
    ########################################################################
    step = 0
    examples = 0
    last_time = time.time()
    last_examples = 0
    rolling_loss = []
    global best_val_acc
    best_val_acc = 0.0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_{timestamp}.pt")
    
    # Create checkpoint directory if it doesn't exist
    if local_rank == 0:
        os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)

    # Print memory usage before training
    if local_rank == 0:
        logging.info("=== Memory usage before training ===")
    check_xpu_memory_usage(local_rank, world_size, device)

    for epoch in range(config['epochs']):
        # If you want to re-generate data each epoch via Rust:
        # if local_rank == 0:
        #     aug1_dir = os.path.join(config['data_root'], 'aug1')
        #     aug2_dir = os.path.join(config['data_root'], 'aug2')
        #     remove_dir(aug1_dir)
        #     remove_dir(aug2_dir)
        #     logging.info(f"Epoch {epoch} generating new data with rust_cmd...")
        #     subprocess.run(config['rust_cmd'], shell=True, check=True)
        #     logging.info(f"Data generation done for epoch={epoch}.")
        dist.barrier()

        aug1_s2_dir = os.path.join(config['data_root'], 'aug1', 's2')
        s2_file_names = os.listdir(aug1_s2_dir)
        np.random.shuffle(s2_file_names)
        total_files = len(s2_file_names)
        if local_rank == 0:
            logging.info(f"Epoch {epoch}: found {total_files} files in s2/aug1/")

        chunk_start = 0
        while chunk_start < total_files:
            chunk_end = min(chunk_start + chunk_batch, total_files)
            chunk_files = s2_file_names[chunk_start:chunk_end]

            if local_rank == 0:
                logging.info(f"Epoch {epoch}, chunk [{chunk_start}:{chunk_end}], loading {len(chunk_files)} files...")
            step, examples, last_time, last_examples, rolling_loss = train_one_chunk(
                chunk_files, epoch, step, examples, last_time, last_examples, rolling_loss
            )
            chunk_start = chunk_end
            
            # Check memory usage periodically
            if chunk_start % (5 * chunk_batch) == 0:
                if local_rank == 0:
                    logging.info("=== Periodic memory check ===")
                check_xpu_memory_usage(local_rank, world_size, device)

        # Save epoch checkpoint
        if local_rank == 0:
            logging.info(f"Epoch {epoch} done, step={step}")
            
        # Save epoch checkpoint using FSDP-aware function
        epoch_ckpt_path = os.path.join("checkpoints", "ssl", f"epoch_{epoch}_{timestamp}.pt")
        save_fsdp_checkpoint(model, optimizer, epoch, step, best_val_acc, epoch_ckpt_path)

    if local_rank == 0:
        logging.info("Training completed.")
        wandb_run.finish()

    # Clean up
    dist.destroy_process_group()

##############################################################################
# 7) Entry Point
##############################################################################
if __name__ == "__main__":
    try:
        fsdp_main()
    except Exception as e:
        import traceback
        print(f"Error in FSDP main: {e}")
        print(traceback.format_exc())
        # Make sure all processes exit
        import sys
        sys.exit(1)