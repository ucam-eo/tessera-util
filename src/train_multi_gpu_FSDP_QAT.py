#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 运行方式:
#   单节点:
# torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 src/train_multi_gpu_FSDP.py --config configs/your_config_with_qat.py
#   多节点:
#     节点1: torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=gpu-16:29400 src/train_multi_gpu_FSDP.py --config configs/your_config_with_qat.py
#     节点2: torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=gpu-16:29400 src/train_multi_gpu_FSDP.py --config configs/your_config_with_qat.py

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

from datetime import timedelta
import math
import time
import gc
import argparse
import logging
import subprocess
from datetime import datetime
from contextlib import nullcontext
import json
import copy

import numpy as np
import torch
torch.set_float32_matmul_precision('high') 
# torch.autograd.set_detect_anomaly(True) 

import torch.amp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.api import FullOptimStateDictConfig

import wandb
import matplotlib.pyplot as plt

# Import fvcore for FLOPs counting
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("Warning: fvcore not installed. FLOPs counting will be disabled.")
    print("Install with: pip install fvcore")
    FlopCountAnalysis = None

# ============== 自定义模块 / 函数 ===============
from models.modules import TransformerEncoder, ProjectionHead 
from models.quantization import FakeQuantizeRepresentation #, quantize_tensor_symmetric, dequantize_tensor_symmetric # These two are not directly used in train script
from models.ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate 
from utils.metrics import linear_probe_evaluate, rankme
from utils.misc import remove_dir, plot_cross_corr 

# FSDP辅助函数：获取GPU内存使用情况
def get_gpu_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

# FSDP辅助函数：打印模型信息 
def print_model_info(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    def get_model_structure(m, depth=0, max_depth=20):
        if depth > max_depth: return "  " * depth + "...\n"
        res = ""
        for n, child in m.named_children():
            res += "  " * depth + f"({n}): {child.__class__.__name__}\n"
            res += get_model_structure(child, depth + 1, max_depth)
        return res
    model_structure = get_model_structure(model)
    return (f"\n{name} Information:\n"
            f"Model Class: {model.__class__.__name__}\n"
            f"Total Parameters: {total_params:,}\n"
            f"Trainable Parameters: {trainable_params:,}\n"
            f"Model Structure:\n{model_structure}")

# 创建评估模型的辅助函数
def create_evaluation_model(config, device):
    s2_num_heads = config['s2_num_heads']; s2_num_layers = config['s2_num_layers']
    s2_dim_feedforward = config['s2_dim_feedforward']
    s1_num_heads = config['s1_num_heads']; s1_num_layers = config['s1_num_layers']
    s1_dim_feedforward = config['s1_dim_feedforward']
    
    s2_enc = TransformerEncoder(
        band_num=10, latent_dim=config['latent_dim'], nhead=s2_num_heads,
        num_encoder_layers=s2_num_layers, dim_feedforward=s2_dim_feedforward,
        dropout=0.1, max_seq_len=config['sample_size_s2'],
    ).to(device)

    s1_enc = TransformerEncoder(
        band_num=2, latent_dim=config['latent_dim'], nhead=s1_num_heads,
        num_encoder_layers=s1_num_layers, dim_feedforward=s1_dim_feedforward,
        dropout=0.1, max_seq_len=config['sample_size_s1'],
    ).to(device)

    projector_eval = ProjectionHead( 
        config['latent_dim'], 
        config['projector_hidden_dim'],
        config['projector_out_dim']
    ).to(device)

    eval_model = MultimodalBTModel(
        s2_enc, s1_enc, projector_eval,
        fusion_method=config['fusion_method'],
        return_repr=True, 
        latent_dim=config['latent_dim'],
        apply_qat_representation=config.get('apply_qat_representation', False),
        qat_representation_bits=config.get('qat_representation_bits', 8)
    ).to(device)
    
    return eval_model

##########################################################################
# 1) ChunkDataset 
##########################################################################
class ChunkDataset(Dataset):
    def __init__(self, aug1_s2_files, aug2_s2_files, aug1_s1_files, aug2_s1_files):
        super().__init__()
        self.all_samples = []
        for idx_file in range(len(aug1_s2_files)):
            arr_aug1_s2 = np.load(aug1_s2_files[idx_file])
            arr_aug2_s2 = np.load(aug2_s2_files[idx_file])
            arr_aug1_s1 = np.load(aug1_s1_files[idx_file])
            arr_aug2_s1 = np.load(aug2_s1_files[idx_file])
            n_samples = min(arr_aug1_s2.shape[0], arr_aug2_s2.shape[0], arr_aug1_s1.shape[0], arr_aug2_s1.shape[0])
            for i in range(n_samples):
                self.all_samples.append((arr_aug1_s2[i], arr_aug2_s2[i], arr_aug1_s1[i], arr_aug2_s1[i]))
        np.random.shuffle(self.all_samples)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        s2_aug1_np, s2_aug2_np, s1_aug1_np, s1_aug2_np = self.all_samples[index]
        return {
            "s2_aug1": torch.tensor(s2_aug1_np, dtype=torch.float32),
            "s2_aug2": torch.tensor(s2_aug2_np, dtype=torch.float32),
            "s1_aug1": torch.tensor(s1_aug1_np, dtype=torch.float32),
            "s1_aug2": torch.tensor(s1_aug2_np, dtype=torch.float32)
        }

##########################################################################
# 2) 主训练脚本 (FSDP + chunk-based loading)
##########################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training (FSDP + chunk-based loading)")
    parser.add_argument('--config', type=str, default="configs/ssl_config.py",
                        help="Path to config file (e.g. configs/ssl_config.py)")
    return parser.parse_args()

def adjust_single_group_lr(iter_count, total_iters, learning_rate, warmup_ratio=0.1, plateau_ratio=0.7, is_resume=False, resume_warmup_steps=0): 
    if is_resume:
        # 对于恢复训练，直接进行余弦退火，不进行warmup
        if iter_count < resume_warmup_steps:
            # 如果设置了resume_warmup_steps，则进行短暂的warmup
            return learning_rate * (iter_count / resume_warmup_steps)
        else:
            # 余弦退火
            effective_iter = iter_count - resume_warmup_steps
            effective_total = total_iters - resume_warmup_steps
            if effective_total <= 0:
                return learning_rate
            decay_progress = effective_iter / effective_total
            decay_progress = max(0.0, min(1.0, decay_progress))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            return learning_rate * cosine_decay
    else:
        # 原有的逻辑
        warmup_iters = int(total_iters * warmup_ratio)
        cosine_start_iter = warmup_iters 
        
        if iter_count < warmup_iters:
            return learning_rate * (iter_count / warmup_iters) if warmup_iters > 0 else learning_rate
        else: 
            if total_iters <= cosine_start_iter:
                 return learning_rate 
            
            decay_progress = (iter_count - cosine_start_iter) / (total_iters - cosine_start_iter)
            decay_progress = max(0.0, min(1.0, decay_progress))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            return learning_rate * cosine_decay

def save_fsdp_model(model, optimizer, config, filename, step, epoch, global_rank, is_final=False):
    start_time = time.time()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    full_model_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_state_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    dist.barrier()
    model_state = None 
    optim_state = None 

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_model_state_config, optim_state_config):
        model_state = model.state_dict()
        if is_final: 
            optim_state = FSDP.optim_state_dict(model, optimizer, optim_state_dict_config=optim_state_config)

    if global_rank == 0:
        checkpoint = {
            'epoch': epoch, 'step': step,
            'model_state_dict': model_state, 'config': config,
        }
        if optim_state is not None:
            checkpoint['optimizer_state_dict'] = optim_state
        
        temp_filename = f"{filename}.temp"
        torch.save(checkpoint, temp_filename)
        os.replace(temp_filename, filename)
        logging.info(f"Model saved to {filename} (took {time.time() - start_time:.2f}s)")
    
    dist.barrier()
    return model_state if global_rank == 0 else None

def load_model_from_checkpoint(checkpoint_path, config, device):
    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    eval_model = create_evaluation_model(config, device)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_key = 'model_state_dict' 
    
    state_dict = checkpoint[state_key]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('_fsdp_wrapped_module.'): 
            new_key = key[len('_fsdp_wrapped_module.'):]
        elif key.startswith('_orig_mod.'): 
             new_key = key[len('_orig_mod.'):]
        new_state_dict[new_key] = value

    eval_model.load_state_dict(new_state_dict, strict=True) 
    eval_model.eval()
    return eval_model

def load_checkpoint_for_resume(checkpoint_path, model, optimizer, scaler, device, global_rank):
    """加载checkpoint用于恢复训练"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if global_rank == 0:
        logging.info(f"Loading checkpoint for resume training: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型状态
    model_state_dict = checkpoint['model_state_dict']
    
    # 处理FSDP包装的模型状态
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(model_state_dict)
    
    # 加载优化器状态（如果存在）
    if 'optimizer_state_dict' in checkpoint:
        if global_rank == 0:
            logging.info("Loading optimizer state from checkpoint")
        optim_state_dict = checkpoint['optimizer_state_dict']
        optim_state = FSDP.optim_state_dict_to_load(
            model, optimizer, optim_state_dict
        )
        optimizer.load_state_dict(optim_state)
    
    # 加载scaler状态（如果存在）
    if 'scaler_state_dict' in checkpoint and scaler is not None:
        if global_rank == 0:
            logging.info("Loading scaler state from checkpoint")
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # 返回训练状态信息
    resume_epoch = checkpoint.get('epoch', 0)
    resume_step = checkpoint.get('step', 0)
    resume_config = checkpoint.get('config', {})
    
    if global_rank == 0:
        logging.info(f"Resumed from epoch {resume_epoch}, step {resume_step}")
    
    return resume_epoch, resume_step, resume_config

# Helper function to compute FLOPs for a single forward pass
def compute_flops_single_pass(model, inputs, device):
    """Compute FLOPs for a single forward pass"""
    if FlopCountAnalysis is None:
        return 0
    
    try:
        # Create a wrapper model that only does forward pass
        class ForwardOnlyWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, s2, s1):
                # Just forward pass, no loss computation
                return self.model(s2, s1)
        
        # Use the wrapped model for FLOP counting
        forward_model = ForwardOnlyWrapper(model)
        flops = FlopCountAnalysis(forward_model, inputs)
        return flops.total()
    except Exception as e:
        if hasattr(model, 'module'):
            # Try with unwrapped model
            try:
                forward_model = ForwardOnlyWrapper(model.module)
                flops = FlopCountAnalysis(forward_model, inputs)
                return flops.total()
            except:
                pass
        logging.warning(f"Failed to compute FLOPs: {e}")
        return 0

best_val_acc = 0.0

def main():
    torch.manual_seed(3407); np.random.seed(3407)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60)) 
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    nproc_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    nnodes = world_size // nproc_per_node if nproc_per_node > 0 else world_size # handle nproc_per_node = 0 if no CUDA

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if global_rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
        if not os.environ.get("WANDB_MODE", "") == "offline": 
            os.environ['WANDB_MODE'] = 'disabled'

    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f: exec(f.read(), config_module)
    config = config_module['config']

    apply_amp = config.get('apply_amp', False)
    if global_rank == 0: logging.info(f"Apply AMP (Automatic Mixed Precision) = {apply_amp}")

    if global_rank == 0:
        logging.info(f"Running on {world_size} GPU(s) across {nnodes} node(s). GlobalRank={global_rank}, LocalRank={local_rank}, device={device}")
        logging.info(f"Initial GPU memory usage: {get_gpu_memory_usage():.2f} MB")
    
    if config.get("disable_wandb_git", False): os.environ['WANDB_DISABLE_GIT'] = 'true'
    
    # 检查是否从checkpoint恢复训练
    is_resume = config.get('resume_from_checkpoint', None) is not None
    if is_resume and global_rank == 0:
        logging.info(f"Resume training from checkpoint: {config['resume_from_checkpoint']}")
    
    run_name = f"FSDP_BT_QAT_{timestamp}" if config.get('apply_qat_representation') else f"FSDP_BT_{timestamp}"
    if is_resume:
        run_name = f"RESUME_{run_name}"
    
    wandb_run = None
    if global_rank == 0:
        if not os.environ.get("WANDB_API_KEY") and config.get("wandb_api_key"):
             os.environ["WANDB_API_KEY"] = config["wandb_api_key"]
        if not os.environ.get("WANDB_API_KEY"): # Default if still not set
             os.environ["WANDB_API_KEY"] = "b03eca52bd30c1fa9bf185ae3ee91d9276f2f92a" # Original hardcoded

        wandb_run = wandb.init(
            project=config.get("wandb_project", "btfm-iterable-qat"), 
            name=run_name, 
            config=config,
            resume="allow" if is_resume else None
        )
        artifact = wandb.Artifact('source-code', type='code')
        relevant_files = [
            __file__, # Log the current script itself
            'src/models/modules.py', 'src/models/ssl_model.py', 'src/models/quantization.py', 
            'src/utils/lr_scheduler.py', 'src/utils/metrics.py', 'src/utils/misc.py',
            args_cli.config 
        ]
        for f_path in relevant_files:
            if os.path.exists(f_path): artifact.add_file(f_path)
            else: logging.warning(f"W&B: Source file {f_path} not found for artifact.")
        wandb.log_artifact(artifact)

    total_steps = 1000 # Default
    if config.get('total_samples', 0) > 0 and config.get('batch_size', 0) > 0 and world_size > 0:
        total_steps = config['epochs'] * config['total_samples'] // config['batch_size'] // world_size
    else:
        if global_rank == 0: logging.warning(f"Could not calculate total_steps accurately due to zero values in config or world_size. Using default: {total_steps}")
    
    total_steps_approx = total_steps # Define total_steps_approx for logging

    if global_rank == 0: logging.info(f"Total calculated steps per rank (approx) = {total_steps_approx}")
    
    qat_start_step_cfg = config.get('qat_representation_start_step', float('inf'))
    if isinstance(qat_start_step_cfg, float) and 0 < qat_start_step_cfg < 1: # If it's a ratio
        config['qat_representation_start_step'] = int(total_steps_approx * qat_start_step_cfg)
        if global_rank == 0: logging.info(f"QAT for representation will start at step: {config['qat_representation_start_step']}")
    elif not isinstance(qat_start_step_cfg, int): # If not ratio and not int, set to inactive
        config['qat_representation_start_step'] = float('inf')

    s2_enc = TransformerEncoder(
        band_num=10, latent_dim=config['latent_dim'], nhead=config['s2_num_heads'],
        num_encoder_layers=config['s2_num_layers'], dim_feedforward=config['s2_dim_feedforward'],
        dropout=0.1, max_seq_len=config['sample_size_s2']).to(device)
    s1_enc = TransformerEncoder(
        band_num=2, latent_dim=config['latent_dim'], nhead=config['s1_num_heads'],
        num_encoder_layers=config['s1_num_layers'], dim_feedforward=config['s1_dim_feedforward'],
        dropout=0.1, max_seq_len=config['sample_size_s1']).to(device)
    
    projector = ProjectionHead(
        config['latent_dim'], 
        config['projector_hidden_dim'], 
        config['projector_out_dim']
    ).to(device)

    model = MultimodalBTModel(
        s2_enc, s1_enc, projector,
        fusion_method=config['fusion_method'],
        return_repr=True, 
        latent_dim=config['latent_dim'],
        apply_qat_representation=config.get('apply_qat_representation', False),
        qat_representation_bits=config.get('qat_representation_bits', 8)
    ).to(device)

    if local_rank == 0: 
        node_id = global_rank // nproc_per_node if nproc_per_node > 0 else 0
        logging.info(f"Node {node_id}, LocalRank {local_rank} - Before FSDP - Total params: {sum(p.numel() for p in model.parameters()):,}")
        logging.info(f"GPU memory usage before FSDP: {get_gpu_memory_usage():.2f} MB")

    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])
    
    # Fixed: Changed parameter name from 'unwrapped_params' to 'nonwrapped_numel'
    def custom_auto_wrap_policy(module, recurse, nonwrapped_numel: int): 
        return isinstance(module, TransformerEncoder)

    mixed_precision_policy = None
    if apply_amp:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )
    
    fsdp_config_dict = {
        "auto_wrap_policy": custom_auto_wrap_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD, 
        "device_id": device, 
        "sync_module_states": True, "forward_prefetch": True,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE, 
        "cpu_offload": CPUOffload(offload_params=False), 
        "use_orig_params": True, 
    }
    if mixed_precision_policy:
        fsdp_config_dict["mixed_precision"] = mixed_precision_policy

    if global_rank == 0:
        logging.info(f"FSDP Configuration: {json.dumps({k: str(v) for k, v in fsdp_config_dict.items()}, indent=2)}")

    model = FSDP(model, **fsdp_config_dict)

    if global_rank == 0:
        logging.info(f"Model structure after FSDP wrapping (first 5000 chars):\n{str(model)[:5000]}...")
        fsdp_params = sum(p.numel() for p in model.parameters())
        logging.info(f"After FSDP wrapping - Parameters on rank {global_rank}: {fsdp_params:,}")
        logging.info(f"GPU memory usage after FSDP: {get_gpu_memory_usage():.2f} MB")

    if config.get('use_torch_compile', False):
        if global_rank == 0: logging.info("Attempting to torch.compile the FSDP model...")
        try:
            # For AMD, "inductor" backend with ROCm support is needed. Default might work or try specific options.
            # Check PyTorch documentation for torch.compile on ROCm.
            # Common modes: "default", "reduce-overhead", "max-autotune"
            # For some versions/platforms, `backend="inductor"` might be explicit.
            model = torch.compile(model, mode="default", dynamic=True) 
            if global_rank == 0: logging.info("torch.compile successfully applied.")
        except Exception as e:
            if global_rank == 0: logging.error(f"torch.compile failed: {e}. Proceeding without compile.")

    all_params = list(model.parameters())
    weight_params = [p for p in all_params if p.requires_grad and len(p.shape) > 1]
    bias_params = [p for p in all_params if p.requires_grad and len(p.shape) <= 1]

    if global_rank == 0:
        logging.info(f"Number of weight parameters for optimizer: {len(weight_params)}")
        logging.info(f"Number of bias parameters for optimizer: {len(bias_params)}")
        if not weight_params and all_params: logging.warning("No weight parameters found based on shape! Using all params in one group.")
    
    # 设置学习率
    if is_resume:
        learning_rate = config.get('resume_learning_rate', config['learning_rate'])
        if global_rank == 0:
            logging.info(f"Resume training with learning rate: {learning_rate}")
    else:
        learning_rate = config['learning_rate']
    
    optimizer_param_groups = []
    if not weight_params and not bias_params and all_params: # Only if both are empty but all_params is not
         optimizer_param_groups = [{'params': all_params, 'weight_decay': config.get('weight_decay', 1e-6)}]
    else:
        if weight_params:
            optimizer_param_groups.append({'params': weight_params, 'weight_decay': config.get('weight_decay', 1e-6)})
        if bias_params:
             optimizer_param_groups.append({'params': bias_params, 'weight_decay': 0.0})
        if not optimizer_param_groups and all_params: # Fallback if logic somehow misses params
            optimizer_param_groups = [{'params': all_params, 'weight_decay': config.get('weight_decay', 1e-6)}]

    optimizer = torch.optim.AdamW(optimizer_param_groups, lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=apply_amp, init_scale=2.**10, growth_factor=1.5, backoff_factor=0.5, growth_interval=100)

    # 初始化训练状态变量
    start_epoch = 0
    g_step = 0
    examples_processed_total = 0

    # 如果从checkpoint恢复训练
    if is_resume:
        try:
            resume_epoch, resume_step, resume_config = load_checkpoint_for_resume(
                config['resume_from_checkpoint'], model, optimizer, scaler, device, global_rank
            )
            start_epoch = resume_epoch
            g_step = resume_step
            
            # 根据步数估算已处理的样本数
            if config.get('batch_size', 0) > 0 and world_size > 0:
                examples_processed_total = g_step * config['batch_size'] * world_size
            
            if global_rank == 0:
                logging.info(f"Successfully resumed training from step {g_step}, epoch {start_epoch}")
                logging.info(f"Estimated examples processed so far: {examples_processed_total}")
        except Exception as e:
            if global_rank == 0:
                logging.error(f"Failed to load checkpoint: {e}")
                logging.info("Starting training from scratch...")
            is_resume = False

    if global_rank == 0 and wandb_run:
        wandb.watch(model, log="gradients", log_freq=config.get('wandb_watch_freq', 400))

    field_ids = None
    if 'field_id_path' in config and os.path.exists(config['field_id_path']):
        if global_rank == 0: logging.info(f"Loading field IDs from {config['field_id_path']}")
        field_ids = np.load(config['field_id_path']) 
        if global_rank == 0: logging.info(f"Field IDs loaded, shape: {field_ids.shape if field_ids is not None else 'None'}")

    rolling_loss_list = []
    global best_val_acc 
    best_val_acc = 0.0 

    # Initialize FLOPs tracking variables
    total_flops_accumulated = 0  # Total FLOPs since start
    last_log_flops = 0  # FLOPs at last log
    flops_since_last_log = 0  # Local accumulator for this rank
    flops_computed_once = False  # Flag to compute FLOPs only once per model config
    single_forward_flops = 0  # FLOPs for a single forward pass

    if global_rank == 0: os.makedirs(os.path.join("checkpoints", "ssl"), exist_ok=True)
    dist.barrier() 

    for epoch in range(start_epoch, config['epochs']):
        if global_rank == 0:
            aug1_dir = os.path.join(config['data_root'], 'aug1')
            aug2_dir = os.path.join(config['data_root'], 'aug2')
            skip_rust_cmd = False
            if epoch == start_epoch and not is_resume:  # 只在非恢复训练的第一个epoch检查
                aug1_s1_dir_check = os.path.join(aug1_dir, 's1')
                if os.path.isdir(aug1_s1_dir_check) and os.listdir(aug1_s1_dir_check): # Check if dir exists and is not empty
                    num_files_check = len(os.listdir(aug1_s1_dir_check))
                    if num_files_check * config.get('chunk_size', 1000000) >= config.get('total_samples',0): 
                        logging.info(f"Epoch {epoch}: Found existing data files potentially matching total_samples, skipping rust_cmd.")
                        skip_rust_cmd = True
            elif is_resume:
                # 恢复训练时跳过数据生成
                skip_rust_cmd = True
                if global_rank == 0:
                    logging.info(f"Resume training: skipping data generation for epoch {epoch}")
            
            if not skip_rust_cmd:
                if os.path.exists(aug1_dir): remove_dir(aug1_dir)
                if os.path.exists(aug2_dir): remove_dir(aug2_dir)
                logging.info(f"Epoch {epoch} starting. Generating new training data via rust_cmd...")
                try:
                    # Ensure rust_cmd is a list if not using shell=True for better arg handling, or ensure shell=True is safe.
                    # Using shell=True as in original.
                    subprocess.run(config['rust_cmd'], shell=True, check=True, executable="/bin/bash") 
                    logging.info(f"Data generation finished for epoch {epoch}.")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Rust data generation failed: {e}. Attempting to continue if old data exists...")
                except FileNotFoundError:
                    logging.error(f"Rust command not found. Check config['rust_cmd'] and PATH. Attempting to continue...")

        dist.barrier() 

        aug1_s2_dir_epoch = os.path.join(config['data_root'], 'aug1', 's2')
        s2_file_names_epoch = []
        if os.path.isdir(aug1_s2_dir_epoch):
            s2_file_names_epoch = os.listdir(aug1_s2_dir_epoch)
        else:
            if global_rank == 0: logging.error(f"Data directory {aug1_s2_dir_epoch} not found!")
            if epoch == start_epoch : raise FileNotFoundError(f"Critical: No data found at {aug1_s2_dir_epoch} on epoch {start_epoch}.")
        
        if not s2_file_names_epoch and epoch == start_epoch: # If still no files and it's the first epoch
             raise FileNotFoundError(f"Critical: No data files loaded for epoch {start_epoch} at {aug1_s2_dir_epoch}.")

        np.random.shuffle(s2_file_names_epoch) 
        total_files_epoch = len(s2_file_names_epoch)
        if global_rank == 0: logging.info(f"Epoch {epoch}: Total .npy files per sensor/aug type = {total_files_epoch}")

        chunk_batch_size = config.get('chunk_batch', 40)
        chunk_start_idx = 0
        
        epoch_start_time = time.time()
        epoch_examples_processed = 0
        
        last_log_time = time.time() # Initialize for the first log interval calculation
        last_log_examples = examples_processed_total # Initialize

        while chunk_start_idx < total_files_epoch:
            chunk_end_idx = min(chunk_start_idx + chunk_batch_size, total_files_epoch)
            current_chunk_files = s2_file_names_epoch[chunk_start_idx:chunk_end_idx]

            if not current_chunk_files: 
                chunk_start_idx = chunk_end_idx
                continue

            if global_rank == 0:
                logging.info(f"Epoch {epoch}, GlobalStep ~{g_step}, Loading chunk [{chunk_start_idx}:{chunk_end_idx}] with {len(current_chunk_files)} files...")

            aug1_s2_paths_chunk = [os.path.join(config['data_root'], 'aug1', 's2', fn) for fn in current_chunk_files]
            aug2_s2_paths_chunk = [os.path.join(config['data_root'], 'aug2', 's2', fn) for fn in current_chunk_files]
            aug1_s1_paths_chunk = [os.path.join(config['data_root'], 'aug1', 's1', fn) for fn in current_chunk_files]
            aug2_s1_paths_chunk = [os.path.join(config['data_root'], 'aug2', 's1', fn) for fn in current_chunk_files]

            chunk_dataset = ChunkDataset(aug1_s2_paths_chunk, aug2_s2_paths_chunk, aug1_s1_paths_chunk, aug2_s1_paths_chunk)
            train_sampler_chunk = DistributedSampler(chunk_dataset, shuffle=False, drop_last=True, rank=global_rank, num_replicas=world_size)
            
            train_loader_chunk = DataLoader(
                chunk_dataset, batch_size=config['batch_size'], sampler=train_sampler_chunk,
                num_workers=config['num_workers'], drop_last=True, pin_memory=torch.cuda.is_available(), 
                persistent_workers=True if config['num_workers'] > 0 else False 
            )
            
            train_sampler_chunk.set_epoch(epoch * 100 + (chunk_start_idx // chunk_batch_size if chunk_batch_size > 0 else 0) ) # Make epoch more unique for sampler

            if local_rank == 0: 
                logging.info(f"  ChunkDataset created with {len(chunk_dataset)} samples. DataLoader steps: {len(train_loader_chunk)}")

            model.train()
            
            for batch_data in train_loader_chunk:
                s2_aug1 = batch_data['s2_aug1'].to(device, non_blocking=True)
                s2_aug2 = batch_data['s2_aug2'].to(device, non_blocking=True)
                s1_aug1 = batch_data['s1_aug1'].to(device, non_blocking=True)
                s1_aug2 = batch_data['s1_aug2'].to(device, non_blocking=True)
                
                # Compute FLOPs for a single forward pass (only once)
                if not flops_computed_once and FlopCountAnalysis is not None:
                    try:
                        # Try to compute FLOPs for a single forward pass
                        # We'll use the first batch as a representative sample
                        with torch.no_grad():
                            single_forward_flops = compute_flops_single_pass(
                                model, (s2_aug1, s1_aug1), device
                            )
                        flops_computed_once = True
                        if global_rank == 0 and single_forward_flops > 0:
                            logging.info(f"Computed FLOPs for single forward pass: {single_forward_flops:,}")
                    except Exception as e:
                        if global_rank == 0:
                            logging.warning(f"Could not compute FLOPs: {e}")
                        flops_computed_once = True  # Don't try again
                
                # 调整学习率
                if is_resume:
                    current_lr_val = adjust_single_group_lr(
                        g_step, total_steps_approx, learning_rate,
                        config['warmup_ratio'], config.get('plateau_ratio', 0.7),
                        is_resume=True, 
                        resume_warmup_steps=config.get('resume_warmup_steps', 0)
                    )
                else:
                    current_lr_val = adjust_single_group_lr(
                        g_step, total_steps_approx, learning_rate,
                        config['warmup_ratio'], config.get('plateau_ratio', 0.7) 
                    )
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr_val
                
                optimizer.zero_grad(set_to_none=True) 

                qat_is_active_this_step = False
                if config.get('apply_qat_representation', False) and g_step >= config.get('qat_representation_start_step', float('inf')):
                    qat_is_active_this_step = True
                
                # Set the flag on the model (potentially FSDP wrapped)
                try:
                    model.qat_active_for_current_step = qat_is_active_this_step
                except AttributeError: # FSDP case
                    if hasattr(model, 'module'):
                        model.module.qat_active_for_current_step = qat_is_active_this_step
                    elif global_rank == 0:
                         logging.warning("Could not set qat_active_for_current_step on model or model.module.")

                if qat_is_active_this_step and global_rank == 0 and g_step == config.get('qat_representation_start_step'):
                    logging.info(f"Global Step {g_step}: QAT for representation is now ACTIVE.")
                
                with torch.cuda.amp.autocast(enabled=apply_amp):
                    proj_feats1, repr1_f32 = model(s2_aug1, s1_aug1) 
                    proj_feats2, repr2_f32 = model(s2_aug2, s1_aug2)
                    loss_main, bar_main, off_main = criterion(proj_feats1, proj_feats2)
                    
                    # Count FLOPs for the two forward passes (main)
                    if single_forward_flops > 0:
                        flops_since_last_log += 2 * single_forward_flops  # Two forward passes
                    
                    loss_mix = torch.tensor(0.0, device=device) 
                    if config.get('apply_mixup', False):
                        B = s2_aug1.size(0); idxs = torch.randperm(B, device=device)
                        alpha_dist = torch.distributions.Beta(config.get('beta_alpha',1.0), config.get('beta_beta',1.0))
                        alpha = alpha_dist.sample().to(device)
                        
                        # Ensure dtype handling for mixup, especially with AMP
                        s2_aug1_m, s2_aug2_m = (s2_aug1.float(), s2_aug2.float()) if apply_amp and s2_aug1.dtype == torch.float16 else (s2_aug1, s2_aug2)
                        s1_aug1_m, s1_aug2_m = (s1_aug1.float(), s1_aug2.float()) if apply_amp and s1_aug1.dtype == torch.float16 else (s1_aug1, s1_aug2)

                        y_m_s2 = alpha * s2_aug1_m + (1 - alpha) * s2_aug2_m[idxs, :]
                        y_m_s1 = alpha * s1_aug1_m + (1 - alpha) * s1_aug2_m[idxs, :]
                        
                        if apply_amp and s2_aug1.dtype == torch.float16:
                            y_m_s2, y_m_s1 = y_m_s2.half(), y_m_s1.half()

                        z_m, _ = model(y_m_s2, y_m_s1)
                        
                        # Count FLOPs for mixup forward pass
                        if single_forward_flops > 0:
                            flops_since_last_log += single_forward_flops  # One more forward pass for mixup
                        
                        z2_perm = torch.gather(proj_feats2, 0, idxs.unsqueeze(1).expand(-1, proj_feats2.size(1)))
                        
                        cc_m_a = compute_cross_correlation(z_m, proj_feats1)
                        cc_m_b = compute_cross_correlation(z_m, z2_perm) 
                        
                        cc_z1_z1 = compute_cross_correlation(proj_feats1, proj_feats1)
                        cc_z2p_z1 = compute_cross_correlation(z2_perm, proj_feats1) 
                        cc_z1_z2p = compute_cross_correlation(proj_feats1, z2_perm) 
                        cc_z2p_z2p = compute_cross_correlation(z2_perm, z2_perm) 

                        cc_m_a_gt = alpha * cc_z1_z1 + (1 - alpha) * cc_z2p_z1
                        cc_m_b_gt = alpha * cc_z1_z2p + (1 - alpha) * cc_z2p_z2p
                        
                        diff_a = (cc_m_a - cc_m_a_gt).pow(2).sum()
                        diff_b = (cc_m_b - cc_m_b_gt).pow(2).sum()
                        loss_mix = config.get('mixup_lambda',1.0) * config['barlow_lambda'] * (diff_a + diff_b)

                    total_loss = loss_main + loss_mix

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('clip_grad_norm', 2.0))
                scaler.step(optimizer)
                scaler.update()

                current_batch_size_effective = s2_aug1.size(0) * world_size 
                examples_processed_total += current_batch_size_effective
                epoch_examples_processed += current_batch_size_effective
                
                rolling_loss_list.append(loss_main.item())
                if len(rolling_loss_list) > 40: rolling_loss_list.pop(0) 
                avg_rolling_loss = sum(rolling_loss_list) / len(rolling_loss_list) if rolling_loss_list else 0.0

                if g_step % config['log_interval_steps'] == 0 and global_rank == 0:
                    current_time_log = time.time()
                    time_elapsed_log = current_time_log - last_log_time
                    examples_since_last_log = examples_processed_total - last_log_examples # This is global examples
                    
                    exps_sec = examples_since_last_log / time_elapsed_log if time_elapsed_log > 0 else 0.0
                    last_log_time = current_time_log
                    last_log_examples = examples_processed_total

                    # Gather FLOPs from all ranks
                    flops_tensor = torch.tensor([flops_since_last_log], dtype=torch.float64, device=device)
                    dist.all_reduce(flops_tensor, op=dist.ReduceOp.SUM)
                    global_flops_since_last_log = flops_tensor.item()
                    
                    # Update total accumulated FLOPs
                    total_flops_accumulated += global_flops_since_last_log
                    
                    # Calculate TFLOPs (1e12 FLOPs)
                    tflops_since_last_log = global_flops_since_last_log / 1e12
                    total_tflops_accumulated = total_flops_accumulated / 1e12
                    
                    # Reset local FLOPs counter
                    flops_since_last_log = 0

                    erank_z, erank_repr = 0.0, 0.0 
                    try:
                        if proj_feats1 is not None: erank_z = rankme(proj_feats1.detach()).item()
                        if repr1_f32 is not None: erank_repr = rankme(repr1_f32.detach()).item()
                    except Exception as e_rank:
                        logging.warning(f"RankMe computation failed: {e_rank}")

                    logging.info(
                        f"[Epoch={epoch}/{config['epochs']-1}, Step={g_step}/{total_steps_approx}] "
                        f"Loss={loss_main.item():.3f} (Mix:{loss_mix.item() if isinstance(loss_mix, torch.Tensor) else loss_mix:.3f}, AvgRoll:{avg_rolling_loss:.3f}) "
                        f"LR={current_lr_val:.5f}, GLB_Batch={current_batch_size_effective}, Ex/s={exps_sec:.1f} "
                        f"Rank(z_proj)={erank_z:.3f}, Rank(repr_f32)={erank_repr:.3f} "
                        f"TFLOPs_interval={tflops_since_last_log:.3f}, TFLOPs_total={total_tflops_accumulated:.3f}"
                    )
                    if wandb_run:
                        wandb_log_dict = {
                            "epoch": epoch, "global_step": g_step,
                            "loss_main": loss_main.item(), "loss_mix": loss_mix.item() if isinstance(loss_mix, torch.Tensor) else loss_mix,
                            "total_loss": total_loss.item(), "avg_rolling_loss": avg_rolling_loss,
                            "learning_rate": current_lr_val, "examples_per_second_global": exps_sec,
                            "rank_z_projection": erank_z, "rank_fused_representation_f32": erank_repr,
                            "grad_scaler_scale": scaler.get_scale() if apply_amp else -1.0, # Log scaler state
                            "tflops_since_last_log": tflops_since_last_log,
                            "tflops_accumulated": total_tflops_accumulated,
                        }
                        # 修复：使用global step作为wandb的step参数
                        wandb.log(wandb_log_dict, step=g_step) 
                elif g_step % config['log_interval_steps'] != 0:
                    # For non-rank0 or non-log steps, just reset the local counter after all_reduce
                    if g_step % config['log_interval_steps'] == 0:
                        # All ranks participate in all_reduce
                        flops_tensor = torch.tensor([flops_since_last_log], dtype=torch.float64, device=device)
                        dist.all_reduce(flops_tensor, op=dist.ReduceOp.SUM)
                        flops_since_last_log = 0

                if config.get('val_interval_steps',0) > 0 and g_step > 0 and g_step % config['val_interval_steps'] == 0:
                    torch.cuda.empty_cache(); gc.collect()
                    dist.barrier() 

                    checkpoint_dir_val = os.path.join("checkpoints", "ssl") 
                    checkpoint_path_val = os.path.join(checkpoint_dir_val, f"checkpoint_{timestamp}.pt")
                    
                    saved_model_state_dict_rank0 = save_fsdp_model(model, optimizer, config, checkpoint_path_val, g_step, epoch, global_rank)
                    
                    if global_rank == 0:
                        logging.info(f"Starting validation at global_step {g_step} using checkpoint {checkpoint_path_val}...")
                        val_config_keys = ['val_s2_bands_file_path', 'val_s2_masks_file_path', 'val_s2_doy_file_path',
                                           'val_s1_asc_bands_file_path', 'val_s1_asc_doy_file_path', 
                                           'val_s1_desc_bands_file_path', 'val_s1_desc_doy_file_path', 'val_labels_path']
                        if not all(config.get(k) for k in val_config_keys):
                            logging.warning("Validation data paths missing in config. Skipping validation.")
                        else:
                            try:
                                from datasets.ssl_dataset import AustrianCropValidation 
                                val_dataset = AustrianCropValidation( # Ensure this class is correctly defined and imported
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
                                    min_valid_timesteps=config.get('val_min_valid_timesteps', 0), 
                                    standardize=config.get('val_standardize', True)
                                )
                                val_loader_eval = DataLoader(val_dataset, batch_size=config.get('val_batch_size', 512), 
                                                             shuffle=False, num_workers=config.get('val_num_workers',0)) 

                                eval_model_val = create_evaluation_model(config, device) 
                                if saved_model_state_dict_rank0:
                                    clean_state_dict_val = {}
                                    for k_val, v_val in saved_model_state_dict_rank0.items():
                                        nk_val = k_val
                                        if k_val.startswith('_fsdp_wrapped_module.'): nk_val = k_val[len('_fsdp_wrapped_module.'):]
                                        elif k_val.startswith('_orig_mod.'): nk_val = k_val[len('_orig_mod.'):]
                                        clean_state_dict_val[nk_val] = v_val
                                    eval_model_val.load_state_dict(clean_state_dict_val, strict=True)
                                else: 
                                    eval_model_val = load_model_from_checkpoint(checkpoint_path_val, config, device)
                                
                                eval_model_val.eval()

                                val_acc, val_f1, val_cm, val_cr = linear_probe_evaluate(
                                    eval_model_val, val_loader_eval,
                                    field_ids=field_ids, 
                                    field_data_path=config.get('fielddata_csv_path'),
                                    training_ratio=config.get('training_ratio', 0.3),
                                    val_test_split_ratio=config.get('val_test_split_ratio', 1/7.0),
                                    classifier_type=config.get('classifier_type', 'lr'),
                                    num_inference=config.get('num_inference', 1),
                                    device=device, apply_amp=apply_amp, 
                                    apply_qat_representation_for_eval=config.get('apply_qat_representation', False),
                                    qat_representation_bits_for_eval=config.get('qat_representation_bits', 8)
                                )
                                logging.info(f"Validation at Step {g_step}: Accuracy={val_acc:.4f}, F1={val_f1:.4f}")
                                if wandb_run:
                                    wandb.log({
                                        "val_accuracy": val_acc, "val_f1_weighted": val_f1,
                                        "val_classification_report_html": wandb.Html(f"<pre>{val_cr}</pre>"),
                                    }, step=g_step) 
                                    try: # Plot confusion matrix
                                        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                                        disp = ConfusionMatrixDisplay(confusion_matrix=val_cm) # Assumes val_cm is from sklearn
                                        disp.plot(ax=ax_cm, cmap='Blues', values_format='d')
                                        ax_cm.set_title(f'Confusion Matrix - Step {g_step}')
                                        plt.tight_layout()
                                        wandb.log({"val_confusion_matrix_plot": wandb.Image(fig_cm)}, step=g_step)
                                        plt.close(fig_cm)
                                    except Exception as e_cm_plot:
                                        logging.warning(f"Could not plot confusion matrix for W&B: {e_cm_plot}")

                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    logging.info(f"New best validation accuracy: {best_val_acc:.4f}. Saving best model.")
                                    best_ckpt_path_val = os.path.join(checkpoint_dir_val, f"best_model_fsdp_{timestamp}.pt")
                                    torch.save({
                                        'epoch': epoch, 'step': g_step,
                                        'model_state_dict': eval_model_val.state_dict(), 
                                        'best_val_acc': best_val_acc, 'config': config
                                    }, best_ckpt_path_val)

                                del eval_model_val, val_dataset, val_loader_eval
                            except ImportError:
                                logging.error("Could not import AustrianCropValidation. Ensure datasets/ssl_dataset.py is correct and in PYTHONPATH.")
                            except Exception as e_val:
                                logging.error(f"Validation error at step {g_step}: {e_val}", exc_info=True)
                            finally:
                                torch.cuda.empty_cache(); gc.collect()
                    
                    dist.barrier() 
                    model.train() 

                g_step += 1 
            
            try:
                if hasattr(train_loader_chunk, '_iterator') and train_loader_chunk._iterator is not None:
                    if hasattr(train_loader_chunk._iterator, '_shutdown_workers'):
                         train_loader_chunk._iterator._shutdown_workers()
            except Exception as e_dl_cleanup:
                if global_rank == 0: logging.warning(f"Minor error during DataLoader cleanup: {e_dl_cleanup}")
            del train_loader_chunk, train_sampler_chunk, chunk_dataset
            gc.collect()
            if global_rank == 0: logging.info(f"Finished processing chunk. Current global step: {g_step}")

            chunk_start_idx = chunk_end_idx 

        epoch_duration = time.time() - epoch_start_time
        if global_rank == 0:
            logging.info(f"Epoch {epoch} finished. Duration: {epoch_duration:.2f}s. Total examples in epoch (global): {epoch_examples_processed}")
            logging.info(f"GPU memory at end of epoch {epoch}: {get_gpu_memory_usage():.2f} MB")
            if wandb_run: wandb.log({"epoch_duration_sec": epoch_duration, "epoch_examples_global": epoch_examples_processed}, step=g_step)

    if global_rank == 0:
        logging.info("Training completed.")
        final_checkpoint_path = os.path.join("checkpoints", "ssl", f"checkpoint_{timestamp}.pt")
        save_fsdp_model(model, optimizer, config, final_checkpoint_path, g_step, config['epochs']-1, global_rank, is_final=True)
        logging.info(f"Final model saved to {final_checkpoint_path}")
        if wandb_run:
            wandb.summary['best_validation_accuracy'] = best_val_acc
            # Check if model artifact for best model was saved and log it
            best_model_path_check = os.path.join("checkpoints", "ssl", f"best_model_fsdp_{timestamp}.pt")
            if os.path.exists(best_model_path_check):
                best_model_artifact = wandb.Artifact(f"best_model_{timestamp}", type="model", metadata={"best_val_acc": best_val_acc, "step": g_step})
                best_model_artifact.add_file(best_model_path_check)
                wandb.log_artifact(best_model_artifact)
            wandb_run.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    # For ConfusionMatrixDisplay
    try:
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        if dist.get_rank() == 0: # Print only on one rank
             print("Warning: scikit-learn ConfusionMatrixDisplay not available. Confusion matrix plots to W&B might fail.")
             print("Consider: pip install scikit-learn>=0.22 (or check your sklearn version)")
        ConfusionMatrixDisplay = None # Fallback

    main()