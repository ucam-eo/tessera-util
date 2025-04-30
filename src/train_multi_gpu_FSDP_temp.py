#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 运行方式:
#   torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 src/train_multi_gpu_FSDP_temp.py

import os
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import math
import time
import gc
import argparse
import logging
import subprocess
from datetime import datetime
from contextlib import nullcontext
import json

import numpy as np
import torch
# 这行可根据需要设定矩阵乘法的精度
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

import torch.amp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # 避免 torch.compile 的错误

# FSDP 相关导入
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

# 直接使用wandb进行可视化
import wandb
import matplotlib.pyplot as plt

# ============== 自定义模块 / 函数 ===============
from models.modules import TransformerEncoder, ProjectionHead, SpectralTemporalTransformer
from models.ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate, LARS
from utils.metrics import linear_probe_evaluate, rankme, rf_probe_evaluate
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr

import torch.nn.attention as attn

# FSDP辅助函数：获取GPU内存使用情况
def get_gpu_memory_usage():
    """返回当前GPU的内存使用情况（单位：MB）"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

# FSDP辅助函数：打印模型信息
def print_model_info(model, name="Model"):
    """打印模型的结构和参数信息"""
    # 获取总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 获取模型结构树（最多20层递归）
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

##########################################################################
# 1) 一个用于加载"部分文件"的Dataset类
##########################################################################
class ChunkDataset(Dataset):
    """
    给定一批文件(aug1_s2_files, aug2_s2_files, aug1_s1_files, aug2_s1_files)，
    逐个加载到CPU内存，然后合并成一个大的list: all_samples = [
      (s2_aug1[i], s2_aug2[i], s1_aug1[i], s1_aug2[i]),
      ...
    ]
    这样就可以在 __getitem__ 里根据索引拿到一条数据。

    注意：
      - 如果 chunk_batch 很大，这也可能一次装太多数据。可再做更细粒度拆分。
      - 假设 Rust 生成的文件都已经洗好牌，所以不再需要 shuffle。
    """
    def __init__(self, aug1_s2_files, aug2_s2_files, aug1_s1_files, aug2_s1_files):
        super().__init__()
        self.all_samples = []

        # 读取这批文件
        for idx_file in range(len(aug1_s2_files)):
            file_aug1_s2 = aug1_s2_files[idx_file]
            file_aug2_s2 = aug2_s2_files[idx_file]
            file_aug1_s1 = aug1_s1_files[idx_file]
            file_aug2_s1 = aug2_s1_files[idx_file]

            arr_aug1_s2 = np.load(file_aug1_s2)
            arr_aug2_s2 = np.load(file_aug2_s2)
            arr_aug1_s1 = np.load(file_aug1_s1)
            arr_aug2_s1 = np.load(file_aug2_s1)

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
        # 再次进行洗牌
        np.random.shuffle(self.all_samples)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        # 读出4块数据并转换成Tensor
        s2_aug1_np, s2_aug2_np, s1_aug1_np, s1_aug2_np = self.all_samples[index]
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

##########################################################################
# 2) 主训练脚本 (FSDP + chunk-based loading)
##########################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training (FSDP + chunk-based loading)")
    parser.add_argument('--config', type=str, default="configs/ssl_config_temp.py",
                        help="Path to config file (e.g. configs/ssl_config_temp.py)")
    return parser.parse_args()

def main():
    # 设置初始化种子
    torch.manual_seed(3407)
    np.random.seed(3407)

    ########################################################################
    # 0) 初始化分布式环境 (使用 PyTorch 内置的 torchrun + FSDP)
    ########################################################################
    dist.init_process_group(backend='nccl')  # 在 AMD ROCm 上通常也使用 NCCL (其实底层是 RCCL)

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 由 torchrun 传进来的本地进程索引
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 设置当前设备
    torch.cuda.set_device(local_rank)

    # 设置日志级别：rank=0 输出info日志，其它rank仅输出warning
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # 屏蔽多余rank的W&B日志
        os.environ['WANDB_MODE'] = 'disabled'

    ########################################################################
    # 1) 解析命令行 & 读取配置
    ########################################################################
    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    # 如果需要AMP
    apply_amp = config.get('apply_amp', False)
    if local_rank == 0:
        logging.info(f"apply_amp = {apply_amp}")

    # 使用 CUDA 设备（AMD ROCm 下同样用 "cuda"）
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        logging.info(f"Running on {world_size} GPU(s). LocalRank={local_rank}, device={device}")
        logging.info(f"Initial GPU memory usage: {get_gpu_memory_usage():.2f} MB")
    
    if local_rank == 0:
        os.environ["WANDB_API_KEY"] = "b03eca52bd30c1fa9bf185ae3ee91d9276f2f92a"

    # 如果用户想禁用W&B获取git信息，可添加环境变量，避免 "dubious ownership" 警告
    if config.get("disable_wandb_git", False):
        os.environ['WANDB_DISABLE_GIT'] = 'true'

    # 读取 chunk_batch
    chunk_batch = config.get('chunk_batch', 40)
    if local_rank == 0:
        logging.info(f"chunk_batch = {chunk_batch}")

    ########################################################################
    # 2) 准备W&B
    ########################################################################
    run_name = f"FSDP_BT_Iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if local_rank == 0:
        # wandb_run = wandb.init(project="btfm-iterable", name=run_name, config=config)
        wandb_run = wandb.init(project="btfm-iterable-temp", name=run_name, config=config)
    else:
        wandb_run = None

    # 计算总训练步数（可能并不精准，因为我们分块加载）
    total_steps = config['epochs'] * config['total_samples'] // config['batch_size'] // world_size
    if local_rank == 0:
        logging.info(f"Total steps per rank (approx) = {total_steps}")

    ########################################################################
    # 3) 构建模型
    ########################################################################
    # 这些超参数改从 config 中读取
    s2_num_heads = config['s2_num_heads']
    s2_num_layers = config['s2_num_layers']
    s2_dim_feedforward = config['s2_dim_feedforward']
    s1_num_heads = config['s1_num_heads']
    s1_num_layers = config['s1_num_layers']
    s1_dim_feedforward = config['s1_dim_feedforward']

    # 也可记录到 wandb config（若需要）
    if local_rank == 0:
        wandb.config.update({
            "s2_num_heads": s2_num_heads,
            "s2_num_layers": s2_num_layers,
            "s2_dim_feedforward": s2_dim_feedforward,
            "s1_num_heads": s1_num_heads,
            "s1_num_layers": s1_num_layers,
            "s1_dim_feedforward": s1_dim_feedforward
        })

    # 构建模型组件
    s2_enc = TransformerEncoder(
        band_num=10,
        latent_dim=config['latent_dim'],
        nhead=s2_num_heads,
        num_encoder_layers=s2_num_layers,
        dim_feedforward=s2_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s2'],
    ).to(device)

    s1_enc = TransformerEncoder(
        band_num=2,
        latent_dim=config['latent_dim'],
        nhead=s1_num_heads,
        num_encoder_layers=s1_num_layers,
        dim_feedforward=s1_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s1'],
    ).to(device)

    if config['fusion_method'] == 'concat':
        proj_in_dim = config['latent_dim']
    else:
        proj_in_dim = config['latent_dim']

    projector = ProjectionHead(
        proj_in_dim,
        config['projector_hidden_dim'],
        config['projector_out_dim']
    ).to(device)

    # 将模型组件组合成完整模型
    model = MultimodalBTModel(
        s2_enc, s1_enc, projector,
        fusion_method=config['fusion_method'],
        return_repr=True,
        latent_dim=config['latent_dim']
    ).to(device)

    # 打印 FSDP 包装前的模型信息
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Before FSDP wrapping - Total model parameters: {total_params:,}")
        logging.info(f"Model structure before FSDP wrapping:\n{str(model)[:2000]}...")
        logging.info(f"GPU memory usage before FSDP: {get_gpu_memory_usage():.2f} MB")

    # 设置损失函数
    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])

    # 定义自动包装策略 - 这将确定哪些层应该自动被FSDP包装
    # 获取 transformer 层类型（你需要根据你的模型结构调整这些）
    module_classes_to_wrap = []
    # 如果 TransformerEncoder 中有子层如 TransformerEncoderLayer，可以添加它们
    # 我们不确定你的 TransformerEncoder 的内部结构，所以默认包装整个 TransformerEncoder
    module_classes_to_wrap.append(TransformerEncoder)

    # 定义 FSDP 混合精度配置 (如果使用 AMP)
    if apply_amp:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            # 保持主要计算在 float16, 但累加在 float32 避免数值问题
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float16,
        )
    else:
        mixed_precision_policy = None

    # 为PyTorch 2.4，使用size_based_auto_wrap_policy而不是transformer_auto_wrap_policy
    # 或者创建自定义包装策略
    def custom_auto_wrap_policy(module, recurse=True, **kwargs):
        """自定义FSDP包装策略，包装指定类型的模块"""
        if isinstance(module, tuple(module_classes_to_wrap)):
            return True
        return False

    # 定义 FSDP 的配置参数
    fsdp_config = {
        "auto_wrap_policy": custom_auto_wrap_policy,  # 使用自定义包装策略
        "sharding_strategy": ShardingStrategy.FULL_SHARD,  # 完全分片策略
        "device_id": local_rank,
        "sync_module_states": True,  # 确保初始模型状态在所有进程间同步
        "forward_prefetch": True,  # 预取前向传播的参数
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,  # 预取反向传播的参数
        "cpu_offload": CPUOffload(offload_params=False),  # 如果内存紧张，可以设为 True
        "use_orig_params": True,  # 必须设置为True才能与torch.compile兼容
    }

    # 添加混合精度配置（如果启用）
    if mixed_precision_policy:
        fsdp_config["mixed_precision"] = mixed_precision_policy

    # 打印FSDP配置信息
    if local_rank == 0:
        logging.info("FSDP Configuration:")
        config_str = json.dumps({k: str(v) for k, v in fsdp_config.items()}, indent=2)
        logging.info(config_str)

    # 使用 FSDP 包装模型
    with (attn.sdpa_kernel(attn.SDPBackend.MATH) if config.get('use_torch_compile', False) else nullcontext()):
        # 使用 FSDP 包装模型
        model = FSDP(model, **fsdp_config)
        
        # 打印包装后的模型结构（只在rank=0上打印）
        if local_rank == 0:
            logging.info(f"Model structure after FSDP wrapping:\n{str(model)[:5000]}...")
            
            # 获取每个rank上的参数数量
            fsdp_params = sum(p.numel() for p in model.parameters())
            logging.info(f"After FSDP wrapping - Parameters on rank {local_rank}: {fsdp_params:,}")
            
            # 打印内存使用情况
            logging.info(f"GPU memory usage after FSDP: {get_gpu_memory_usage():.2f} MB")
            
            # 打印参数统计
            total_params_before = total_params
            sharded_ratio = fsdp_params / total_params_before
            logging.info(f"Sharding ratio: {sharded_ratio:.4f} (lower means better distribution)")
            
            # 根据world_size计算理想的分片比例
            ideal_ratio = 1.0 / world_size
            logging.info(f"Ideal sharding ratio: {ideal_ratio:.4f}")
        
        if config.get('use_torch_compile', False):
            model = torch.compile(model, mode="default")
            # model = torch.compile(model, mode="max-autotune")
            if local_rank == 0:
                logging.info("Using torch.compile to optimize the model...")

        # 分别收集所有GPU上的参数数量（用于确认分片是否均匀）
        local_param_count = sum(p.numel() for p in model.parameters())
        all_ranks_param_counts = [torch.tensor([0], device=device) for _ in range(world_size)]
        local_count_tensor = torch.tensor([local_param_count], device=device)
        
        # 收集所有rank的参数数量
        dist.all_gather(all_ranks_param_counts, local_count_tensor)
        
        if local_rank == 0:
            rank_counts = [int(t.item()) for t in all_ranks_param_counts]
            logging.info(f"Parameter counts across all ranks: {rank_counts}")
            
            # 修正：允许参数分布有小差异（例如少于0.1%的差异）
            max_count = max(rank_counts)
            min_count = min(rank_counts)
            diff_percentage = (max_count - min_count) / max_count * 100
            
            if diff_percentage < 0.1:  # 允许0.1%的差异
                logging.info(f"✅ Parameters are evenly distributed across GPUs (差异小于0.1%: {diff_percentage:.4f}%)")
            else:
                logging.info(f"⚠️ Parameters are NOT evenly distributed across GPUs (差异: {diff_percentage:.4f}%)")

        # 创建优化器参数组的逻辑修正
        # 在FSDP中，我们直接使用模型的所有参数，然后根据形状分为权重和偏置组
        all_params = list(model.parameters())
        
        # 尝试按照原始参数的维度分组（weight_dim_fn和bias_dim_fn函数是FSDP提供的）
        weight_params = []
        bias_params = []
        
        # 直接检查参数形状，不使用参数名称
        for param in all_params:
            if param.requires_grad:
                # 先检查param的shape属性而不是ndim
                if len(param.shape) > 1:
                    weight_params.append(param)
                else:
                    bias_params.append(param)
        
        if local_rank == 0:
            logging.info(f"Number of weight parameters: {len(weight_params)}")
            logging.info(f"Number of bias parameters: {len(bias_params)}")
            
            # 如果weight_params为空，说明分类逻辑有问题，直接使用所有参数
            if len(weight_params) == 0:
                logging.warning("No weight parameters found! Using all parameters in one group.")
                weight_params = all_params
                bias_params = []
        
        # 如果weight_params仍然为空，使用所有参数为一个组
        if len(weight_params) == 0:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'], 
                weight_decay=1e-6
            )
        else:
            # 创建具有两个参数组的优化器
            optimizer = torch.optim.AdamW(
                [{'params': weight_params}, {'params': bias_params}],
                lr=config['learning_rate'], 
                weight_decay=1e-6
            )

        # 使用新的 GradScaler 写法（如果需要 AMP）
        scaler = torch.amp.GradScaler("cuda") if apply_amp else None
        
        if local_rank == 0 and wandb_run is not None:
            # 注意：FSDP 下 wandb.watch 可能不会记录所有梯度和参数
            wandb.watch(model, log="gradients", log_freq=400)

        ########################################################################
        # 4) 训练循环 (chunk-based)
        ########################################################################
        def _train_one_chunk(
            chunk_files,
            epoch,
            step,
            examples,
            last_time,
            last_examples,
            rolling_loss
        ):
            """
            针对给定的一批文件(chunk_files)，构建ChunkDataset和DataLoader，
            并在这一批数据上训练若干step。返回更新后的 step / examples 等。
            """
            # 构造4组文件路径列表
            aug1_s2_dir = os.path.join(config['data_root'], 'aug1', 's2')
            aug2_s2_dir = os.path.join(config['data_root'], 'aug2', 's2')
            aug1_s1_dir = os.path.join(config['data_root'], 'aug1', 's1')
            aug2_s1_dir = os.path.join(config['data_root'], 'aug2', 's1')

            aug1_s2_paths = [os.path.join(aug1_s2_dir, fn) for fn in chunk_files]
            aug2_s2_paths = [os.path.join(aug2_s2_dir, fn) for fn in chunk_files]
            aug1_s1_paths = [os.path.join(aug1_s1_dir, fn) for fn in chunk_files]
            aug2_s1_paths = [os.path.join(aug2_s1_dir, fn) for fn in chunk_files]

            # 构造 chunk 数据集
            chunk_dataset = ChunkDataset(
                aug1_s2_files=aug1_s2_paths,
                aug2_s2_files=aug2_s2_paths,
                aug1_s1_files=aug1_s1_paths,
                aug2_s1_files=aug2_s1_paths
            )
            train_sampler = DistributedSampler(chunk_dataset, shuffle=False, drop_last=True)

            train_loader = DataLoader(
                chunk_dataset,
                batch_size=config['batch_size'],
                sampler=train_sampler,
                num_workers=config['num_workers'],
                drop_last=True,
                pin_memory=False,
                persistent_workers=False
            )

            train_sampler.set_epoch(epoch)

            if local_rank == 0:
                logging.info(
                    f"   => This chunk has {len(chunk_dataset)} samples total, "
                    f"steps in dataloader = {len(train_loader)}."
                )

            model.train()

            for batch_data in train_loader:
                s2_aug1 = batch_data['s2_aug1'].to(device, non_blocking=True)
                s2_aug2 = batch_data['s2_aug2'].to(device, non_blocking=True)
                s1_aug1 = batch_data['s1_aug1'].to(device, non_blocking=True)
                s1_aug2 = batch_data['s1_aug2'].to(device, non_blocking=True)

                # 学习率调度 - 根据实际optimizer的参数组数量调整代码
                if len(optimizer.param_groups) > 1:
                    # 原始代码需要两个参数组
                    adjust_learning_rate(
                        optimizer,
                        step,
                        total_steps,
                        config['learning_rate'],
                        config['warmup_ratio'],
                        config['plateau_ratio']
                    )
                else:
                    # 仅有一个参数组的调度
                    current_lr = adjust_single_group_lr(
                        step, 
                        total_steps,
                        config['learning_rate'],
                        config['warmup_ratio'],
                        config['plateau_ratio']
                    )
                    optimizer.param_groups[0]['lr'] = current_lr

                optimizer.zero_grad()

                if apply_amp:
                    # 使用新的autocast写法
                    with torch.amp.autocast("cuda"):
                        z1, repr1 = model(s2_aug1, s1_aug1)
                        z2, repr2 = model(s2_aug2, s1_aug2)
                        loss_main, bar_main, off_main = criterion(z1, z2)

                        # 如果需要 mixup
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

                            z_m, _ = model(y_m_s2, y_m_s1)

                            # 通过 gather 替换高级索引 z2[idxs]
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

                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
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
                        alpha = torch.distributions.Beta(
                            config['beta_alpha'],
                            config['beta_beta']
                        ).sample().to(device)
                        y_m_s2 = alpha * s2_aug1 + (1 - alpha) * s2_aug2[idxs, :]
                        y_m_s1 = alpha * s1_aug1 + (1 - alpha) * s1_aug2[idxs, :]

                        z_m, _ = model(y_m_s2, y_m_s1)

                        # 通过 gather 替换高级索引 z2[idxs]
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

                # 日志打印
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
                    
                    # 记录当前GPU内存使用情况
                    # current_gpu_mem = get_gpu_memory_usage()

                    logging.info(
                        f"[Epoch={epoch}, Step={step}] "
                        f"Loss={loss_main.item():.2f}, MixLoss={loss_mix:.2f}, AvgLoss={avg_loss:.2f}, "
                        f"LR={current_lr:.4f}, batchsize={batch_sz}, Examples/sec={exps:.2f}, "
                        f"Rank(z)={erank_z:.4f}, Rank(repr)={erank_repr:.4f}, "
                        # f"GPU_Mem={current_gpu_mem:.2f}MB"
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
                        # "gpu_memory_usage_mb": current_gpu_mem
                    }
                    
                    progress_percentage = (step / total_steps) * 100
                    # Create a custom HTML progress bar
                    progress_bar_html = f"""
                    <div style="border: 1px solid #ccc; width: 100%; height: 20px; border-radius: 3px; overflow: hidden;">
                    <div style="background-color: #4CAF50; width: {progress_percentage}%; height: 100%; text-align: center; line-height: 20px; color: white;">
                        {progress_percentage:.1f}%
                    </div>
                    </div>
                    """
                    wandb_dict["training_progress"] = wandb.Html(progress_bar_html)

                    wandb.log(wandb_dict, step=step)

                # 验证 + 保存best
                if (config['val_interval_steps'] > 0) and (step > 0) and \
                (step % config['val_interval_steps'] == 0) and (local_rank == 0):
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
                        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)

                        model.eval()
                        # 线性探针：返回 accuracy, F1, 以及混淆矩阵
                        val_acc, val_f1, val_cm, val_report = linear_probe_evaluate(model, val_loader, device=device)
                        wandb.log({"val_acc": val_acc, "val_f1": val_f1}, step=step)
                        logging.info(f"Validation at step {step}: val_acc={val_acc:.4f}, F1 Score={val_f1:.4f}")
                        wandb.log({"classification_report_linear": wandb.Html("<pre>" + val_report + "</pre>")}, step=step)
                        logging.info(f"Classification Report:\n{val_report}")
                        
                        # 绘制混淆矩阵
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(val_cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax.figure.colorbar(im, ax=ax)
                        ax.set(
                            xticks=range(val_cm.shape[1]),
                            yticks=range(val_cm.shape[0]),
                            xticklabels=range(val_cm.shape[1]),
                            yticklabels=range(val_cm.shape[0]),
                            title='Confusion Matrix',
                            ylabel='True label',
                            xlabel='Predicted label'
                        )
                        thresh = val_cm.max() / 2.
                        for i in range(val_cm.shape[0]):
                            for j in range(val_cm.shape[1]):
                                ax.text(j, i, format(val_cm[i, j], 'd'),
                                        ha="center", va="center",
                                        color="white" if val_cm[i, j] > thresh else "black")
                        fig.tight_layout()
                        wandb.log({"val_confusion_matrix": wandb.Image(fig)}, step=step)
                        plt.close(fig)

                        global best_val_acc
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            # 使用 FSDP 保存模型 - 使用 state_dict_type 确保正确保存
                            full_state_dict_config = FullStateDictConfig(
                                offload_to_cpu=True,
                                rank0_only=True
                            )
                            
                            # 只有 rank=0 保存完整模型
                            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                                state_dict = model.state_dict()
                                if local_rank == 0:
                                    logging.info(f"Saving best model with val_acc={val_acc:.4f} to {best_ckpt_path}")
                                    torch.save({
                                        'epoch': epoch,
                                        'step': step,
                                        'model_state_dict': state_dict,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'best_val_acc': best_val_acc,
                                    }, best_ckpt_path)
                        
                        model.train()

                step += 1  # 全局step

            # 清理
            try:
                train_loader._iterator._shutdown_workers()
            except:
                pass
            del train_loader
            del train_sampler
            del chunk_dataset
            gc.collect()

            return step, examples, last_time, last_examples, rolling_loss

        ########################################################################
        # 一些全局变量
        ########################################################################
        step = 0
        examples = 0
        last_time = time.time()
        last_examples = 0
        rolling_loss = []
        global best_val_acc
        best_val_acc = 0.0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_fsdp_{timestamp}.pt")

        # 确保checkpoints目录存在
        if local_rank == 0:
            os.makedirs(os.path.join("checkpoints", "ssl"), exist_ok=True)

        ########################################################################
        # ============== 开始多个 EPOCH ==============
        ########################################################################
        for epoch in range(config['epochs']):
            # 如果你需要在每个epoch调用rust脚本生成数据，可以在rank=0上执行，然后dist.barrier()等待
            if local_rank == 0:
                # 如果是第一个epoch，检查一下有没有上次run留下来的数据，有的话就不需要在这个epoch重新生成了
                aug1_dir = os.path.join(config['data_root'], 'aug1')
                aug2_dir = os.path.join(config['data_root'], 'aug2')
                if epoch == 0:
                    # 检查是否需要跳过
                    aug1_s1_dir = os.path.join(aug1_dir, 's1')
                    if os.path.isdir(aug1_s1_dir):
                        num_files = len(os.listdir(aug1_s1_dir))
                    else:
                        num_files = 0

                    if num_files * 1000000 == config['total_samples']:
                        logging.info(f"Epoch {epoch}: Found existing data files, skipping rust_cmd...")
                    else:
                        remove_dir(aug1_dir)
                        remove_dir(aug2_dir)
                        logging.info(f"Epoch {epoch} started. Generating new training data via rust_cmd...")
                        subprocess.run(config['rust_cmd'], shell=True, check=True)
                        logging.info(f"Data generation finished for epoch {epoch}.")
                else:
                    # 每个epoch都重新生成数据
                    remove_dir(aug1_dir)
                    remove_dir(aug2_dir)
                    logging.info(f"Epoch {epoch} started. Generating new training data via rust_cmd...")
                    subprocess.run(config['rust_cmd'], shell=True, check=True)
                    logging.info(f"Data generation finished for epoch {epoch}.")

            dist.barrier()

            # 收集当前数据文件名
            aug1_s2_dir = os.path.join(config['data_root'], 'aug1', 's2')
            s2_file_names = os.listdir(aug1_s2_dir)
            np.random.shuffle(s2_file_names)
            total_files = len(s2_file_names)
            if local_rank == 0:
                logging.info(f"Epoch {epoch}: total new files = {total_files} in each folder")

            # 分块加载训练
            chunk_start = 0
            while chunk_start < total_files:
                chunk_end = min(chunk_start + chunk_batch, total_files)
                chunk_files = s2_file_names[chunk_start:chunk_end]
                if local_rank == 0:
                    logging.info(f"Epoch {epoch}, chunk [{chunk_start}:{chunk_end}], loading {len(chunk_files)} files...")

                step, examples, last_time, last_examples, rolling_loss = _train_one_chunk(
                    chunk_files, epoch, step, examples, last_time, last_examples, rolling_loss
                )
                chunk_start = chunk_end

            if local_rank == 0:
                logging.info(f"Epoch {epoch} finished, current step = {step}")
                # 记录每个epoch结束时的GPU内存使用
                logging.info(f"GPU memory at end of epoch {epoch}: {get_gpu_memory_usage():.2f} MB")

        if local_rank == 0:
            logging.info("Training completed.")
            if wandb_run is not None:
                wandb_run.finish()

        dist.destroy_process_group()

# 添加单组学习率调度函数
def adjust_single_group_lr(iter_count, total_iters, learning_rate, warmup_ratio=0.1, plateau_ratio=0.7):
    """为单一参数组调整学习率"""
    # # 先把学习率乘以0.2
    # learning_rate *= 0.2
    warmup_iters = int(total_iters * warmup_ratio)
    plateau_iters = int(total_iters * plateau_ratio)
    
    if iter_count < warmup_iters:
        # 预热阶段，线性增加学习率
        return learning_rate * (iter_count / warmup_iters)
    elif iter_count < plateau_iters:
        # 稳定学习率阶段
        return learning_rate
    else:
        # 余弦衰减阶段
        decay_ratio = (iter_count - plateau_iters) / (total_iters - plateau_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return learning_rate * cosine_decay

if __name__ == "__main__":
    main()