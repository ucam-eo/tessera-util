#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用法示例:
   torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 src/train_365.py
"""

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

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

import torch.amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import wandb
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F

# ==================== 原有引用: Barlow Twins相关、验证函数、mixup等 ====================
from models.ssl_model import BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate
from utils.metrics import linear_probe_evaluate, rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr, plot_4_augs_in_one

import torch.nn.attention as attn

# ==================== 新增/修改的部分：PixelTransformer, 新Dataset等 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training (Multi-GPU + chunk-based loading, new 365-day approach)")
    parser.add_argument('--config', type=str, default="configs/ssl_config_365.py",
                        help="Path to config file (e.g. configs/new_ssl_config.py)")
    return parser.parse_args()


class ChunkDataset365(Dataset):
    """
    加载一批(多个)文件(通常是S1和S2各对应的文件对)，将其拼到内存里；
    每条样本形状：
      - s2_bands: (365, 10)
      - s2_mask: (365,)
      - s1_bands: (365, 2)
      - s1_mask: (365,)
    动态生成两个增强版本(aug1, aug2)并返回.

    假设文件命名形如:
      s2: B1_F100_365_bands.npy, B1_F100_365_masks.npy, ...
      s1: B1_F100_365_bands.npy, B1_F100_365_masks.npy, ...
    并且 s2_files[i] 与 s1_files[i] 在文件前缀上对应同一个块(比如 B1_F100_...).
    """

    def __init__(self, s1_files, s2_files, config):
        super().__init__()
        self.config = config
        self.all_s1_bands = []
        self.all_s1_masks = []
        self.all_s2_bands = []
        self.all_s2_masks = []
        self.all_len = 0

        # ============= 依次读取文件到CPU内存 =============
        for i in range(len(s1_files)):
            s1_band_file, s1_mask_file = s1_files[i]
            s2_band_file, s2_mask_file = s2_files[i]

            s1_bands_np = np.load(s1_band_file)  # shape: (N,365,2)
            s1_masks_np = np.load(s1_mask_file)  # shape: (N,365)
            s2_bands_np = np.load(s2_band_file)  # shape: (N,365,10)
            s2_masks_np = np.load(s2_mask_file)  # shape: (N,365)

            # 这里做简单的拼接
            self.all_s1_bands.append(s1_bands_np)
            self.all_s1_masks.append(s1_masks_np)
            self.all_s2_bands.append(s2_bands_np)
            self.all_s2_masks.append(s2_masks_np)
            self.all_len += s1_bands_np.shape[0]

        # 将所有块拼接到一个大array中(也可不拼接，根据情况)
        self.all_s1_bands = np.concatenate(self.all_s1_bands, axis=0)  # (total_samples,365,2)
        self.all_s1_masks = np.concatenate(self.all_s1_masks, axis=0)  # (total_samples,365)
        self.all_s2_bands = np.concatenate(self.all_s2_bands, axis=0)  # (total_samples,365,10)
        self.all_s2_masks = np.concatenate(self.all_s2_masks, axis=0)  # (total_samples,365)

        # 再次打乱
        idxs = np.arange(self.all_len)
        np.random.shuffle(idxs)
        self.all_s1_bands = self.all_s1_bands[idxs]
        self.all_s1_masks = self.all_s1_masks[idxs]
        self.all_s2_bands = self.all_s2_bands[idxs]
        self.all_s2_masks = self.all_s2_masks[idxs]

    def __len__(self):
        return self.all_len

    def __getitem__(self, index):
        """
        返回: s2_aug1_bands, s2_aug1_masks, s2_aug2_bands, s2_aug2_masks,
              s1_aug1_bands, s1_aug1_masks, s1_aug2_bands, s1_aug2_masks
        都是 Tensor 类型
        """
        s1_bands = self.all_s1_bands[index]  # (365,2)
        s1_mask  = self.all_s1_masks[index]  # (365,)
        s2_bands = self.all_s2_bands[index]  # (365,10)
        s2_mask  = self.all_s2_masks[index]  # (365,)

        # 动态生成两个增强
        # 强调：这里仅仅对 mask 做改动(从2->1)，bands 的原始值保留
        s1_aug1_bands, s1_aug1_mask = self.random_mask_augmentation(s1_bands, s1_mask)
        s1_aug2_bands, s1_aug2_mask = self.random_mask_augmentation(s1_bands, s1_mask)
        s2_aug1_bands, s2_aug1_mask = self.random_mask_augmentation(s2_bands, s2_mask)
        s2_aug2_bands, s2_aug2_mask = self.random_mask_augmentation(s2_bands, s2_mask)

        return {
            "s2_aug1_bands": torch.from_numpy(s2_aug1_bands).float(),
            "s2_aug1_mask":  torch.from_numpy(s2_aug1_mask).long(),
            "s2_aug2_bands": torch.from_numpy(s2_aug2_bands).float(),
            "s2_aug2_mask":  torch.from_numpy(s2_aug2_mask).long(),
            "s1_aug1_bands": torch.from_numpy(s1_aug1_bands).float(),
            "s1_aug1_mask":  torch.from_numpy(s1_aug1_mask).long(),
            "s1_aug2_bands": torch.from_numpy(s1_aug2_bands).float(),
            "s1_aug2_mask":  torch.from_numpy(s1_aug2_mask).long(),
        }

    def random_mask_augmentation(self, bands_np, mask_np):
        """
        1) 随机遮挡 20% 的有效时序点 (mask=2 -> 1)
        2) 随机选取30天的连续时间窗口, 对该窗口内的有效时序也做遮挡 (2->1)
        """
        # 复制一份, 不要改动原始
        aug_bands = bands_np.copy()  # shape: (365, band_num)
        aug_mask  = mask_np.copy()   # shape: (365,)
        
        # 生成0-1的随机数
        random_num = np.random.rand()

        # 随机遮挡 20% 有效点 ---
        if random_num < 0.5:
            valid_indices = np.where(aug_mask == 2)[0]
            if len(valid_indices) > 0:
                num_to_mask = int(0.2 * len(valid_indices))
                chosen = np.random.choice(valid_indices, size=num_to_mask, replace=False)
                aug_mask[chosen] = 1
        else:
            start_day = np.random.randint(0, 365 - 30 + 1)
            end_day = start_day + 30
            for d in range(start_day, end_day):
                if aug_mask[d] == 2:
                    aug_mask[d] = 1
                    
        # 注：bands 本身值我们不做改动(除非你想把被mask掉的地方置零, 但目前不需要).
        return aug_bands, aug_mask


# ---------------- 新的 PixelTransformer ----------------
class TemporalPositionalEncoder(nn.Module):
    """
    用于给 365 天做位置编码，这里给出一个最简单的实现，也可替换为可学习PE
    """
    def __init__(self, d_model=512, max_len=365):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x.shape = (B, T, d_model)
        这里直接将pe[:T, :]加到x上
        """
        T = x.size(1)
        pos_enc = self.pe[:T, :].unsqueeze(0).to(x.device)  # (1, T, d_model)
        return x + pos_enc

class PixelTransformer(nn.Module):
    """
    一个示例: 用可学习标记替换missing/invalid时序点 + TransformerEncoder
    band_num 不同，对应哨兵1 (2) 与 哨兵2 (10).
    """
    def __init__(self, band_num=10, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 波段特征嵌入
        self.band_embed = nn.Linear(band_num, d_model)

        # 定义可学习标记
        self.missing_token = nn.Parameter(torch.randn(d_model))  # 对应mask=0
        self.invalid_token = nn.Parameter(torch.randn(d_model))  # 对应mask=1

        # 位置编码
        self.pos_encoder = TemporalPositionalEncoder(d_model=d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, bands, mask):
        """
        bands: (B, 365, band_num)
        mask:  (B, 365) in {0,1,2}
           - 0 => missing
           - 1 => invalid
           - 2 => valid
        返回: (B, d_model) [做一个pool or取最后隐向量? 这里可以自己设计]
        """
        B, T, _ = bands.shape
        device = bands.device

        # 1) 先做线性嵌入: (B,T,band_num)->(B,T,d_model)
        x = self.band_embed(bands)  # (B,T,d_model)

        # 2) 把 missing/invalid 位置替换成可学习token
        #   mask==0 => missing
        #   mask==1 => invalid
        #   mask==2 => valid => 用原始 x
        #  首先复制 x，做一个 view 方便替换
        x_view = x.view(-1, x.shape[-1])  # (B*T, d_model)
        mask_flat = mask.view(-1)         # (B*T,)

        missing_pos  = (mask_flat == 0).unsqueeze(-1)  # (B*T, 1)
        invalid_pos  = (mask_flat == 1).unsqueeze(-1)  # (B*T, 1)
        missing_token = self.missing_token.unsqueeze(0)  # (1, d_model)
        invalid_token = self.invalid_token.unsqueeze(0)  # (1, d_model)

        # 先替换 missing
        x_view = torch.where(missing_pos, missing_token, x_view)
        # 再替换 invalid
        x_view = torch.where(invalid_pos, invalid_token, x_view)

        # 3) 位置编码
        x = x_view.view(B, T, -1)  # (B,T,d_model)
        x = self.pos_encoder(x)    # (B,T,d_model)

        # 4) 构建 Transformer 的 src_key_padding_mask
        #   只对 mask==2 (valid) 的位置参与注意力
        #   => mask==2 => True(不屏蔽), 其他 => False(需要屏蔽)
        #   PyTorch 的 src_key_padding_mask 语义: "True表示需要被padding/忽略"
        #   所以我们需要取反
        #   => key_padding_mask.shape=(B,T), True表示不参加attn
        bool_valid = (mask == 2)  # True/False
        key_padding_mask = ~bool_valid  # True的地方表示要被mask掉 => invalid or missing

        # 5) 输入 TransformerEncoder
        #   注意: src_key_padding_mask=(B,T), batch_first=True时可以直接传
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # 6) 做一个pooling: 只pool有效位置(可选做 mean pooling 也可只取CLS等)
        #   这里简单做 mean pool (只对 mask=2 的位置求平均)
        #   shape x: (B,T,d_model)
        #   mask=2 => valid => 1, else 0
        mask_2 = (mask == 2).float()  # (B,T)
        sum_valid = torch.sum(mask_2, dim=1, keepdim=True)  # (B,1)
        sum_valid = torch.clamp(sum_valid, min=1e-5)        # 避免除0
        # (B,T,d_model) -> (B,d_model)
        x_pool = torch.sum(x * mask_2.unsqueeze(-1), dim=1) / sum_valid

        return x_pool


class MultimodalBTNewModel(nn.Module):
    """
    新的多模态模型: S1用一个PixelTransformer, S2再用一个PixelTransformer, 然后融合, 最后投影
    """
    def __init__(self,
                 s2_transformer,   # PixelTransformer for S2(band_num=10)
                 s1_transformer,   # PixelTransformer for S1(band_num=2)
                 projector,        # ProjectionHead
                 fusion_method='concat',
                 return_repr=False):
        super().__init__()
        self.s2_transformer = s2_transformer
        self.s1_transformer = s1_transformer
        self.projector = projector
        self.fusion_method = fusion_method
        self.return_repr = return_repr

        # 根据融合方式不同, 可以做一个可选的线性变换
        if fusion_method == 'concat':
            # 假设 s2_transformer 输出 d_model=512, s1_transformer输出 512 => 拼接成 1024
            # 再线性变到 512
            self.fuse_linear = nn.Linear(1024, 128)
        elif fusion_method == 'sum':
            # 前提是 s2, s1 输出同维度 => 直接 element-wise sum => 512
            self.fuse_linear = nn.Linear(512, 128)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def forward(self,
                s2_bands, s2_mask,
                s1_bands, s1_mask):
        """
        s2_bands: (B,365,10)
        s2_mask:  (B,365)
        s1_bands: (B,365,2)
        s1_mask:  (B,365)
        """
        s2_repr = self.s2_transformer(s2_bands, s2_mask)  # (B,512)
        s1_repr = self.s1_transformer(s1_bands, s1_mask)  # (B,512)

        if self.fusion_method == 'concat':
            fused = torch.cat([s2_repr, s1_repr], dim=-1)  # (B,1024)
            fused = self.fuse_linear(fused)                # -> (B,512)
        elif self.fusion_method == 'sum':
            fused = s2_repr + s1_repr  # (B,512)
            fused = self.fuse_linear(fused)

        feats = self.projector(fused)  # (B, project_dim)

        if self.return_repr:
            return feats, fused        # feats for barlow, fused as "repr"
        else:
            return feats


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)


def main():
    dist.init_process_group(backend='nccl')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        os.environ['WANDB_MODE'] = 'disabled'  # 屏蔽多余rank的wandb日志

    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    # AMP
    apply_amp = config.get('apply_amp', False)
    if local_rank == 0:
        logging.info(f"apply_amp = {apply_amp}")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        logging.info(f"Running on {world_size} GPU(s). LocalRank={local_rank}, device={device}")

    # W&B init
    if local_rank == 0:
        os.environ["WANDB_API_KEY"] = "b03eca52bd30c1fa9bf185ae3ee91d9276f2f92a"
        if config.get("disable_wandb_git", False):
            os.environ['WANDB_DISABLE_GIT'] = 'true'
        run_name = f"BT_365_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run = wandb.init(project="btfm-iterable", name=run_name, config=config)
    else:
        wandb_run = None

    total_steps = config['epochs'] * config['total_samples'] // config['batch_size'] // world_size
    if local_rank == 0:
        logging.info(f"Total steps per rank (approx) = {total_steps}")

    # ============ 构建模型 ============
    s2_pt = PixelTransformer(
        band_num=10,
        d_model=config['d_model'],
        nhead=config['s2_num_heads'],
        num_layers=config['s2_num_layers'],
        dim_feedforward=config['s2_dim_feedforward']
    ).to(device)

    s1_pt = PixelTransformer(
        band_num=2,
        d_model=config['d_model'],
        nhead=config['s1_num_heads'],
        num_layers=config['s1_num_layers'],
        dim_feedforward=config['s1_dim_feedforward']
    ).to(device)

    projector = ProjectionHead(
        input_dim=config['fuse_out_dim'],  # 一般512
        hidden_dim=config['projector_hidden_dim'],
        output_dim=config['projector_out_dim']
    ).to(device)

    model = MultimodalBTNewModel(
        s2_pt, s1_pt, projector,
        fusion_method=config['fusion_method'],
        return_repr=True
    ).to(device)

    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])
    
    with attn.sdpa_kernel(attn.SDPBackend.MATH):

        if config.get('use_torch_compile', False):
            model = torch.compile(model, mode="default")
            if local_rank == 0:
                logging.info("Using torch.compile to optimize the model...")

        if local_rank == 0:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"Model has {num_params} trainable parameters.")

        weight_params = [p for n, p in model.named_parameters() if p.ndim > 1]
        bias_params   = [p for n, p in model.named_parameters() if p.ndim == 1]
        optimizer = torch.optim.AdamW(
            [{'params': weight_params}, {'params': bias_params}],
            lr=config['learning_rate'], weight_decay=1e-6
        )

        scaler = torch.amp.GradScaler("cuda") if apply_amp else None

        # DDP 封装
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if local_rank == 0:
            wandb.watch(model.module, log="gradients", log_freq=200)

        # ============ 准备数据, chunk-based ============

        def _train_one_chunk(s1_chunk_files, s2_chunk_files, epoch, step, examples,
                            last_time, last_examples, rolling_loss):
            """
            s1_chunk_files, s2_chunk_files: [(bands_file, masks_file), ...]
            """
            chunk_dataset = ChunkDataset365(s1_chunk_files, s2_chunk_files, config)
            sampler = DistributedSampler(chunk_dataset, shuffle=False, drop_last=True)
            loader = DataLoader(
                chunk_dataset,
                batch_size=config['batch_size'],
                sampler=sampler,
                num_workers=config['num_workers'],
                drop_last=True,
                pin_memory=False
            )
            sampler.set_epoch(epoch)

            if local_rank == 0:
                logging.info(f" => This chunk has {len(chunk_dataset)} samples, dataloader steps={len(loader)}")

            model.train()

            for batch_data in loader:
                s2_aug1_bands = batch_data['s2_aug1_bands'].to(device, non_blocking=True)
                s2_aug1_mask  = batch_data['s2_aug1_mask'].to(device, non_blocking=True)
                s2_aug2_bands = batch_data['s2_aug2_bands'].to(device, non_blocking=True)
                s2_aug2_mask  = batch_data['s2_aug2_mask'].to(device, non_blocking=True)

                s1_aug1_bands = batch_data['s1_aug1_bands'].to(device, non_blocking=True)
                s1_aug1_mask  = batch_data['s1_aug1_mask'].to(device, non_blocking=True)
                s1_aug2_bands = batch_data['s1_aug2_bands'].to(device, non_blocking=True)
                s1_aug2_mask  = batch_data['s1_aug2_mask'].to(device, non_blocking=True)

                adjust_learning_rate(
                    optimizer,
                    step,
                    total_steps,
                    config['learning_rate'],
                    config['warmup_ratio'],
                    config['plateau_ratio']
                )
                optimizer.zero_grad()

                if apply_amp:
                    with torch.amp.autocast("cuda"):
                        z1, repr1 = model(s2_aug1_bands, s2_aug1_mask, s1_aug1_bands, s1_aug1_mask)
                        z2, repr2 = model(s2_aug2_bands, s2_aug2_mask, s1_aug2_bands, s1_aug2_mask)
                        loss_main, bar_main, off_main = criterion(z1, z2)

                        # mixup
                        loss_mix = 0.0
                        if config.get('apply_mixup', False):
                            B = s2_aug1_bands.size(0)
                            idxs = torch.randperm(B, device=device)
                            alpha = torch.distributions.Beta(
                                config['beta_alpha'],
                                config['beta_beta']
                            ).sample().to(device)
                            # 对 S2 做 mix
                            y_m_s2 = alpha * s2_aug1_bands + (1 - alpha) * s2_aug2_bands[idxs, :]
                            # 同理对 mask 做“合并”意义不大，这里可以直接取2个mask的 "或"？？ 
                            # 但 mix mask 不太直观，先简单保留aug1的mask
                            y_m_s2_mask = s2_aug1_mask.clone()

                            # 对 S1 做 mix
                            y_m_s1 = alpha * s1_aug1_bands + (1 - alpha) * s1_aug2_bands[idxs, :]
                            y_m_s1_mask = s1_aug1_mask.clone()

                            z_m, _ = model(y_m_s2, y_m_s2_mask, y_m_s1, y_m_s1_mask)
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
                    z1, repr1 = model(s2_aug1_bands, s2_aug1_mask, s1_aug1_bands, s1_aug1_mask)
                    z2, repr2 = model(s2_aug2_bands, s2_aug2_mask, s1_aug2_bands, s1_aug2_mask)
                    loss_main, bar_main, off_main = criterion(z1, z2)

                    loss_mix = 0.0
                    if config.get('apply_mixup', False):
                        B = s2_aug1_bands.size(0)
                        idxs = torch.randperm(B, device=device)
                        alpha = torch.distributions.Beta(
                            config['beta_alpha'],
                            config['beta_beta']
                        ).sample().to(device)
                        # 对S2做mix
                        y_m_s2 = alpha * s2_aug1_bands + (1 - alpha) * s2_aug2_bands[idxs, :]
                        y_m_s2_mask = s2_aug1_mask.clone()
                        # 对S1做mix
                        y_m_s1 = alpha * s1_aug1_bands + (1 - alpha) * s1_aug2_bands[idxs, :]
                        y_m_s1_mask = s1_aug1_mask.clone()

                        z_m, _ = model(y_m_s2, y_m_s2_mask, y_m_s1, y_m_s1_mask)
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

                batch_sz = s2_aug1_bands.size(0)
                examples += batch_sz

                # 日志
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

                    logging.info(
                        f"[Epoch={epoch}, Step={step}] "
                        f"Loss={loss_main.item():.2f}, MixLoss={loss_mix:.2f}, AvgLoss={avg_loss:.2f}, "
                        f"LR={current_lr:.4f}, batchsize={batch_sz}, Examples/sec={exps:.2f}, "
                        f"Rank(z)={erank_z:.4f}, Rank(repr)={erank_repr:.4f}"
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
                    }

                    # 定期画cross corr
                    if step % (10 * config['log_interval_steps']) == 0:
                        # 1) 从本batch随便挑一个index
                        sample_idx = np.random.randint(0, batch_sz)

                        # 2) 取出 mask
                        mask_s2_a1 = s2_aug1_mask[sample_idx].cpu().numpy()
                        mask_s2_a2 = s2_aug2_mask[sample_idx].cpu().numpy()
                        mask_s1_a1 = s1_aug1_mask[sample_idx].cpu().numpy()
                        mask_s1_a2 = s1_aug2_mask[sample_idx].cpu().numpy()

                        # 3) 画图
                        fig_masks = plot_4_augs_in_one(mask_s2_a1, mask_s2_a2,
                                                    mask_s1_a1, mask_s1_a2,
                                                    title="Mask Visualization at Step={}".format(step))
                        wandb_dict["mask_aug_composite"] = wandb.Image(fig_masks)
                        plt.close(fig_masks)
                        
                        try:
                            fig_cc = plot_cross_corr(z1, z2)
                            cross_corr_img = wandb.Image(fig_cc)
                            plt.close(fig_cc)
                            wandb_dict["cross_corr"] = cross_corr_img
                        except:
                            pass
                        try:
                            fig_cc_repr = plot_cross_corr(repr1, repr2)
                            cross_corr_img_repr = wandb.Image(fig_cc_repr)
                            plt.close(fig_cc_repr)
                            wandb_dict["cross_corr_repr"] = cross_corr_img_repr
                        except:
                            pass

                    wandb.log(wandb_dict, step=step)

                # 验证 + 保存best
                # TODO: to be implemented
                step += 1

            # 清理
            try:
                loader._iterator._shutdown_workers()
            except:
                pass
            del loader
            del sampler
            del chunk_dataset
            gc.collect()

            return step, examples, last_time, last_examples, rolling_loss


        # ============ 开始训练: 多Epoch + 多Chunk ============

        global best_val_acc
        best_val_acc = 0.0
        step = 0
        examples = 0
        last_time = time.time()
        last_examples = 0
        rolling_loss = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_{timestamp}.pt")
        os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)

        for epoch in range(config['epochs']):
            # 收集所有文件, 按 chunk_batch 切分
            # 假设 S1 文件夹: data/ssl_training/ready_to_use_365/s1
            #       S2 文件夹: data/ssl_training/ready_to_use_365/s2
            # 里面的文件成对出现: B1_F100_365_bands.npy + B1_F100_365_masks.npy
            # 这里提供示例做法:
            s1_dir = os.path.join(config['data_root'], 's1')
            s2_dir = os.path.join(config['data_root'], 's2')
            all_s1_bands = sorted([f for f in os.listdir(s1_dir) if f.endswith('bands.npy')])
            # 通过替换字符串找到对应的 masks 文件
            # 也可以 zip 方式: B1_F100_365_bands.npy -> B1_F100_365_masks.npy
            # 具体需你确保文件命名匹配
            all_s1_pairs = []
            for bf in all_s1_bands:
                base = bf.replace('bands.npy', '')
                maskf = base + 'masks.npy'
                all_s1_pairs.append((os.path.join(s1_dir, bf), os.path.join(s1_dir, maskf)))

            all_s2_bands = sorted([f for f in os.listdir(s2_dir) if f.endswith('bands.npy')])
            all_s2_pairs = []
            for bf in all_s2_bands:
                base = bf.replace('bands.npy', '')
                maskf = base + 'masks.npy'
                all_s2_pairs.append((os.path.join(s2_dir, bf), os.path.join(s2_dir, maskf)))

            # 打乱顺序
            idx = np.arange(len(all_s1_pairs))
            np.random.shuffle(idx)
            all_s1_pairs = [all_s1_pairs[i] for i in idx]
            all_s2_pairs = [all_s2_pairs[i] for i in idx]

            chunk_batch = config.get('chunk_batch', 5)
            chunk_start = 0
            total_files = len(all_s1_pairs)
            if local_rank == 0:
                logging.info(f"Epoch {epoch}: total new files = {total_files}")

            while chunk_start < total_files:
                chunk_end = min(chunk_start + chunk_batch, total_files)
                s1_chunk_files = all_s1_pairs[chunk_start:chunk_end]
                s2_chunk_files = all_s2_pairs[chunk_start:chunk_end]

                if local_rank == 0:
                    logging.info(f"Epoch {epoch}, chunk [{chunk_start}:{chunk_end}], loading {len(s1_chunk_files)} files...")

                step, examples, last_time, last_examples, rolling_loss = _train_one_chunk(
                    s1_chunk_files, s2_chunk_files, epoch, step, examples,
                    last_time, last_examples, rolling_loss
                )
                chunk_start = chunk_end

            if local_rank == 0:
                logging.info(f"Epoch {epoch} finished, current step = {step}")

        if local_rank == 0:
            logging.info("Training completed.")
            wandb_run.finish()

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
