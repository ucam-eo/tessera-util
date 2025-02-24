#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NUMEXPR_MAX_THREADS"] = "24"

import math
import time
import gc
import argparse
import logging
import subprocess
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp  # 若在Intel XPU不可用，请改为 from intel_extension_for_pytorch.xpu.amp import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch

import wandb
import matplotlib.pyplot as plt

# ============== 你原先用到的模块 / 函数 ===============
from models.modules import TransformerEncoder, ProjectionHead, SpectralTemporalTransformer
from models.ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate
from utils.metrics import linear_probe_evaluate, rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr

# mpirun -n 8 python src/train_multi_xpu.py

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
                # 在这里若有标准化需求，可以先行处理，
                # 或者在 __getitem__ 中做都行。
                self.all_samples.append((
                    arr_aug1_s2[i],
                    arr_aug2_s2[i],
                    arr_aug1_s1[i],
                    arr_aug2_s1[i]
                ))

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
# 2) 主训练脚本 (多XPU + chunk-based loading)
##########################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training (Multi-XPU + chunk-based loading)")
    parser.add_argument('--config', type=str, default="configs/ssl_config.py",
                        help="Path to config file (e.g. configs/ssl_config.py)")
    return parser.parse_args()

def main():
    ############################################################################
    # 0) 初始化分布式环境 (oneCCL + PyTorch)
    ############################################################################
    mpi_world_size = int(os.environ.get('PMI_SIZE', '-1'))
    mpi_rank = int(os.environ.get('PMI_RANK', '-1'))
    if mpi_world_size > 0:
        os.environ['WORLD_SIZE'] = str(mpi_world_size)
        os.environ['RANK'] = str(mpi_rank)
    else:
        os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '1')
        os.environ['RANK'] = os.environ.get('RANK', '0')

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    dist.init_process_group(backend='ccl')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = global_rank

    ############################################################################
    # 1) 解析命令行 & 读取配置
    ############################################################################
    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    # rank=0 输出info日志，其它只输出warning
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        os.environ['WANDB_MODE'] = 'disabled'  # 屏蔽多余rank的W&B日志

    device_str = f"xpu:{local_rank}"
    torch.xpu.set_device(device_str)
    if local_rank == 0:
        logging.info(f"Running on {world_size} XPU(s). LocalRank={local_rank}, device={device_str}")

    # 读取 chunk_batch 配置
    chunk_batch = config.get('chunk_batch', 40)
    if local_rank == 0:
        logging.info(f"chunk_batch = {chunk_batch}")

    ############################################################################
    # 2) 准备W&B
    ############################################################################
    run_name = f"BT_Iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if local_rank == 0:
        wandb_run = wandb.init(project="btfm-iterable", name=run_name, config=config)
    else:
        wandb_run = None

    # 计算总训练步数（可能并不精准，因为我们分块加载）
    total_steps = config['epochs'] * config['total_samples'] // config['batch_size']
    if local_rank == 0:
        logging.info(f"Total steps (approx) = {total_steps}")

    ############################################################################
    # 3) 构建模型
    ############################################################################
    s2_enc = TransformerEncoder(
        band_num=12,  # 例如 10个波段+sin/cos
        latent_dim=config['latent_dim'],
        nhead=16,
        num_encoder_layers=32,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=config['sample_size_s2']
    ).to(device_str)
    s1_enc = TransformerEncoder(
        band_num=4,   # 例如 2个波段+sin/cos
        latent_dim=config['latent_dim'],
        nhead=16,
        num_encoder_layers=32,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=config['sample_size_s1']
    ).to(device_str)
    
    # s2_enc = SpectralTemporalTransformer(data_dim=12, nhead=16, num_layers=32).to(device_str)
    # s1_enc = SpectralTemporalTransformer(data_dim=4, nhead=16, num_layers=32).to(device_str)

    if config['fusion_method'] == 'concat':
        proj_in_dim = config['latent_dim'] * 2
    else:
        proj_in_dim = config['latent_dim']
    projector = ProjectionHead(proj_in_dim,
                               config['projector_hidden_dim'],
                               config['projector_out_dim']).to(device_str)

    if config['fusion_method'] == 'transformer':
        model = MultimodalBTModel(s2_enc, s1_enc, projector,
                                  fusion_method=config['fusion_method'],
                                  return_repr=True,
                                  latent_dim=config['latent_dim']).to(device_str)
    else:
        model = MultimodalBTModel(s2_enc, s1_enc, projector,
                                  fusion_method=config['fusion_method'],
                                  return_repr=True).to(device_str)

    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])

    if local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model has {num_params} trainable parameters.")

    weight_params = [p for n, p in model.named_parameters() if p.ndim > 1]
    bias_params   = [p for n, p in model.named_parameters() if p.ndim == 1]
    optimizer = torch.optim.SGD([{'params': weight_params},
                                 {'params': bias_params}],
                                lr=config['learning_rate'],
                                momentum=0.9,
                                weight_decay=1e-6)

    scaler = amp.GradScaler()

    # IPEX优化 & DDP封装
    model, optimizer = ipex.optimize(model, optimizer=optimizer,
                                     dtype=torch.float32,
                                     auto_kernel_selection=False,
                                     inplace=False)
    model = DDP(model,
                device_ids=[device_str],
                output_device=device_str,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=False)

    ############################################################################
    # 4) 训练循环 (chunk-based)
    ############################################################################
    step = 0
    examples = 0
    last_time = time.time()
    last_examples = 0
    rolling_loss = []
    rolling_size = 40
    best_val_acc = 0.0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_{timestamp}.pth")

    # ============== 开始多个 EPOCH ==============
    for epoch in range(config['epochs']):
        # rank=0 进程调用rust生成数据
        if local_rank == 0:
            aug1_dir = os.path.join(config['data_root'], 'aug1')
            aug2_dir = os.path.join(config['data_root'], 'aug2')
            remove_dir(aug1_dir)
            remove_dir(aug2_dir)
            logging.info(f"Epoch {epoch} started. Generating new training data via rust_cmd...")
            subprocess.run(config['rust_cmd'], shell=True, check=True)
            logging.info(f"Data generation finished for epoch {epoch}.")

        # 同步，等待数据全部生成
        dist.barrier()

        # ============ 收集所有文件名(这里假设 file_names 是 S2 的 aug1) ============
        #   然后推断 S2 aug2, S1 aug1, S1 aug2 的对应文件
        #   你也可以在这里写更健壮的逻辑，比如把4个文件都收集好
        aug1_s2_dir = os.path.join(config['data_root'], 'aug1', 's2')
        aug2_s2_dir = os.path.join(config['data_root'], 'aug2', 's2')
        aug1_s1_dir = os.path.join(config['data_root'], 'aug1', 's1')
        aug2_s1_dir = os.path.join(config['data_root'], 'aug2', 's1')

        s2_file_names = sorted(os.listdir(aug1_s2_dir))  # Rust生成时已随机
        total_files = len(s2_file_names)
        if local_rank == 0:
            logging.info(f"Epoch {epoch}: total new files = {total_files} in each folder")

        # ============ 分块加载训练 ============
        chunk_start = 0
        while chunk_start < total_files:
            chunk_end = min(chunk_start + chunk_batch, total_files)
            # 取出这一批文件名
            chunk_files = s2_file_names[chunk_start:chunk_end]
            if local_rank == 0:
                logging.info(f"Epoch {epoch}, chunk [{chunk_start}:{chunk_end}], loading {len(chunk_files)} files...")

            # 构造4组文件路径列表
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
            # 分布式Sampler
            train_sampler = DistributedSampler(chunk_dataset, shuffle=False, drop_last=True)
            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                chunk_dataset,
                batch_size=config['batch_size'],
                sampler=train_sampler,
                num_workers=config['num_workers'],
                drop_last=True,
                pin_memory=True,
                persistent_workers=True
            )

            # 如果你想对chunk内部再shuffle，可以把 DistributedSampler(..., shuffle=True)
            #   但大多数情况下，Rust那边已经shuffle好了

            train_sampler.set_epoch(epoch)  # 确保多Epoch shuffle种子不同

            if local_rank == 0:
                logging.info(f"   => This chunk has {len(chunk_dataset)} samples total, steps in dataloader = {len(train_loader)}.")

            model.train()

            # ============ 在这一批文件上进行训练 =============
            for batch_data in train_loader:
                s2_aug1 = batch_data['s2_aug1'].to(device_str, non_blocking=True)
                s2_aug2 = batch_data['s2_aug2'].to(device_str, non_blocking=True)
                s1_aug1 = batch_data['s1_aug1'].to(device_str, non_blocking=True)
                s1_aug2 = batch_data['s1_aug2'].to(device_str, non_blocking=True)

                adjust_learning_rate(optimizer, step, total_steps,
                                     config['learning_rate'],
                                     config['warmup_ratio'],
                                     config['plateau_ratio'])

                optimizer.zero_grad()
                with amp.autocast():
                    z1, repr1 = model(s2_aug1, s1_aug1)
                    z2, repr2 = model(s2_aug2, s1_aug2)
                    loss_main, bar_main, off_main = criterion(z1, z2)

                    # Mixup
                    loss_mix = 0.0
                    if config['apply_mixup']:
                        B = s2_aug1.size(0)
                        idxs = torch.randperm(B, device=device_str)
                        alpha = torch.distributions.Beta(config['beta_alpha'], config['beta_beta']).sample().to(device_str)
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
                        loss_mix = config['mixup_lambda'] * config['barlow_lambda'] * (diff_a + diff_b)

                    total_loss = loss_main + loss_mix

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()

                batch_sz = s2_aug1.size(0)
                examples += batch_sz

                if (step % config['log_interval_steps'] == 0) and (local_rank == 0):
                    current_time = time.time()
                    exps = (examples - last_examples) / (current_time - last_time)
                    last_time = current_time
                    last_examples = examples
                    rolling_loss.append(loss_main.item())
                    if len(rolling_loss) > rolling_size:
                        rolling_loss = rolling_loss[-rolling_size:]
                    avg_loss = sum(rolling_loss)/len(rolling_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    erank_z = rankme(z1)
                    erank_repr = rankme(repr1)
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

                    # 定期画一下相关系数
                    if step % (10 * config['log_interval_steps']) == 0:
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
                if (config['val_interval_steps'] > 0) and (step > 0) and (step % config['val_interval_steps'] == 0) and (local_rank == 0):
                    if all(config.get(k) for k in [
                        'val_s2_bands_file_path', 'val_s2_masks_file_path', 'val_s2_doy_file_path',
                        'val_s1_asc_bands_file_path', 'val_s1_asc_doy_file_path',
                        'val_s1_desc_bands_file_path', 'val_s1_desc_doy_file_path',
                        'val_labels_path'
                    ]):
                        from torch.utils.data import DataLoader
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
                        val_acc = linear_probe_evaluate(model.module, val_loader, device=device_str)
                        wandb.log({"val_acc": val_acc}, step=step)
                        logging.info(f"Validation at step {step}: val_acc={val_acc:.4f}")
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            save_checkpoint(model.module, optimizer, epoch, step, best_val_acc, best_ckpt_path)
                        model.train()

                step += 1  # 全局step

            # ============ chunk训练结束，释放内存 ============
            del train_loader, chunk_dataset
            gc.collect()

            chunk_start = chunk_end  # 进入下一个 chunk

        if local_rank == 0:
            logging.info(f"Epoch {epoch} finished, current step = {step}")

    # 训练结束
    if local_rank == 0:
        logging.info("Training completed.")
        wandb_run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
