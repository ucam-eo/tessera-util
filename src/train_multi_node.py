#!/usr/bin/env python
# -*- coding: utf-8 -*-

ENABLE_WANDB = True

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
from socket import gethostname

import numpy as np
import torch
import torch.amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch

# 如果需要，才导入 wandb
if ENABLE_WANDB:
    import wandb
import matplotlib.pyplot as plt

# ============== 你原先用到的模块 / 函数 ===============
from models.modules import TransformerEncoder, ProjectionHead, SpectralTemporalTransformer
from models.ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate
from utils.metrics import linear_probe_evaluate, rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr

# 你也可以再加一些 DEBUG 标记
DEBUG_STEPS_IN_DATALOADER = 100      # 每多少个step打印一次 "I'm in dataloader iteration"
DEBUG_ITEMS_IN_GETITEM = 1000000     # 每加载多少条数据打印一次信息（过大数据时可以改小）

# 新增：可选的调试开关，用于在每个 iteration 后执行 dist.barrier()
DEBUG_DIST_BARRIER = False

##########################################################################
# 1) 一个用于加载"部分文件"的Dataset类，并且加更多调试log
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
        start_time = time.time()
        logging.info(f"[ChunkDataset] __init__ start, #files={len(aug1_s2_files)}")

        self.all_samples = []
        total_samples_loaded = 0

        # 读取这批文件
        for idx_file in range(len(aug1_s2_files)):
            file_aug1_s2 = aug1_s2_files[idx_file]
            file_aug2_s2 = aug2_s2_files[idx_file]
            file_aug1_s1 = aug1_s1_files[idx_file]
            file_aug2_s1 = aug2_s1_files[idx_file]

            t0 = time.time()
            # 增加 try/except 防止意外IO错误
            try:
                arr_aug1_s2 = np.load(file_aug1_s2, allow_pickle=False)
                arr_aug2_s2 = np.load(file_aug2_s2, allow_pickle=False)
                arr_aug1_s1 = np.load(file_aug1_s1, allow_pickle=False)
                arr_aug2_s1 = np.load(file_aug2_s1, allow_pickle=False)
            except Exception as e:
                logging.error(f"[ChunkDataset] Error loading file(s): {file_aug1_s2}, {file_aug2_s2}, {file_aug1_s1}, {file_aug2_s1}")
                logging.error(str(e))
                raise

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
            total_samples_loaded += n_samples

            t1 = time.time()
            logging.info(f"[ChunkDataset] Loaded file idx={idx_file}, #samples={n_samples}, time={t1 - t0:.2f}s")

        end_time = time.time()
        logging.info(f"[ChunkDataset] __init__ done. total_samples={total_samples_loaded}, took={end_time - start_time:.2f}s")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        # 每隔一大段输出一下日志，避免刷屏太多
        if index % DEBUG_ITEMS_IN_GETITEM == 0 and index > 0:
            logging.info(f"[ChunkDataset] __getitem__ index={index}/{len(self.all_samples)}")

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
# 2) 主训练脚本 (多节点多XPU + chunk-based loading)
##########################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training (Multi-Node, Multi-XPU + chunk-based loading)")
    parser.add_argument('--config', type=str, default="configs/ssl_config.py",
                        help="Path to config file (e.g. configs/ssl_config.py)")
    return parser.parse_args()

def main():
    torch.autograd.set_detect_anomaly(True)

    # 设置初始化种子 (确保所有rank使用同样的seed, 包括numpy)
    torch.manual_seed(3407)
    np.random.seed(3407)

    ############################################################################
    # 0) 初始化分布式环境 (oneCCL + PyTorch)
    ############################################################################
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    init_method   = 'tcp://' + os.environ["MASTER_ADDR"] + ':' + os.environ["MASTER_PORT"]
    rank          = int(os.environ.get("PMI_RANK", -1))
    world_size    = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend='ccl', init_method=init_method, rank=rank, world_size=world_size)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 这里假设 "每个节点的local rank = global rank在本节点的偏移"
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
    # node_idx = global_rank // gpus_per_node
    # local_rank = global_rank - node_idx * gpus_per_node
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    
    logging.info(
        f"Rank = {rank}, local Rank = {local_rank}, sees torch.xpu.device_count() = {torch.xpu.device_count()}, "
        f"SLURM_GPUS_ON_NODE = {os.environ.get('SLURM_GPUS_ON_NODE')}, "
        f"ZE_FLAT_DEVICE_HIERARCHY = {os.environ.get('ZE_FLAT_DEVICE_HIERARCHY')}"
    )

    # rank=0 输出info日志，其它只输出warning
    if global_rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        if ENABLE_WANDB:
            os.environ['WANDB_MODE'] = 'disabled'  # 屏蔽多余rank的W&B日志

    # 先给 rank=0 来一条提示
    if global_rank == 0:
        logging.info("============ Launching PyTorch Distributed Training ============")
        logging.info(f"world_size={world_size}, global_rank={global_rank}, local_rank={local_rank}")

    # device = torch.device("xpu", local_rank)
    device = f"xpu:{local_rank}"
    torch.xpu.set_device(device)
    # if global_rank == 0:
    logging.info(f"Global rank: {rank}, hostname: {gethostname()}, local rank: {local_rank}, Running on device={device}, world_size={world_size}")

    ############################################################################
    # 1) 解析命令行 & 读取配置
    ############################################################################
    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    # 新增：根据配置文件中的apply_amp字段决定是否启用AMP，默认为False
    apply_amp = config.get('apply_amp', False)
    if global_rank == 0:
        logging.info(f"apply_amp = {apply_amp}")

    # 读取 chunk_batch 配置
    chunk_batch = config.get('chunk_batch', 40)
    if global_rank == 0:
        logging.info(f"chunk_batch = {chunk_batch}")

    # 计算总训练步数（可能并不精准，因为我们分块加载）
    total_steps = config['epochs'] * config['total_samples'] // config['batch_size'] // world_size
    if global_rank == 0:
        logging.info(f"Total steps per rank (approx) = {total_steps}")

    ############################################################################
    # 2) 准备W&B
    ############################################################################
    run_name = f"BT_Iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if ENABLE_WANDB:
        if global_rank == 0:
            wandb_run = wandb.init(project="btfm-iterable", name=run_name, config=config)
        else:
            wandb_run = None
    else:
        wandb_run = None

    ############################################################################
    # 3) 构建模型
    ############################################################################
    s2_enc = TransformerEncoder(
        band_num=12,
        latent_dim=config['latent_dim'],
        nhead=16,
        num_encoder_layers=32,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=config['sample_size_s2']
    ).to(device)
    s1_enc = TransformerEncoder(
        band_num=4,
        latent_dim=config['latent_dim'],
        nhead=16,
        num_encoder_layers=32,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=config['sample_size_s1']
    ).to(device)

    if config['fusion_method'] == 'concat':
        proj_in_dim = config['latent_dim'] * 2
    else:
        proj_in_dim = config['latent_dim']

    projector = ProjectionHead(
        proj_in_dim,
        config['projector_hidden_dim'],
        config['projector_out_dim']
    ).to(device)

    if config['fusion_method'] == 'transformer':
        model = MultimodalBTModel(
            s2_enc, s1_enc, projector,
            fusion_method=config['fusion_method'],
            return_repr=True,
            latent_dim=config['latent_dim']
        ).to(device)
    else:
        model = MultimodalBTModel(
            s2_enc, s1_enc, projector,
            fusion_method=config['fusion_method'],
            return_repr=True
        ).to(device)

    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])

    if global_rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model has {num_params} trainable parameters.")

    weight_params = [p for n, p in model.named_parameters() if p.ndim > 1]
    bias_params   = [p for n, p in model.named_parameters() if p.ndim == 1]
    optimizer = torch.optim.SGD([{'params': weight_params},
                                 {'params': bias_params}],
                                lr=config['learning_rate'],
                                momentum=0.9,
                                weight_decay=1e-6)

    # AMP
    scaler = torch.amp.GradScaler() if apply_amp else None

    # IPEX优化 & DDP封装
    # model, optimizer = ipex.optimize(model, optimizer=optimizer,
    #                                  dtype=torch.float32,
    #                                 #  auto_kernel_selection=False,
    #                                  inplace=False
    # )
    
    model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
    
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=False)
    
    # model = model.to(device)
    # model = DDP(model, device_ids=[local_rank])
    
    logging.info(f"Model built and moved to device {device}")

    ############################################################################
    # 4) 训练循环 (chunk-based)
    ############################################################################

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
        start_t_chunk = time.time()

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

        if global_rank == 0:
            logging.info(f"   => This chunk has {len(chunk_dataset)} samples total, steps in dataloader = {len(train_loader)}.")

        model.train()

        for step_in_loader, batch_data in enumerate(train_loader):
            s2_aug1 = batch_data['s2_aug1'].to(device, non_blocking=True)
            s2_aug2 = batch_data['s2_aug2'].to(device, non_blocking=True)
            s1_aug1 = batch_data['s1_aug1'].to(device, non_blocking=True)
            s1_aug2 = batch_data['s1_aug2'].to(device, non_blocking=True)

            adjust_learning_rate(optimizer, step, total_steps,
                                 config['learning_rate'],
                                 config['warmup_ratio'],
                                 config['plateau_ratio'])

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast(device_type="xpu"):
                    z1, repr1 = model(s2_aug1, s1_aug1)
                    z2, repr2 = model(s2_aug2, s1_aug2)
                    loss_main, bar_main, off_main = criterion(z1, z2)

                    loss_mix = 0.0
                    if config.get('apply_mixup', False):
                        
                        z1 = z1.detach()  # 不再让mixup那部分对z1回传梯度
                        z2 = z2.detach()
                        
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
                        loss_mix = config['mixup_lambda'] * config['barlow_lambda'] * (diff_a + diff_b)

                    total_loss = loss_main + loss_mix

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
            else:
                z1, repr1 = model(s2_aug1, s1_aug1)
                z2, repr2 = model(s2_aug2, s1_aug2)
                loss_main, bar_main, off_main = criterion(z1, z2)

                # if global_rank == 0:
                #     logging.info(f"loss_main={loss_main.item()}, barlow_main={bar_main.item()}, off_main={off_main.item()}")

                loss_mix = 0.0
                if config.get('apply_mixup', False):
                    
                    z1 = z1.detach()  # 不再让mixup那部分对z1回传梯度
                    z2 = z2.detach()
                    
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
                    loss_mix = config['mixup_lambda'] * config['barlow_lambda'] * (diff_a + diff_b)

                total_loss = loss_main + loss_mix
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()

                if global_rank == 0:
                    logging.info("Optimizer step done.")

            batch_sz = s2_aug1.size(0)
            examples += batch_sz

            # 关键改动：在进入 barrier 之前先做一次 GPU 同步
            if DEBUG_DIST_BARRIER:
                logging.info(f"[Rank={global_rank}] synchronizing device before barrier (step_in_loader={step_in_loader}, step={step})")
                torch.xpu.synchronize(device)  # 显式 GPU 同步

                logging.info(f"[Rank={global_rank}] entering dist.barrier() after step_in_loader={step_in_loader}, step={step}")
                dist.barrier()
                logging.info(f"[Rank={global_rank}] passed dist.barrier() after step_in_loader={step_in_loader}, step={step}")

            # 以下是你注释掉的 log_interval 和 wandb 逻辑，这里不删除，只保持原样注释
            if (step % config['log_interval_steps'] == 0) and (global_rank == 0):
                current_time = time.time()
                exps = (examples - last_examples) / (current_time - last_time + 1e-9)
                last_time = current_time
                last_examples = examples
                rolling_loss.append(loss_main.item())
                if len(rolling_loss) > 40:
                    rolling_loss = rolling_loss[-40:]
                avg_loss = sum(rolling_loss)/len(rolling_loss)
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
                    f"Loss={loss_main.item():.4f}, MixLoss={loss_mix:.4f}, AvgLoss={avg_loss:.4f}, "
                    f"LR={current_lr:.6f}, batchsize={batch_sz}, Examples/sec={exps:.2f}, "
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
                    if ENABLE_WANDB:
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

                if ENABLE_WANDB:
                    wandb.log(wandb_dict, step=step)

            # 验证 + 保存best
            if (config['val_interval_steps'] > 0) and (step > 0) and (step % config['val_interval_steps'] == 0) and (global_rank == 0):
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
                    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
                    model.eval()
                    val_acc = linear_probe_evaluate(model.module, val_loader, device=device)
                    if ENABLE_WANDB:
                        wandb.log({"val_acc": val_acc}, step=step)
                    logging.info(f"Validation at step {step}: val_acc={val_acc:.4f}")
                    global best_val_acc
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_checkpoint(model.module, optimizer, epoch, step, best_val_acc, best_ckpt_path)
                    model.train()

            step += 1  # 全局step

        # 训练完这个 chunk
        end_t_chunk = time.time()
        if global_rank == 0:
            logging.info(f"_train_one_chunk done: chunk_files={chunk_files}, took={end_t_chunk - start_t_chunk:.2f}s")

        try:
            # 强行 shutdown dataloader
            train_loader._iterator._shutdown_workers()
        except:
            pass
        del train_loader
        del train_sampler
        del chunk_dataset
        gc.collect()

        return step, examples, last_time, last_examples, rolling_loss

    ############################################################################
    # 准备一些全局变量
    ############################################################################
    step = 0
    examples = 0
    last_time = time.time()
    last_examples = 0
    rolling_loss = []
    global best_val_acc
    best_val_acc = 0.0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_{timestamp}.pth")

    # rank=0 先打印一下注意事项
    if global_rank == 0:
        logging.info("==== Starting training loop ====")

    ############################################################################
    # ============== 开始多个 EPOCH ==============
    ############################################################################
    for epoch in range(config['epochs']):
        # 假设数据是前面Rust生成好的，这里不再生成
        if global_rank == 0:
            aug1_dir = os.path.join(config['data_root'], 'aug1')
            aug2_dir = os.path.join(config['data_root'], 'aug2')
            remove_dir(aug1_dir)
            remove_dir(aug2_dir)
            logging.info(f"Epoch {epoch} started. Generating new training data via rust_cmd...")
            subprocess.run(config['rust_cmd'], shell=True, check=True)
            logging.info(f"Data generation finished for epoch {epoch}.")

        dist.barrier()

        # 读取到 aug1/s2 下的所有文件名（假设相同数量文件）
        aug1_s2_dir = os.path.join(config['data_root'], 'aug1', 's2')
        if global_rank == 0:
            s2_file_names = os.listdir(aug1_s2_dir)
            s2_file_names.sort()
            np.random.shuffle(s2_file_names)  # file-level shuffle
            # 将文件名转成 bytes 再合并成一个大 tensor 进行广播
            max_len = 256
            fn_bytes = []
            for fn in s2_file_names:
                encoded = fn.encode('utf-8')
                padded = encoded + b'\0' * (max_len - len(encoded))
                fn_bytes.append(padded)
            all_bytes = b''.join(fn_bytes)
            file_tensor = torch.ByteTensor(list(all_bytes)).to(device)
            num_files = len(s2_file_names)
        else:
            file_tensor = torch.ByteTensor([]).to(device)
            num_files = 0

        # 广播文件总数
        num_files_tensor = torch.tensor([num_files], dtype=torch.int32, device=device)
        dist.broadcast(num_files_tensor, src=0)
        num_files = num_files_tensor.item()

        if global_rank != 0:
            max_len = 256
            file_tensor = torch.empty(num_files * max_len, dtype=torch.uint8, device=device)

        dist.broadcast(file_tensor, src=0)

        if global_rank != 0:
            array_bytes = file_tensor.cpu().numpy().tobytes()
            s2_file_names = []
            max_len = 256
            for i in range(num_files):
                start_idx = i * max_len
                end_idx = (i+1) * max_len
                raw_name = array_bytes[start_idx:end_idx].rstrip(b'\0')
                s2_file_names.append(raw_name.decode('utf-8'))

        total_files = len(s2_file_names)
        if global_rank == 0:
            logging.info(f"Epoch {epoch}: total new files = {total_files} in each folder")

        # ============ 分块加载训练 ============
        chunk_start = 0
        while chunk_start < total_files:
            chunk_end = min(chunk_start + chunk_batch, total_files)
            chunk_files = s2_file_names[chunk_start:chunk_end]

            if global_rank == 0:
                logging.info(f"Epoch={epoch}, chunk [{chunk_start}:{chunk_end}], #files={len(chunk_files)}")

            step, examples, last_time, last_examples, rolling_loss = _train_one_chunk(
                chunk_files, epoch, step, examples, last_time, last_examples, rolling_loss
            )

            chunk_start = chunk_end

        if global_rank == 0:
            logging.info(f"Epoch {epoch} finished, current step={step}")

    # 训练结束
    if global_rank == 0:
        logging.info("Training completed.")
        if ENABLE_WANDB and wandb_run is not None:
            wandb_run.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
