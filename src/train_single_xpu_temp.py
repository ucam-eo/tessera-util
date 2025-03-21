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
import torch.amp  # 使用 torch.amp
# 删除分布式相关的 import
#from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch

import wandb
import matplotlib.pyplot as plt

# ============== 你原先用到的模块 / 函数 ===============
from models.modules import TransformerEncoder, ProjectionHead, SpectralTemporalTransformer
from models.ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate, LARS
from utils.metrics import linear_probe_evaluate, rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr

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
# 2) 主训练脚本 (单个XPU训练)
##########################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training (Single-XPU)")
    parser.add_argument('--config', type=str, default="configs/ssl_config_temp.py",
                        help="Path to config file (e.g. configs/ssl_config.py)")
    return parser.parse_args()

def main():
    # 设置初始化种子
    torch.manual_seed(3407)
    np.random.seed(3407)

    # 单机单卡设置：local_rank = 0, world_size = 1
    local_rank = 0
    world_size = 1

    # 配置日志：单设备全部输出info日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 使用整数设备索引和torch.device
    device = torch.device("xpu:0")
    torch.xpu.set_device(device)
    logging.info(f"Running on a single XPU, device={device}")

    # 解析命令行 & 读取配置
    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    # 根据配置文件中的apply_amp字段决定是否启用AMP，默认为False
    apply_amp = config.get('apply_amp', False)
    logging.info(f"apply_amp = {apply_amp}")

    # 准备W&B
    run_name = f"BT_Iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_run = wandb.init(project="btfm-iterable", name=run_name, config=config)

    # 计算总训练步数（可能并不精准，因为我们分块加载）
    total_steps = config['epochs'] * config['total_samples'] // config['batch_size'] // world_size
    logging.info(f"Total steps (approx) = {total_steps}")

    ############################################################################
    # 3) 构建模型
    ############################################################################
    s2_num_heads = 8
    s2_num_layers = 16
    s2_dim_feedforward = 512
    s1_num_heads = 8
    s1_num_layers = 16
    s1_dim_feedforward = 512

    # 同步到 wandb
    wandb.config.update({
        "s2_num_heads": s2_num_heads,
        "s2_num_layers": s2_num_layers,
        "s2_dim_feedforward": s2_dim_feedforward,
        "s1_num_heads": s1_num_heads,
        "s1_num_layers": s1_num_layers,
        "s1_dim_feedforward": s1_dim_feedforward
    })

    s2_enc = TransformerEncoder(
        band_num=12,  # 例如 10个波段+sin/cos
        latent_dim=config['latent_dim'],
        nhead=s2_num_heads,
        num_encoder_layers=s2_num_layers,
        dim_feedforward=s2_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s2']
    ).to(device)
    s1_enc = TransformerEncoder(
        band_num=4,   # 例如 2个波段+sin/cos
        latent_dim=config['latent_dim'],
        nhead=s1_num_heads,
        num_encoder_layers=s1_num_layers,
        dim_feedforward=s1_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s1']
    ).to(device)

    if config['fusion_method'] == 'concat':
        proj_in_dim = config['latent_dim'] * 2
    else:
        proj_in_dim = config['latent_dim']
    projector = ProjectionHead(proj_in_dim,
                               config['projector_hidden_dim'],
                               config['projector_out_dim']).to(device)

    if config['fusion_method'] == 'transformer':
        model = MultimodalBTModel(s2_enc, s1_enc, projector,
                                  fusion_method=config['fusion_method'],
                                  return_repr=True,
                                  latent_dim=config['latent_dim']).to(device)
    else:
        model = MultimodalBTModel(s2_enc, s1_enc, projector,
                                  fusion_method=config['fusion_method'],
                                  return_repr=True).to(device)

    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {num_params} trainable parameters.")

    weight_params = [p for n, p in model.named_parameters() if p.ndim > 1]
    bias_params   = [p for n, p in model.named_parameters() if p.ndim == 1]
    param_lrs = [{'params': weight_params}, {'params': bias_params}]

    logging.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    
    optimizer = torch.optim.AdamW([{'params': weight_params}, {'params': bias_params}],
                                  lr=config['learning_rate'], weight_decay=1e-6)

    # 根据apply_amp配置决定是否使用AMP
    if apply_amp:
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    # IPEX优化
    model, optimizer = ipex.optimize(model, optimizer=optimizer,
                                     dtype=torch.float32,
                                     inplace=False)
    # 单卡训练，无需 DDP 封装

    ############################################################################
    # 4) 训练循环 (chunk-based) — 分块加载训练
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
        针对给定的一批文件(chunk_files)，构造ChunkDataset和DataLoader，
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
        # 单卡训练，无需DistributedSampler，直接使用DataLoader
        train_loader = DataLoader(
            chunk_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=True,
            pin_memory=True,
            persistent_workers=False
        )

        logging.info(f"   => This chunk has {len(chunk_dataset)} samples total, steps in dataloader = {len(train_loader)}.")

        model.train()

        # 在这一批文件上进行训练
        for batch_data in train_loader:
            s2_aug1 = batch_data['s2_aug1'].to(device, non_blocking=True)
            s2_aug2 = batch_data['s2_aug2'].to(device, non_blocking=True)
            s1_aug1 = batch_data['s1_aug1'].to(device, non_blocking=True)
            s1_aug2 = batch_data['s1_aug2'].to(device, non_blocking=True)

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

                    # Mixup
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
                    loss_mix = config['mixup_lambda'] * config['barlow_lambda'] * (diff_a + diff_b)
                total_loss = loss_main + loss_mix
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()

            batch_sz = s2_aug1.size(0)
            examples += batch_sz

            if (step % config['log_interval_steps'] == 0):
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
            if (config['val_interval_steps'] > 0) and (step > 0) and (step % config['val_interval_steps'] == 0):
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
                    
                    # 调用新的 linear_probe_evaluate 函数计算 acc, f1 和混淆矩阵
                    val_acc, val_f1, val_cm = linear_probe_evaluate(model, val_loader, device=device)
                    wandb.log({"val_acc": val_acc, "val_f1": val_f1}, step=step)
                    logging.info(f"Validation at step {step}: val_acc={val_acc:.4f}, F1 Score={val_f1:.4f}")
                    logging.info(f"Confusion Matrix:\n{val_cm}")
                    # 绘制混淆矩阵图像，并 log 到 wandb
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
                    
                    global best_val_acc
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_checkpoint(model, optimizer, epoch, step, best_val_acc, best_ckpt_path)
                    model.train()

            step += 1  # 全局step

        # 训练完一个chunk，释放资源
        try:
            train_loader._iterator._shutdown_workers()
        except Exception:
            pass
        del train_loader
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
    
    # 读取 chunk_batch 配置
    chunk_batch = config.get('chunk_batch', 40)
    logging.info(f"chunk_batch = {chunk_batch}")

    ############################################################################
    # 开始多个 EPOCH
    ############################################################################
    for epoch in range(config['epochs']):
        # 进程调用 rust 生成数据
        aug1_dir = os.path.join(config['data_root'], 'aug1')
        aug2_dir = os.path.join(config['data_root'], 'aug2')
        remove_dir(aug1_dir)
        remove_dir(aug2_dir)
        logging.info(f"Epoch {epoch} started. Generating new training data via rust_cmd...")
        subprocess.run(config['rust_cmd'], shell=True, check=True)
        logging.info(f"Data generation finished for epoch {epoch}.")

        # 收集所有文件名（这里假设 file_names 是 S2 的 aug1）
        aug1_s2_dir = os.path.join(config['data_root'], 'aug1', 's2')
        s2_file_names = os.listdir(aug1_s2_dir)
        np.random.shuffle(s2_file_names)  # Shuffle file names randomly
        total_files = len(s2_file_names)
        logging.info(f"Epoch {epoch}: total new files = {total_files} in each folder")

        # 分块加载训练
        chunk_start = 0
        while chunk_start < total_files:
            chunk_end = min(chunk_start + chunk_batch, total_files)
            chunk_files = s2_file_names[chunk_start:chunk_end]
            logging.info(f"Epoch {epoch}, chunk [{chunk_start}:{chunk_end}], loading {len(chunk_files)} files...")

            step, examples, last_time, last_examples, rolling_loss = _train_one_chunk(
                chunk_files, epoch, step, examples, last_time, last_examples, rolling_loss
            )

            chunk_start = chunk_end

        logging.info(f"Epoch {epoch} finished, current step = {step}")

    # 训练结束
    logging.info("Training completed.")
    wandb_run.finish()
    
if __name__ == "__main__":
    main()
