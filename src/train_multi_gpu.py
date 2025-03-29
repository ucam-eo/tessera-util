#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 运行方式:
#   torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 src/train_multi_gpu.py

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

import numpy as np
import torch
# 这行可根据需要设定矩阵乘法的精度
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

import torch.amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # 避免 torch.compile 的错误

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
# 2) 主训练脚本 (多GPU + chunk-based loading)
##########################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training (Multi-GPU + chunk-based loading)")
    parser.add_argument('--config', type=str, default="configs/ssl_config.py",
                        help="Path to config file (e.g. configs/ssl_config.py)")
    return parser.parse_args()


def main():
    # 设置初始化种子
    torch.manual_seed(3407)
    np.random.seed(3407)

    ########################################################################
    # 0) 初始化分布式环境 (使用 PyTorch 内置的 torchrun + DDP)
    ########################################################################
    dist.init_process_group(backend='nccl')  # 在 AMD ROCm 上通常也使用 NCCL (其实底层是 RCCL)

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 由 torchrun 传进来的本地进程索引
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

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
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        logging.info(f"Running on {world_size} GPU(s). LocalRank={local_rank}, device={device}")
    
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
    run_name = f"BT_Iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if local_rank == 0:
        wandb_run = wandb.init(project="btfm-iterable", name=run_name, config=config)
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

    model = MultimodalBTModel(
        s2_enc, s1_enc, projector,
        fusion_method=config['fusion_method'],
        return_repr=True,
        latent_dim=config['latent_dim']
    ).to(device)

    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])
    
    
    # with attn.sdpa_kernel(attn.SDPBackend.MATH):
    with (attn.sdpa_kernel(attn.SDPBackend.MATH) if config.get('use_torch_compile', False) else nullcontext()):
        if config.get('use_torch_compile', False):
            # 可以根据需求加上一些额外选项，比如 mode="max-autotune"
            # TODO: this is problematic
            # model = torch.compile(model, mode="default", fullgraph=True, dynamic=True)
            model = torch.compile(model, mode="default")
            # model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
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

        # 使用新的 GradScaler 写法
        scaler = torch.amp.GradScaler("cuda") if apply_amp else None

        # 分布式封装
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # broadcast_buffers=False,
            # find_unused_parameters=True
        )
        
        if local_rank == 0:
            wandb.watch(model.module, log="gradients", log_freq=400)

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

                # 学习率调度
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
                    # if grad_norm > 2.0 and local_rank == 0:
                    #     logging.info(f"Clipped gradient from {grad_norm:.2f} to 2.0")
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
                    # if grad_norm > 2.0 and local_rank == 0:
                    #     logging.info(f"Clipped gradient from {grad_norm:.2f} to 2.0")
                    
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

                    # 定期画一下相关系数
                    # if step % (10 * config['log_interval_steps']) == 0:
                    #     try:
                    #         fig_cc = plot_cross_corr(z1, z2)
                    #         cross_corr_img = wandb.Image(fig_cc)
                    #         plt.close(fig_cc)
                    #         wandb_dict["cross_corr"] = cross_corr_img
                    #     except:
                    #         pass
                    #     try:
                    #         fig_cc_repr = plot_cross_corr(repr1, repr2)
                    #         cross_corr_img_repr = wandb.Image(fig_cc_repr)
                    #         plt.close(fig_cc_repr)
                    #         wandb_dict["cross_corr_repr"] = cross_corr_img_repr
                    #     except:
                    #         pass

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
                        # logging.info(f"Confusion Matrix:\n{val_cm}")
                        wandb.log({"classification_report_linear": wandb.Html("<pre>" + val_report + "</pre>")}, step=step)
                        logging.info(f"Classification Report:\n{val_report}")
                        
                        # RF探针
                        # val_acc, val_f1, val_cm, val_report = rf_probe_evaluate(model, val_loader, device=device)
                        # wandb.log({"val_acc_rf": val_acc, "val_f1_rf": val_f1}, step=step)
                        # logging.info(f"Validation at step {step}: val_acc_rf={val_acc:.4f}, F1 Score_rf={val_f1:.4f}")
                        # # logging.info(f"Confusion Matrix_rf:\n{val_cm}")
                        # wandb.log({"classification_report_rf": wandb.Html("<pre>" + val_report + "</pre>")}, step=step)
                        # logging.info(f"Classification Report_rf:\n{val_report}")

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
                            save_checkpoint(model.module, optimizer, epoch, step, best_val_acc, best_ckpt_path)
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
        best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_{timestamp}.pt")

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

        if local_rank == 0:
            logging.info("Training completed.")
            wandb_run.finish()

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
