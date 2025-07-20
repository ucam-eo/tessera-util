import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NUMEXPR_MAX_THREADS"] = "24"
import time
import math
import subprocess
import argparse
import logging
from datetime import datetime
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import wandb
import numpy as np

from models.modules import TransformerEncoder, ProjectionHead, SpectralTemporalTransformer
from models.ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate, ValidationBasedLRScheduler
from utils.metrics import linear_probe_evaluate, rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr
import matplotlib.pyplot as plt

from torch.utils.data import IterableDataset, get_worker_info

# Define constants
S2_BAND_MEAN = np.array([1711.0938,1308.8511,1546.4543,3010.1293,3106.5083,
                        2068.3044,2685.0845,2931.5889,2514.6928,1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026,1862.9751,1803.1792,1741.7837,1677.4543,
                        1888.7862,1736.3090,1715.8104,1514.5199,1398.4779], dtype=np.float32)
S1_BAND_MEAN = np.array([5484.0407,3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334,1726.0670], dtype=np.float32)

class HDF5Dataset_Multimodal_Tiles_Iterable(IterableDataset):
    """
    Preprocessed data loader: reads preprocessed .npy files with directory structure:
       data_root/
         ├─ aug1/
         │    ├─ s2/   -> Each .npy file has shape approx (N, 20, 12)
         │    └─ s1/   -> Each .npy file has shape approx (N, 20, 4)
         ├─ aug2/
         │    ├─ s2/
         │    └─ s1/
         └─ aug3/
              ├─ s2/
              └─ s1/
    Each file group consists of six .npy files corresponding to s2_aug1, s2_aug2, s2_aug3, 
    s1_aug1, s1_aug2, s1_aug3, yields individual samples.
    """
    def __init__(self,
                 data_root,
                 min_valid_timesteps=10,
                 sample_size_s2=20,
                 sample_size_s1=20,
                 standardize=True,
                 shuffle_tiles=False):
        super().__init__()
        self.data_root = data_root
        self.min_valid_timesteps = min_valid_timesteps
        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1
        self.standardize = standardize
        self.shuffle_tiles = shuffle_tiles

        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std = S1_BAND_STD

        # Construct data paths for all three augmentations
        self.aug1_s2_dir = os.path.join(data_root, "aug1", "s2")
        self.aug2_s2_dir = os.path.join(data_root, "aug2", "s2")
        self.aug3_s2_dir = os.path.join(data_root, "aug3", "s2")
        self.aug1_s1_dir = os.path.join(data_root, "aug1", "s1")
        self.aug2_s1_dir = os.path.join(data_root, "aug2", "s1")
        self.aug3_s1_dir = os.path.join(data_root, "aug3", "s1")
        
        # Check if all directories exist
        for d in [self.aug1_s2_dir, self.aug2_s2_dir, self.aug3_s2_dir, 
                 self.aug1_s1_dir, self.aug2_s1_dir, self.aug3_s1_dir]:
            if not os.path.exists(d):
                raise RuntimeError(f"Directory {d} not found!")
                
        # Get and shuffle filenames
        file_names = sorted(os.listdir(self.aug1_s2_dir))
        np.random.shuffle(file_names)
        self.file_groups = []
        
        # Group corresponding files from all augmentations
        for fn in file_names:
            file_path_aug1_s2 = os.path.join(self.aug1_s2_dir, fn)
            file_path_aug2_s2 = os.path.join(self.aug2_s2_dir, fn)
            file_path_aug3_s2 = os.path.join(self.aug3_s2_dir, fn)
            file_path_aug1_s1 = os.path.join(self.aug1_s1_dir, fn)
            file_path_aug2_s1 = os.path.join(self.aug2_s1_dir, fn)
            file_path_aug3_s1 = os.path.join(self.aug3_s1_dir, fn)
            
            # Check if all files exist
            if (os.path.exists(file_path_aug1_s2) and 
                os.path.exists(file_path_aug2_s2) and 
                os.path.exists(file_path_aug3_s2) and
                os.path.exists(file_path_aug1_s1) and 
                os.path.exists(file_path_aug2_s1) and
                os.path.exists(file_path_aug3_s1)):
                
                self.file_groups.append({
                    "s2_aug1": file_path_aug1_s2,
                    "s2_aug2": file_path_aug2_s2,
                    "s2_aug3": file_path_aug3_s2,
                    "s1_aug1": file_path_aug1_s1,
                    "s1_aug2": file_path_aug2_s1,
                    "s1_aug3": file_path_aug3_s1
                })
                
        if len(self.file_groups) == 0:
            raise RuntimeError("No valid file groups found in preprocessed dataset!")

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            groups_to_process = self.file_groups
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(self.file_groups) / num_workers))
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_groups))
            groups_to_process = self.file_groups[start:end]
            
        if self.shuffle_tiles:
            np.random.shuffle(groups_to_process)
            
        for group in groups_to_process:
            # Load all six files (three augmentations for both S1 and S2)
            s2_aug1_array = np.load(group["s2_aug1"]) # (N, 20, 11), 样本数目，时间，波段
            s2_aug2_array = np.load(group["s2_aug2"])
            s2_aug3_array = np.load(group["s2_aug3"])
            s1_aug1_array = np.load(group["s1_aug1"]) # (N, 20, 3)
            s1_aug2_array = np.load(group["s1_aug2"])
            s1_aug3_array = np.load(group["s1_aug3"])
            # 沿着时间维度进行shuffle
            s2_aug1_array = s2_aug1_array[:, np.random.permutation(s2_aug1_array.shape[1])]
            s2_aug2_array = s2_aug2_array[:, np.random.permutation(s2_aug2_array.shape[1])]
            s2_aug3_array = s2_aug3_array[:, np.random.permutation(s2_aug3_array.shape[1])]
            s1_aug1_array = s1_aug1_array[:, np.random.permutation(s1_aug1_array.shape[1])]
            s1_aug2_array = s1_aug2_array[:, np.random.permutation(s1_aug2_array.shape[1])]
            s1_aug3_array = s1_aug3_array[:, np.random.permutation(s1_aug3_array.shape[1])]
            # 全部转为torch tensor
            s2_aug1_array = torch.tensor(s2_aug1_array, dtype=torch.float32)
            s2_aug2_array = torch.tensor(s2_aug2_array, dtype=torch.float32)
            s2_aug3_array = torch.tensor(s2_aug3_array, dtype=torch.float32)
            s1_aug1_array = torch.tensor(s1_aug1_array, dtype=torch.float32)
            s1_aug2_array = torch.tensor(s1_aug2_array, dtype=torch.float32)
            s1_aug3_array = torch.tensor(s1_aug3_array, dtype=torch.float32)
            
            def augment_spectral_and_temporal(x, augment_method = ['fft', 'time_drop', 'band_mask']):
                """
                x: 输入张量 (B, T, N), 最后一列为时间列
                freq_keep_ratio: 保留低频成分的比例
                drop_prob: 时间步丢弃概率
                """
                B, T, N = x.shape
                
                # 分离数据和时序列
                data = x[..., :-1]  # (B, T, N-1)
                time = x[..., -1:]  # (B, T, 1)
                
                # 增强1: 傅里叶低通滤波
                def spectral_augment(data, time, time_dim=1, freq_keep_ratio=0.7):
                    """
                    支持无序时序的傅里叶增强
                    data: 波段数据 (B, T, C)
                    time: 时间列 (B, T, 1)
                    """
                    B, T, C = data.shape
                    
                    # 按时间排序（每个样本独立排序）
                    # 获取排序索引 (B, T)
                    _, sorted_indices = torch.sort(time.squeeze(-1), dim=time_dim)
                    
                    # 重排数据为时间有序
                    sorted_data = torch.gather(
                        data, 
                        time_dim, 
                        sorted_indices.unsqueeze(-1).expand(-1, -1, C)
                    )  # (B, T, C)
                    
                    # 傅里叶变换
                    fft = torch.fft.rfft(sorted_data, dim=time_dim)  # (B, T//2+1, C)
                    
                    # 创建低频掩码
                    num_freq = fft.shape[time_dim]
                    keep = int(num_freq * freq_keep_ratio)
                    mask = torch.zeros_like(fft)
                    mask[:, :keep] = 1
                    
                    # 频率过滤与逆变换
                    filtered_sorted = torch.fft.irfft(fft * mask, n=T, dim=time_dim)  # (B, T, C)
                    
                    # 恢复原始时序顺序
                    original_order = torch.argsort(sorted_indices, dim=time_dim)  # (B, T)
                    restored_data = torch.gather(
                        filtered_sorted,
                        time_dim,
                        original_order.unsqueeze(-1).expand(-1, -1, C)
                    )
                    
                    return restored_data

                # 增强2: 时序随机丢弃与复制（改进版）
                def temporal_augment(data, time, drop_prob=0.3):
                    # 生成丢弃掩码 (B, T)
                    drop_mask = torch.rand(B, T) < drop_prob
                    
                    # 创建替换索引矩阵（核心优化）
                    replace_idx = torch.arange(T).expand(B, T).clone()  # (B, T)
                    
                    # 批量生成有效索引（向量化操作）
                    valid_masks = ~drop_mask
                    for b in range(B):
                        valid_indices = valid_masks[b].nonzero().squeeze()
                        num_valid = valid_indices.size(0)
                        num_drop = T - num_valid
                        
                        if num_drop > 0 and num_valid > 0:
                            # 生成替换索引（允许重复）
                            selected = valid_indices[torch.randint(0, num_valid, (num_drop,))]
                            replace_idx[b, drop_mask[b]] = selected
                    
                    # 同时替换数据和时问
                    new_data = torch.gather(data, 1, replace_idx.unsqueeze(-1).expand(-1, -1, data.shape[-1]))
                    new_time = torch.gather(time, 1, replace_idx.unsqueeze(-1).expand(-1, -1, 1))
                    return new_data, new_time
                
                def band_masking(data, mask_ratio=0.2):
                    # 生成随机掩码 (每个样本独立掩码不同波段)
                    mask = torch.rand(B, 1, data.shape[-1], device=data.device) > mask_ratio
                    masked_data = data * mask
                    
                    return masked_data
                
                # 应用增强组合
                if 'fft' in augment_method:
                    data = spectral_augment(data, time)
                if 'time_drop' in augment_method:
                    data, time = temporal_augment(data, time)  # 同时处理时问
                if 'band_mask' in augment_method:
                    data = band_masking(data)
                
                return torch.cat([data, time], dim=-1)  # (B, T, N)
            
            # 分别应用三种增强
            # 对于aug1进行fft
            s2_aug1_array = augment_spectral_and_temporal(s2_aug1_array, augment_method=['fft'])
            s1_aug1_array = augment_spectral_and_temporal(s1_aug1_array, augment_method=['fft'])
            # 对于aug2进行time_drop
            s2_aug2_array = augment_spectral_and_temporal(s2_aug2_array, augment_method=['time_drop'])
            s1_aug2_array = augment_spectral_and_temporal(s1_aug2_array, augment_method=['time_drop'])
            # 对于aug3进行band_mask
            s2_aug3_array = augment_spectral_and_temporal(s2_aug3_array, augment_method=['band_mask'])
            s1_aug3_array = augment_spectral_and_temporal(s1_aug3_array, augment_method=['band_mask'])
            
            # Find the minimum number of samples across all files
            n_samples = min(s2_aug1_array.shape[0],
                          s2_aug2_array.shape[0],
                          s2_aug3_array.shape[0],
                          s1_aug1_array.shape[0],
                          s1_aug2_array.shape[0],
                          s1_aug3_array.shape[0])
            
            for i in range(n_samples):
                yield {
                    "s2_aug1": s2_aug1_array[i],
                    "s2_aug2": s2_aug2_array[i],
                    "s2_aug3": s2_aug3_array[i],
                    "s1_aug1": s1_aug1_array[i],
                    "s1_aug2": s1_aug2_array[i],
                    "s1_aug3": s1_aug3_array[i]
                }

def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training")
    parser.add_argument('--config', type=str, default="configs/ssl_config.py", help="Path to config file (e.g. configs/ssl_config.py)")
    return parser.parse_args()

def main():
    args_cli = parse_args()
    # Load configuration
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Initialize wandb with "Barlow Siblings" run name
    run_name = f"BS_Iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_run = wandb.init(project="btfm-iterable-temp", name=run_name, config=config)
    
    # Upload source code
    artifact = wandb.Artifact('source-code', type='code')
    artifact.add_file('src/train_sibling.py')
    artifact.add_file('src/datasets/ssl_dataset.py')
    artifact.add_file('src/models/modules.py')
    artifact.add_file('src/models/ssl_model.py')
    artifact.add_file('src/utils/lr_scheduler.py')
    artifact.add_file('src/utils/metrics.py')
    artifact.add_file('src/utils/misc.py')
    artifact.add_file('configs/ssl_config.py')
    wandb.log_artifact(artifact)

    total_steps = config['epochs'] * config['total_samples'] // config['batch_size']
    logging.info(f"Total steps = {total_steps}")

    # Build model
    s2_num_heads = 16
    s2_num_layers = 4
    s2_dim_feedforward = 1024
    s1_num_heads = 16
    s1_num_layers = 4
    s1_dim_feedforward = 1024
    
    # Sync to wandb
    wandb.config.update({
        "s2_num_heads": s2_num_heads,
        "s2_num_layers": s2_num_layers,
        "s2_dim_feedforward": s2_dim_feedforward,
        "s1_num_heads": s1_num_heads,
        "s1_num_layers": s1_num_layers,
        "s1_dim_feedforward": s1_dim_feedforward,
        "training_method": "Barlow Siblings"  # Add this to indicate we're using the siblings approach
    })
    
    s2_enc = TransformerEncoder(
        band_num=10,
        latent_dim=config['latent_dim'],
        nhead=s2_num_heads,
        num_encoder_layers=s2_num_layers,
        dim_feedforward=s2_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s2']
    ).to(device)
    
    s1_enc = TransformerEncoder(
        band_num=2,
        latent_dim=config['latent_dim'],
        nhead=s1_num_heads,
        num_encoder_layers=s1_num_layers,
        dim_feedforward=s1_dim_feedforward,
        dropout=0.1,
        max_seq_len=config['sample_size_s1']
    ).to(device)
    
    if config['fusion_method'] == 'concat':
        proj_in_dim = config['latent_dim']
    else:
        proj_in_dim = config['latent_dim']
        
    projector = ProjectionHead(proj_in_dim, config['projector_hidden_dim'], config['projector_out_dim']).to(device)
    
    if config['fusion_method'] == 'transformer':
        model = MultimodalBTModel(s2_enc, s1_enc, projector, fusion_method=config['fusion_method'], return_repr=True, latent_dim=config['latent_dim']).to(device)
    else:
        model = MultimodalBTModel(s2_enc, s1_enc, projector, fusion_method=config['fusion_method'], return_repr=True).to(device)
        
    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])

    logging.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    weight_params = [p for n, p in model.named_parameters() if p.ndim > 1]
    bias_params = [p for n, p in model.named_parameters() if p.ndim == 1]
    
    optimizer = torch.optim.AdamW([{'params': weight_params}, {'params': bias_params}],
                               lr=config['learning_rate'], weight_decay=1e-6)
    
    # Control whether to use AMP based on config
    if config.get('apply_amp', False):
        scaler = amp.GradScaler()
    else:
        scaler = None

    step = 0
    examples = 0
    last_time = time.time()
    last_examples = 0
    rolling_loss = []
    rolling_size = 40
    best_val_acc = 0.0
    
    # Get timestamp for checkpoint
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_{timestamp}.pt")
    
    # Initialize the validation-based LR scheduler
    lr_scheduler = ValidationBasedLRScheduler(
        base_lr=config['learning_rate'],
        total_steps=total_steps,
        warmup_ratio=config['warmup_ratio'],
        patience=config.get('lr_patience', 5),  # Default to 5 if not specified in config
        reduction_factor=config.get('lr_reduction_factor', 0.5),  # Default to 0.5 if not specified
        min_lr=config.get('min_lr', 1e-6),  # Default to 1e-6 if not specified
        weight_factor=0.2,
        bias_factor=0.0048
    )

    for epoch in range(config['epochs']):
        # Generate new data code was commented out in the original

        # Create dataset with three augmentations
        dataset_train = HDF5Dataset_Multimodal_Tiles_Iterable(
            data_root=config['data_root'],
            min_valid_timesteps=config['min_valid_timesteps'],
            sample_size_s2=config['sample_size_s2'],
            sample_size_s1=config['sample_size_s1'],
            standardize=True,
            shuffle_tiles=config['shuffle_tiles']
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            drop_last=True,
        )
        
        model.train()
        for batch_data in train_loader:
            # Load all six tensors (three augmentations for both S1 and S2)
            s2_aug1 = batch_data['s2_aug1'].to(device, non_blocking=True)
            s2_aug2 = batch_data['s2_aug2'].to(device, non_blocking=True)
            s2_aug3 = batch_data['s2_aug3'].to(device, non_blocking=True)
            s1_aug1 = batch_data['s1_aug1'].to(device, non_blocking=True)
            s1_aug2 = batch_data['s1_aug2'].to(device, non_blocking=True)
            s1_aug3 = batch_data['s1_aug3'].to(device, non_blocking=True)

            # adjust_learning_rate(optimizer, step, total_steps, config['learning_rate'],
            #                      config['warmup_ratio'], config['plateau_ratio'])
            current_lr = lr_scheduler.adjust_learning_rate(optimizer, step)
            optimizer.zero_grad()
            
            # Use apply_amp parameter to decide whether to enable automatic mixed precision
            with (amp.autocast() if config.get('apply_amp', False) else nullcontext()):
                # Forward pass for all three augmentations
                z1, repr1 = model(s2_aug1, s1_aug1)  # aug1
                z2, repr2 = model(s2_aug2, s1_aug2)  # aug2
                z3, repr3 = model(s2_aug3, s1_aug3)  # aug3
                
                # Calculate Barlow Twins losses for all three pairs
                loss_12, bar_12, off_12 = criterion(z1, z2)  # aug1-aug2
                loss_23, bar_23, off_23 = criterion(z2, z3)  # aug2-aug3
                loss_13, bar_13, off_13 = criterion(z1, z3)  # aug1-aug3
                
                # Main Barlow Siblings loss is the sum of the three pairwise losses
                loss_main = loss_12 + loss_23 + loss_13
                loss_main = loss_main / 3.0
                
                # Initialize mixup loss
                loss_mix = 0.0
                
                # Apply mixup if configured
                if config['apply_mixup']:
                    B = s2_aug1.size(0)
                    idxs = torch.randperm(B, device=device)
                    alpha = torch.distributions.Beta(config['beta_alpha'], config['beta_beta']).sample().to(device)
                    
                    # Mixup for pair 1-2
                    y_m_s2_12 = alpha * s2_aug1 + (1 - alpha) * s2_aug2[idxs, :]
                    y_m_s1_12 = alpha * s1_aug1 + (1 - alpha) * s1_aug2[idxs, :]
                    z_m_12, _ = model(y_m_s2_12, y_m_s1_12)
                    
                    cc_m_a_12 = compute_cross_correlation(z_m_12, z1)
                    cc_m_b_12 = compute_cross_correlation(z_m_12, z2)
                    cc_z1_z1 = compute_cross_correlation(z1, z1)
                    cc_z2idx_z1 = compute_cross_correlation(z2[idxs], z1)
                    cc_z1_z2 = compute_cross_correlation(z1, z2)
                    cc_z2idx_z2 = compute_cross_correlation(z2[idxs], z2)
                    cc_m_a_gt_12 = alpha * cc_z1_z1 + (1 - alpha) * cc_z2idx_z1
                    cc_m_b_gt_12 = alpha * cc_z1_z2 + (1 - alpha) * cc_z2idx_z2
                    diff_a_12 = (cc_m_a_12 - cc_m_a_gt_12).pow(2).sum()
                    diff_b_12 = (cc_m_b_12 - cc_m_b_gt_12).pow(2).sum()
                    loss_mix_12 = config['mixup_lambda'] * config['barlow_lambda'] * (diff_a_12 + diff_b_12)
                    
                    # Mixup for pair 2-3
                    alpha = torch.distributions.Beta(config['beta_alpha'], config['beta_beta']).sample().to(device)
                    y_m_s2_23 = alpha * s2_aug2 + (1 - alpha) * s2_aug3[idxs, :]
                    y_m_s1_23 = alpha * s1_aug2 + (1 - alpha) * s1_aug3[idxs, :]
                    z_m_23, _ = model(y_m_s2_23, y_m_s1_23)
                    
                    cc_m_a_23 = compute_cross_correlation(z_m_23, z2)
                    cc_m_b_23 = compute_cross_correlation(z_m_23, z3)
                    cc_z2_z2 = compute_cross_correlation(z2, z2)
                    cc_z3idx_z2 = compute_cross_correlation(z3[idxs], z2)
                    cc_z2_z3 = compute_cross_correlation(z2, z3)
                    cc_z3idx_z3 = compute_cross_correlation(z3[idxs], z3)
                    cc_m_a_gt_23 = alpha * cc_z2_z2 + (1 - alpha) * cc_z3idx_z2
                    cc_m_b_gt_23 = alpha * cc_z2_z3 + (1 - alpha) * cc_z3idx_z3
                    diff_a_23 = (cc_m_a_23 - cc_m_a_gt_23).pow(2).sum()
                    diff_b_23 = (cc_m_b_23 - cc_m_b_gt_23).pow(2).sum()
                    loss_mix_23 = config['mixup_lambda'] * config['barlow_lambda'] * (diff_a_23 + diff_b_23)
                    
                    # Mixup for pair 1-3
                    alpha = torch.distributions.Beta(config['beta_alpha'], config['beta_beta']).sample().to(device)
                    y_m_s2_13 = alpha * s2_aug1 + (1 - alpha) * s2_aug3[idxs, :]
                    y_m_s1_13 = alpha * s1_aug1 + (1 - alpha) * s1_aug3[idxs, :]
                    z_m_13, _ = model(y_m_s2_13, y_m_s1_13)
                    
                    cc_m_a_13 = compute_cross_correlation(z_m_13, z1)
                    cc_m_b_13 = compute_cross_correlation(z_m_13, z3)
                    # We already computed cc_z1_z1 above
                    cc_z3idx_z1 = compute_cross_correlation(z3[idxs], z1)
                    cc_z1_z3 = compute_cross_correlation(z1, z3)
                    cc_z3idx_z3 = compute_cross_correlation(z3[idxs], z3)
                    cc_m_a_gt_13 = alpha * cc_z1_z1 + (1 - alpha) * cc_z3idx_z1
                    cc_m_b_gt_13 = alpha * cc_z1_z3 + (1 - alpha) * cc_z3idx_z3
                    diff_a_13 = (cc_m_a_13 - cc_m_a_gt_13).pow(2).sum()
                    diff_b_13 = (cc_m_b_13 - cc_m_b_gt_13).pow(2).sum()
                    loss_mix_13 = config['mixup_lambda'] * config['barlow_lambda'] * (diff_a_13 + diff_b_13)
                    
                    # Total mixup loss is the sum of the three pairwise mixup losses
                    loss_mix = loss_mix_12 + loss_mix_23 + loss_mix_13
                    loss_mix = loss_mix / 3.0
                    
                # Total loss is the sum of the main Barlow Siblings loss and the mixup loss
                total_loss = loss_main + loss_mix
                
            if config.get('apply_amp', False):
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()
                
            examples += s2_aug1.size(0)
            if step % config['log_interval_steps'] == 0:
                current_time = time.time()
                exps = (examples - last_examples) / (current_time - last_time)
                last_time = current_time
                last_examples = examples
                rolling_loss.append(loss_main.item())
                if len(rolling_loss) > rolling_size:
                    rolling_loss = rolling_loss[-rolling_size:]
                avg_loss = sum(rolling_loss) / len(rolling_loss)
                current_lr = optimizer.param_groups[0]['lr']
                erank_z = rankme(z1)
                erank_repr = rankme(repr1)
                
                # Log detailed information about the different loss components
                logging.info(f"[Epoch={epoch}, Step={step}] Loss={loss_main.item():.2f}, "
                             f"Loss12={loss_12.item():.2f}, Loss23={loss_23.item():.2f}, Loss13={loss_13.item():.2f}, "
                             f"MixLoss={loss_mix:.2f}, AvgLoss={avg_loss:.2f}, LR={current_lr:.4f}, "
                             f"batchsize={s2_aug1.size(0)}, Examples/sec={exps:.2f}, "
                             f"Rank(z)={erank_z:.4f}, Rank(repr)={erank_repr:.4f}")
                
                wandb_dict = {
                    "epoch": epoch,
                    "loss_main": loss_main.item(),
                    "loss_12": loss_12.item(),
                    "loss_23": loss_23.item(),
                    "loss_13": loss_13.item(),
                    "mix_loss": loss_mix,
                    "avg_loss": avg_loss,
                    "lr": current_lr,
                    "examples/sec": exps,
                    "total_loss": total_loss.item(),
                    "rank_z": erank_z,
                    "rank_repr": erank_repr,
                }
                
                cross_corr_img = None
                cross_corr_img_repr = None
                cross_corr_img_12 = None
                cross_corr_img_23 = None
                cross_corr_img_13 = None
                
                # if step % (30 * config['log_interval_steps']) == 0:
                #     try:
                #         # Log cross-correlation matrices for all pairs
                #         fig_cc_12 = plot_cross_corr(z1, z2)
                #         cross_corr_img_12 = wandb.Image(fig_cc_12)
                #         plt.close(fig_cc_12)
                        
                #         # fig_cc_23 = plot_cross_corr(z2, z3)
                #         # cross_corr_img_23 = wandb.Image(fig_cc_23)
                #         # plt.close(fig_cc_23)
                        
                #         # fig_cc_13 = plot_cross_corr(z1, z3)
                #         # cross_corr_img_13 = wandb.Image(fig_cc_13)
                #         # plt.close(fig_cc_13)
                        
                #         # For backward compatibility, also log the first pair as "cross_corr"
                #         cross_corr_img = cross_corr_img_12
                        
                #         # Log representation cross-correlation for the first pair
                #         fig_cc_repr = plot_cross_corr(repr1, repr2)
                #         cross_corr_img_repr = wandb.Image(fig_cc_repr)
                #         plt.close(fig_cc_repr)
                #     except Exception:
                #         pass
                    
                # Add all images to wandb logging dictionary
                if cross_corr_img:
                    wandb_dict["cross_corr"] = cross_corr_img
                if cross_corr_img_repr:
                    wandb_dict["cross_corr_repr"] = cross_corr_img_repr
                
                wandb.log(wandb_dict, step=step)
            
            # Validation code (unchanged from original)
            if (config['val_interval_steps'] > 0) and (step % config['val_interval_steps'] == 0) and (step > 0):
                # If validation dataset paths are configured, perform validation
                if all(config.get(k) for k in ['val_s2_bands_file_path', 'val_s2_masks_file_path',
                                                'val_s2_doy_file_path', 'val_s1_asc_bands_file_path',
                                                'val_s1_asc_doy_file_path', 'val_s1_desc_bands_file_path',
                                                'val_s1_desc_doy_file_path', 'val_labels_path']):
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
                    # Call linear_probe_evaluate function to compute acc, f1, and confusion matrix
                    val_acc, val_f1, val_cm, val_cr = linear_probe_evaluate(model, val_loader, device=device)
                    wandb.log({
                        "val_acc": val_acc, 
                        "val_f1": val_f1,
                        "val_cr": wandb.Html(f"<pre>{val_cr}</pre>"),
                        }, step=step)
                    
                    # Check for plateau and possibly reduce learning rate
                    lr_reduced = lr_scheduler.check_validation_plateau(val_acc)
                    
                    # Log scheduler status
                    logging.info(f"LR Scheduler: {lr_scheduler.get_status()}")
                    
                    val_wandb_dict = {
                        "lr_reductions": lr_scheduler.lr_reductions,
                        "stagnant_count": lr_scheduler.stagnant_count
                    }
                    wandb.log(val_wandb_dict, step=step)
                    
                    logging.info(f"Validation at step {step}: val_acc={val_acc:.4f}, F1 Score={val_f1:.4f}")
                    logging.info(f"Confusion Matrix:\n{val_cm}")
                    logging.info(f"Classification Report:\n{val_cr}")
                    
                    # Plot confusion matrix and log to wandb
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
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_checkpoint(model, optimizer, epoch, step, best_val_acc, best_ckpt_path)
                    model.train()
            step += 1
        logging.info(f"Epoch {epoch} finished, current step = {step}")
    logging.info("Training completed.")
    wandb_run.finish()

if __name__ == "__main__":
    main()