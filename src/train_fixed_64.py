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

import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import wandb
import matplotlib.pyplot as plt

from datasets.ssl_dataset import HDF5Dataset_Multimodal_Tiles_Iterable_64_Fixed, AustrianCropValidation_64_Fixed
from models.modules import TransformerEncoder_64_Fixed, ProjectionHead
from models.ssl_model import MultimodalBTModel_64_Fixed, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate
from utils.metrics import linear_probe_evaluate_64_fixed, rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr

def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training with 64 timesteps")
    parser.add_argument('--config', type=str, default="configs/ssl_config_64_fixed.py", help="Path to config file")
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

    run_name = f"BT_64_Fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_run = wandb.init(project="btfm-64-fixed", name=run_name, config=config)
    
    # Save source code
    artifact = wandb.Artifact('source-code', type='code')
    artifact.add_file('src/train_fixed_64.py')
    artifact.add_file('src/datasets/ssl_dataset.py')
    artifact.add_file('src/models/modules.py')
    artifact.add_file('src/models/ssl_model.py')
    artifact.add_file('src/utils/lr_scheduler.py')
    artifact.add_file('src/utils/metrics.py')
    artifact.add_file('src/utils/misc.py')
    artifact.add_file('configs/ssl_config_64_fixed.py')
    wandb.log_artifact(artifact)

    total_steps = config['epochs'] * config['total_samples'] // config['batch_size']
    logging.info(f"Total steps = {total_steps}")

    # Build model
    s2_num_heads = 2
    s2_num_layers = 2
    s2_dim_feedforward = 1024
    s1_num_heads = 2
    s1_num_layers = 2
    s1_dim_feedforward = 1024
    
    # Sync to wandb
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
    
    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])

    logging.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # Optimizer setup
    weight_params = [p for n, p in model.named_parameters() if p.ndim > 1]
    bias_params = [p for n, p in model.named_parameters() if p.ndim == 1]
    optimizer = torch.optim.AdamW([{'params': weight_params}, {'params': bias_params}],
                              lr=config['learning_rate'], weight_decay=1e-6)
    
    # AMP setup
    if config.get('apply_amp', False):
        scaler = amp.GradScaler()
    else:
        scaler = None

    # Training state variables
    step = 0
    examples = 0
    last_time = time.time()
    last_examples = 0
    rolling_loss = []
    rolling_size = 40
    best_val_acc = 0.0
    
    # Checkpoint path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_64_fixed_{timestamp}.pt")
    
    # Load field IDs for validation
    field_ids = None
    if 'field_id_path' in config and os.path.exists(config['field_id_path']):
        logging.info(f"Loading field IDs from {config['field_id_path']}")
        field_ids = np.load(config['field_id_path'])
        logging.info(f"Field IDs loaded, shape: {field_ids.shape}")

    for epoch in range(config['epochs']):
        # Create dataset
        dataset_train = HDF5Dataset_Multimodal_Tiles_Iterable_64_Fixed(
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
            s2_aug1 = batch_data['s2_aug1'].to(device, non_blocking=True)
            s2_aug2 = batch_data['s2_aug2'].to(device, non_blocking=True)
            s1_aug1 = batch_data['s1_aug1'].to(device, non_blocking=True)
            s1_aug2 = batch_data['s1_aug2'].to(device, non_blocking=True)
            s2_mask = batch_data['s2_mask'].to(device, non_blocking=True)
            s1_mask = batch_data['s1_mask'].to(device, non_blocking=True)

            adjust_learning_rate(optimizer, step, total_steps, config['learning_rate'],
                                config['warmup_ratio'], config['plateau_ratio'])
            optimizer.zero_grad()
            
            # Forward pass with AMP if enabled
            with (amp.autocast() if config.get('apply_amp', False) else nullcontext()):
                z1, repr1 = model(s2_aug1, s1_aug1, s2_mask, s1_mask)
                z2, repr2 = model(s2_aug2, s1_aug2, s2_mask, s1_mask)
                loss_main, bar_main, off_main = criterion(z1, z2)
                
                loss_mix = 0.0
                if config['apply_mixup']:
                    B = s2_aug1.size(0)
                    idxs = torch.randperm(B, device=device)
                    alpha = torch.distributions.Beta(config['beta_alpha'], config['beta_beta']).sample().to(device)
                    y_m_s2 = alpha * s2_aug1 + (1 - alpha) * s2_aug2[idxs, :]
                    y_m_s1 = alpha * s1_aug1 + (1 - alpha) * s1_aug2[idxs, :]
                    z_m, _ = model(y_m_s2, y_m_s1, s2_mask, s1_mask)
                    
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
            
            # Backward pass with AMP if enabled
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
            
            # Logging
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
                
                logging.info(f"[Epoch={epoch}, Step={step}] Loss={loss_main.item():.2f}, MixLoss={loss_mix:.2f}, "
                            f"AvgLoss={avg_loss:.2f}, LR={current_lr:.4f}, batchsize={s2_aug1.size(0)}, "
                            f"Examples/sec={exps:.2f}, Rank(z)={erank_z:.4f}, Rank(repr)={erank_repr:.4f}")
                
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
                
                wandb.log(wandb_dict, step=step)
            
            # Validation
            if config['val_interval_steps'] > 0 and step % config['val_interval_steps'] == 0:
                # If validation dataset paths are configured
                if all(config.get(k) for k in ['val_s2_bands_file_path', 'val_s2_masks_file_path',
                                              'val_s2_doy_file_path', 'val_s1_asc_bands_file_path',
                                              'val_s1_asc_doy_file_path', 'val_s1_desc_bands_file_path',
                                              'val_s1_desc_doy_file_path', 'val_labels_path']):
                    from torch.utils.data import DataLoader
                    
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
                    
                    # Save best model
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