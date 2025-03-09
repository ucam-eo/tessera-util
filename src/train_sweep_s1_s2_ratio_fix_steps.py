#!/usr/bin/env python
import os
import csv
import json
import time
import math
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
import wandb
import subprocess

# --- Import your existing modules ---
from datasets.ssl_dataset import HDF5Dataset_Multimodal_Tiles_Iterable, AustrianCropValidation
from models.modules import TransformerEncoder, ProjectionHead
from models.ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from utils.lr_scheduler import adjust_learning_rate
from utils.metrics import linear_probe_evaluate, rankme
from utils.misc import remove_dir, save_checkpoint, plot_cross_corr

import matplotlib.pyplot as plt

# -----------------------------
#   Global Config / Constants
# -----------------------------
# CSV_PATH = "C:/.../param_sweep_temp.csv"  # 修改为实际 CSV 路径，如需要
CSV_PATH = "param_sweep.csv"
BATCH_SIZE = 512              # keep batch size constant across runs
HOST_NAME = "Daintree"          # used to label runs from a specific machine

# ------------------------------------
#   1.  Load up the CSV in memory using csv library
# ------------------------------------
def parse_value(val):
    try:
        if '.' in val:
            return float(val)
        else:
            return int(val)
    except:
        return val

ALL_ROWS = []
with open(CSV_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        new_row = {}
        for key, value in row.items():
            # 将 "ratio_s1/s2" 重命名为 "ratio_s1_s2"
            if key == "ratio_s1/s2":
                new_row["ratio_s1_s2"] = parse_value(value)
            else:
                new_row[key] = parse_value(value)
        ALL_ROWS.append(new_row)
logging.info(f"Loaded {len(ALL_ROWS)} rows from CSV.")

# ----------------------------------------------------------------------
#   Compute unique experiment_id based on hyperparameters
# ----------------------------------------------------------------------
def compute_experiment_id(config):
    keys = ["s2nhead", "s2layer", "s2dim_feedforward", "s2non_embedding",
            "s1nhead", "s1layer", "s1dim_feedforward", "s1non_embedding",
            "ratio_s1_s2", "alltotal", "actual_total"]
    id_parts = [str(config.get(k)) for k in keys]
    return "_".join(id_parts)

# ----------------------------------------------------------------------
#   Check for duplicate experiments using wandb API (only within same sweep)
# ----------------------------------------------------------------------
def is_duplicate_experiment(experiment_id):
    try:
        api = wandb.Api()
        current_run = wandb.run
        current_run_id = current_run.id if current_run is not None else None
        current_sweep_id = current_run.sweep_id if current_run is not None else None
        runs = api.runs("frankfeng1223/btfm-param-sweep-s1-s2-ratio-s1-s2-ratio")
        for run in runs:
            if run.id == current_run_id:
                continue
            # 仅检测相同 sweep_id 的 run
            run_sweep_id = getattr(run, "sweep_id", None)
            if current_sweep_id is not None and run_sweep_id != current_sweep_id:
                continue
            run_experiment_id = run.config.get("experiment_id")
            if run_experiment_id == experiment_id and run.state == "finished":
                return True
    except Exception as e:
        logging.error(f"Error checking duplicate experiments: {e}")
    return False

# ----------------------------------------------------------------------
#   The actual training / run function for wandb
# ----------------------------------------------------------------------
def train_and_evaluate():
    run = wandb.init(project="btfm-param-sweep-s1-s2-ratio-s1-s2-ratio")
    try:
        config = wandb.config
        row_idx = int(config.get("row_index", 0))
        row_data = ALL_ROWS[row_idx]
        # 合并当前 wandb.config 和 CSV 行数据，生成最终配置字典
        final_config = {**dict(config), **row_data}
        final_config["experiment_id"] = compute_experiment_id(final_config)

        if is_duplicate_experiment(final_config["experiment_id"]):
            print(f"Experiment with ID {final_config['experiment_id']} is already completed. Skipping.")
            wandb.log({"status": "duplicate_skip"})
            run.finish()
            try:
                api = wandb.Api()
                run_path = f"{run.entity}/{run.project}/{run.id}"
                run_obj = api.run(run_path)
                run_obj.delete()
                logging.info("Successfully deleted the run from W&B.")
            except Exception as ex_del:
                logging.error(f"Could not delete the run from W&B: {ex_del}")
            return

        logging.info(f"Starting run for row_index={row_idx}, actual_total={row_data['actual_total']}")
        if float(row_data["actual_total"]) >= 20000000:  # 由于GPU只有个4090，因此不能跑太大的模型
            print("Skipping configuration because actual_total >= 20M")
            wandb.log({"status": "skip_due_to_actual_total"})
            run.finish()
            try:
                api = wandb.Api()
                run_path = f"{run.entity}/{run.project}/{run.id}"
                run_obj = api.run(run_path)
                run_obj.delete()
                logging.info("Successfully deleted the run from W&B.")
            except Exception as ex_del:
                logging.error(f"Could not delete the run from W&B: {ex_del}")
            return

        # 更新 wandb.config（仅用于显示）
        wandb.config.update(row_data)
        wandb.config.update({"experiment_id": final_config["experiment_id"]})

        # 从 final_config 中提取超参数
        s2nhead = int(final_config["s2nhead"])
        s2layer = int(final_config["s2layer"])
        s2dim_feedforward = int(final_config["s2dim_feedforward"])
        s1nhead = int(final_config["s1nhead"])
        s1layer = int(final_config["s1layer"])
        s1dim_feedforward = int(final_config["s1dim_feedforward"])

        s2non_embedding = final_config["s2non_embedding"]
        s1non_embedding = final_config["s1non_embedding"]
        ratio = final_config["ratio_s1_s2"]
        alltotal = final_config["alltotal"]
        actual_total = final_config["actual_total"]

        wandb.config["host_name"] = HOST_NAME

        # Windows版本一般使用 cuda，如果不可用则 fallback 到 cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 固定步数设定：根据总样本数和 batch_size 计算总步长（只训练一个 epoch）
        training_config = {
            "data_root": "data/ssl_training/sweep/s1_s2_ratio",
            "batch_size": BATCH_SIZE,
            "learning_rate": 0.1,
            "barlow_lambda": 5e-3,
            "fusion_method": "sum",
            "latent_dim": 128,
            "projector_hidden_dim": 512,
            "projector_out_dim": 512,
            "min_valid_timesteps": 20,
            "sample_size_s2": 20,
            "sample_size_s1": 20,
            "num_workers": 8,
            "shuffle_tiles": True,
            "log_fraction": 0.001,
            "val_fraction": 0.01,
            "warmup_ratio": 0.2,
            "plateau_ratio": 0.2,
            "apply_mixup": False,
            "mixup_lambda": 1.0,
            "beta_alpha": 1.0,
            "beta_beta": 1.0,
            "total_samples": 8500000,
            "val_s2_bands_file_path": "data/ssl_training/austrian_crop/bands_downsample_100.npy",
            "val_s2_masks_file_path": "data/ssl_training/austrian_crop/masks_downsample_100.npy",
            "val_s2_doy_file_path": "data/ssl_training/austrian_crop/doys.npy",
            "val_s1_asc_bands_file_path": "data/ssl_training/austrian_crop/sar_ascending_downsample_100.npy",
            "val_s1_asc_doy_file_path": "data/ssl_training/austrian_crop/sar_ascending_doy.npy",
            "val_s1_desc_bands_file_path": "data/ssl_training/austrian_crop/sar_descending_downsample_100.npy",
            "val_s1_desc_doy_file_path": "data/ssl_training/austrian_crop/sar_descending_doy.npy",
            "val_labels_path": "data/ssl_training/austrian_crop/fieldtype_17classes_downsample_100.npy"
        }
        total_steps = int(training_config["total_samples"] / training_config["batch_size"])
        wandb.config["computed_total_steps"] = total_steps
        logging.info(f"Row={row_idx}: total_steps={total_steps}")

        s2_enc = TransformerEncoder(
            band_num=12,
            latent_dim=training_config["latent_dim"],
            nhead=s2nhead,
            num_encoder_layers=s2layer,
            dim_feedforward=s2dim_feedforward,
            dropout=0.1,
            max_seq_len=training_config["sample_size_s2"]
        ).to(device)

        s1_enc = TransformerEncoder(
            band_num=4,
            latent_dim=training_config["latent_dim"],
            nhead=s1nhead,
            num_encoder_layers=s1layer,
            dim_feedforward=s1dim_feedforward,
            dropout=0.1,
            max_seq_len=training_config["sample_size_s1"]
        ).to(device)

        if training_config['fusion_method'] == 'concat':
            proj_in_dim = training_config['latent_dim'] * 2
        else:
            proj_in_dim = training_config['latent_dim']

        projector = ProjectionHead(
            proj_in_dim,
            training_config['projector_hidden_dim'],
            training_config['projector_out_dim']
        ).to(device)

        if training_config['fusion_method'] == 'transformer':
            model = MultimodalBTModel(
                s2_enc,
                s1_enc,
                projector,
                fusion_method=training_config['fusion_method'],
                return_repr=True,
                latent_dim=training_config['latent_dim']
            ).to(device)
        else:
            model = MultimodalBTModel(
                s2_enc,
                s1_enc,
                projector,
                fusion_method=training_config['fusion_method'],
                return_repr=True
            ).to(device)

        criterion = BarlowTwinsLoss(lambda_coeff=training_config['barlow_lambda'])
        weight_params = [p for p in model.parameters() if p.ndim > 1]
        bias_params   = [p for p in model.parameters() if p.ndim == 1]
        optimizer = torch.optim.SGD(
            [{'params': weight_params}, {'params': bias_params}],
            lr=training_config['learning_rate'],
            momentum=0.9,
            weight_decay=1e-6
        )

        scaler = amp.GradScaler()
        warmup_steps = int(training_config["warmup_ratio"] * total_steps)
        plateau_steps = int(training_config["plateau_ratio"] * total_steps)
        log_interval = max(1, int(training_config["log_fraction"] * total_steps))
        val_interval = max(1, int(training_config["val_fraction"] * total_steps))
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Starting training with up to {total_steps} steps, row_index={row_idx}")
        step = 0
        examples = 0
        best_val_acc = 0.0
        rolling_loss = []
        rolling_size = 40
        epoch = 0

        while step < total_steps:
            dataset_train = HDF5Dataset_Multimodal_Tiles_Iterable(
                data_root=training_config['data_root'],
                min_valid_timesteps=training_config['min_valid_timesteps'],
                sample_size_s2=training_config['sample_size_s2'],
                sample_size_s1=training_config['sample_size_s1'],
                standardize=True,
                shuffle_tiles=training_config['shuffle_tiles']
            )
            train_loader = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=training_config['batch_size'],
                num_workers=training_config['num_workers'],
                drop_last=True,
                pin_memory=True,
                persistent_workers=True
            )
            model.train()
            for batch_data in train_loader:
                if step >= total_steps:
                    break
                s2_aug1 = batch_data['s2_aug1'].to(device, non_blocking=True)
                s2_aug2 = batch_data['s2_aug2'].to(device, non_blocking=True)
                s1_aug1 = batch_data['s1_aug1'].to(device, non_blocking=True)
                s1_aug2 = batch_data['s1_aug2'].to(device, non_blocking=True)
                adjust_learning_rate(
                    optimizer,
                    step,
                    total_steps,
                    training_config['learning_rate'],
                    training_config['warmup_ratio'],
                    training_config['plateau_ratio']
                )
                optimizer.zero_grad()
                with amp.autocast():
                    z1, repr1 = model(s2_aug1, s1_aug1)
                    z2, repr2 = model(s2_aug2, s1_aug2)
                    loss_main, bar_main, off_main = criterion(z1, z2)
                    loss_mix = 0.0
                    if training_config['apply_mixup']:
                        B = s2_aug1.size(0)
                        idxs = torch.randperm(B, device=device)
                        alpha = torch.distributions.Beta(
                            training_config['beta_alpha'],
                            training_config['beta_beta']
                        ).sample().to(device)
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
                        loss_mix = (
                            training_config['mixup_lambda']
                            * training_config['barlow_lambda']
                            * (diff_a + diff_b)
                        )
                    total_loss = loss_main + loss_mix
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                examples += s2_aug1.size(0)
                if step % log_interval == 0 or step == 1:
                    avg_loss = (sum(rolling_loss[-(rolling_size - 1):]) + loss_main.item()) / \
                               min(len(rolling_loss) + 1, rolling_size)
                    rolling_loss.append(loss_main.item())
                    if len(rolling_loss) > rolling_size:
                        rolling_loss = rolling_loss[-rolling_size:]
                    current_lr = optimizer.param_groups[0]['lr']
                    erank_z = rankme(z1)
                    erank_repr = rankme(repr1)
                    wandb_dict = {
                        "step": step,
                        "epoch": epoch,
                        "loss_main": loss_main.item(),
                        "mix_loss": loss_mix,
                        "total_loss": total_loss.item(),
                        "avg_loss": avg_loss,
                        "lr": current_lr,
                        "examples_seen": examples,
                        "rank_z": erank_z,
                        "rank_repr": erank_repr
                    }
                    wandb.log(wandb_dict, step=step)
                    logging.info(f"Step={step}, loss={loss_main.item():.4f}, avg_loss={avg_loss:.4f}, "
                                 f"lr={current_lr:.4f}, rank_z={erank_z:.4f}, rank_repr={erank_repr:.4f}")
                if step % val_interval == 0 and step > 0:
                    val_keys = [
                        'val_s2_bands_file_path',
                        'val_s2_masks_file_path',
                        'val_s2_doy_file_path',
                        'val_s1_asc_bands_file_path',
                        'val_s1_asc_doy_file_path',
                        'val_s1_desc_bands_file_path',
                        'val_s1_desc_doy_file_path',
                        'val_labels_path'
                    ]
                    if all(k in training_config for k in val_keys) and all(training_config[k] for k in val_keys):
                        model.eval()
                        val_dataset = AustrianCropValidation(
                            s2_bands_file_path=training_config['val_s2_bands_file_path'],
                            s2_masks_file_path=training_config['val_s2_masks_file_path'],
                            s2_doy_file_path=training_config['val_s2_doy_file_path'],
                            s1_asc_bands_file_path=training_config['val_s1_asc_bands_file_path'],
                            s1_asc_doy_file_path=training_config['val_s1_asc_doy_file_path'],
                            s1_desc_bands_file_path=training_config['val_s1_desc_bands_file_path'],
                            s1_desc_doy_file_path=training_config['val_s1_desc_doy_file_path'],
                            labels_path=training_config['val_labels_path'],
                            sample_size_s2=training_config['sample_size_s2'],
                            sample_size_s1=training_config['sample_size_s1'],
                            min_valid_timesteps=0,
                            standardize=True
                        )
                        val_loader = torch.utils.data.DataLoader(
                            val_dataset, batch_size=512, shuffle=False, num_workers=0
                        )
                        val_acc = linear_probe_evaluate(model, val_loader, device=device)
                        wandb.log({"val_acc": val_acc}, step=step)
                        logging.info(f"Validation step={step}: val_acc={val_acc:.4f}")
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            ckpt_name = f"checkpoints/sweep_s1_s2_ratio/best_model_row{row_idx}.pt"
                            save_checkpoint(model, optimizer, epoch, step, best_val_acc, ckpt_name)
                        model.train()
            logging.info(f"Epoch {epoch} finished, current step={step}, total_steps={total_steps}")
            epoch += 1
        logging.info("Training completed.")
        wandb.log({"final_val_acc": best_val_acc, "final_step": step})
        run.finish()
    except Exception as e:
        logging.error(f"Encountered an error: {e}")
        run.finish(exit_code=1)
        try:
            api = wandb.Api()
            run_path = f"{run.entity}/{run.project}/{run.id}"
            run_obj = api.run(run_path)
            run_obj.delete()
            logging.info("Successfully deleted the run from W&B.")
        except Exception as ex_del:
            logging.error(f"Could not delete the run from W&B: {ex_del}")
        raise e

# ----------------------------------------------------------------------
#   The main function that sets up the Sweep
# ----------------------------------------------------------------------
def main():
    # 设置随机数种子
    torch.manual_seed(3407)
    np.random.seed(3407)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="Either 'init' to init sweep, or 'agent' to run agent.")
    args = parser.parse_args()

    sweep_config = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "row_index": {"values": list(range(len(ALL_ROWS)))},
            "s2nhead": {"values": [row["s2nhead"] for row in ALL_ROWS]},
            "s2layer": {"values": [row["s2layer"] for row in ALL_ROWS]},
            "s2dim_feedforward": {"values": [row["s2dim_feedforward"] for row in ALL_ROWS]},
            "s2non_embedding": {"values": [row["s2non_embedding"] for row in ALL_ROWS]},
            "s1nhead": {"values": [row["s1nhead"] for row in ALL_ROWS]},
            "s1layer": {"values": [row["s1layer"] for row in ALL_ROWS]},
            "s1dim_feedforward": {"values": [row["s1dim_feedforward"] for row in ALL_ROWS]},
            "s1non_embedding": {"values": [row["s1non_embedding"] for row in ALL_ROWS]},
            "ratio_s1_s2": {"values": [row["ratio_s1_s2"] for row in ALL_ROWS]},
            "alltotal": {"values": [row["alltotal"] for row in ALL_ROWS]},
            "actual_total": {"values": [row["actual_total"] for row in ALL_ROWS]}
        }
    }

    if args.mode == "init":
        sweep_id = wandb.sweep(
            sweep_config,
            project="btfm-param-sweep-s1-s2-ratio-s1-s2-ratio"
        )
        print("Created sweep with ID:", sweep_id)
    elif args.mode == "agent":
        wandb.agent(
            sweep_id="ztg9m4tz",  # 请确保使用实际的 sweep_id
            function=train_and_evaluate,
            project="btfm-param-sweep-s1-s2-ratio-s1-s2-ratio",
            entity="frankfeng1223",
            count=20,
        )
    else:
        print("Unknown mode. Use --mode init or --mode agent")

if __name__ == "__main__":
    main()
