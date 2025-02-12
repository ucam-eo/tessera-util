# src/train_downstream_regression.py

import os
import argparse
import logging
import numpy as np
import torch
import wandb
import tqdm
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.modules import TransformerEncoder, ProjectionHead
from models.ssl_model import MultimodalBTModel
from models.downstream_model import MultimodalDownstreamModel, RegressionHead
from models.builder import build_ssl_model
from datasets.downstream_dataset import BorneoCropRegression, borneo_crop_regression_collate_fn

# ------------------------------
# 参数解析
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Downstream Regression Training")
    parser.add_argument('--config', type=str, default="configs/downstream_borneo.py",
                        help="Path to config file (e.g. configs/downstream_borneo.py)")
    return parser.parse_args()

# ------------------------------
# 主函数
# ------------------------------
def main():
    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = wandb.init(project="btfm-downstream", config=config)

    # ------------------------------
    # 构建数据集并划分训练/验证/测试
    # ------------------------------
    dataset = BorneoCropRegression(
        s2_bands_file_path = config['s2_bands_file_path'],
        s2_masks_file_path = config['s2_masks_file_path'],
        s2_doy_file_path   = config['s2_doy_file_path'],
        s1_asc_bands_file_path = config['s1_asc_bands_file_path'],
        s1_asc_doy_file_path   = config['s1_asc_doy_file_path'],
        s1_desc_bands_file_path = config['s1_desc_bands_file_path'],
        s1_desc_doy_file_path   = config['s1_desc_doy_file_path'],
        chm_path = config['chm_path'],
        split='train',  # 此处所有有效像素先全部加载，后续根据比例划分
        shuffle=True,
        standardize = config.get('standardize', True),
        min_valid_timesteps = config.get('min_valid_timesteps', 0),
        sample_size_s2 = config.get('sample_size_s2', 20),
        sample_size_s1 = config.get('sample_size_s1', 20)
    )
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    train_count = int(total_samples * config["train_ratio"])
    remaining = total_samples - train_count
    val_count = int(remaining * config["val_ratio"])
    test_count = remaining - val_count
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count+val_count]
    test_indices = indices[train_count+val_count:]
    logging.info(f"Total samples: {total_samples}, Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"], 
                              collate_fn=borneo_crop_regression_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"],
                            collate_fn=borneo_crop_regression_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=config["num_workers"],
                             collate_fn=borneo_crop_regression_collate_fn)

    ssl_model = build_ssl_model(config, device)
    
    logging.info("After loading checkpoint, s2_backbone weights:")
    logging.info(ssl_model.s2_backbone.fc_out.weight)
    logging.info(f"Loading SSL checkpoint from {config['checkpoint_path']}")
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
    ssl_model.load_state_dict(checkpoint[state_key], strict=True)
    logging.info("SSL backbone weights after loading checkpoint (s2 backbone):")
    logging.info(ssl_model.s2_backbone.fc_out.weight)
    
    # 冻结 SSL 骨干参数
    for param in ssl_model.s2_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.s1_backbone.parameters():
        param.requires_grad = False

    # ------------------------------
    # 构建下游回归模型
    # ------------------------------
    if config["fusion_method"] == "concat":
        regressor_in_dim = config['latent_dim'] * 2
    else:
        regressor_in_dim = config['latent_dim']
    reg_head = RegressionHead(input_dim=regressor_in_dim, hidden_dim=512).to(device)
    downstream_model = MultimodalDownstreamModel(s2_backbone=ssl_model.s2_backbone,
                                                   s1_backbone=ssl_model.s1_backbone,
                                                   head=reg_head,
                                                   fusion_method=config["fusion_method"]).to(device)
    logging.info("Downstream model constructed. s2 backbone weights:")
    logging.info(downstream_model.s2_backbone.fc_out.weight)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, downstream_model.parameters()),
                      lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss().to(device)
    
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_folder = os.path.join("checkpoints", "downstream_regression")
    os.makedirs(checkpoint_folder, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_folder, "downstream_regression_best.pt")
    
    # ------------------------------
    # 训练循环
    # ------------------------------
    for epoch in range(config["epochs"]):
        downstream_model.train()
        train_loss_sum = 0.0
        train_preds = []
        train_targets = []
        for s2_batch, s1_batch, targets_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Train"):
            s2_batch = s2_batch.to(device)   # (B, sample_size_s2, C_s2)
            s1_batch = s1_batch.to(device)   # (B, sample_size_s1, C_s1)
            targets_batch = targets_batch.to(device)  # (B, 1)
            optimizer.zero_grad()
            preds = downstream_model(s2_batch, s1_batch)  # (B, 1)
            loss = criterion(preds, targets_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * s2_batch.size(0)
            train_preds.append(preds.detach().cpu().numpy())
            train_targets.append(targets_batch.detach().cpu().numpy())
        avg_train_loss = train_loss_sum / len(train_dataset)
        # 计算训练MAE和RMSE
        train_preds = np.concatenate(train_preds, axis=0).flatten()
        train_targets = np.concatenate(train_targets, axis=0).flatten()
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        train_r2 = r2_score(train_targets, train_preds)
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "train_r2": train_r2,
                   "train_mae": train_mae, "train_rmse": train_rmse}, step=epoch)
        logging.info(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.4f}, MAE = {train_mae:.4f}, RMSE = {train_rmse:.4f}, R2 = {train_r2:.4f}")
        
        # 验证
        downstream_model.eval()
        val_loss_sum = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for s2_batch, s1_batch, targets_batch in tqdm.tqdm(val_loader, desc="Validation"):
                s2_batch = s2_batch.to(device)
                s1_batch = s1_batch.to(device)
                targets_batch = targets_batch.to(device)
                preds = downstream_model(s2_batch, s1_batch)
                loss = criterion(preds, targets_batch)
                val_loss_sum += loss.item() * s2_batch.size(0)
                val_preds.append(preds.detach().cpu().numpy())
                val_targets.append(targets_batch.detach().cpu().numpy())
        avg_val_loss = val_loss_sum / len(val_dataset)
        val_preds = np.concatenate(val_preds, axis=0).flatten()
        val_targets = np.concatenate(val_targets, axis=0).flatten()
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)
        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "val_r2": val_r2,
                   "val_mae": val_mae, "val_rmse": val_rmse}, step=epoch)
        logging.info(f"[Epoch {epoch+1}] Val Loss = {avg_val_loss:.4f}, MAE = {val_mae:.4f}, RMSE = {val_rmse:.4f}, R2 = {val_r2:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(downstream_model.state_dict(), best_ckpt_path)
            logging.info(f"Saved best model at epoch {best_epoch}")
    
    # ------------------------------
    # 测试评估
    # ------------------------------
    logging.info(f"Loading best model from epoch {best_epoch} for test evaluation.")
    downstream_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    downstream_model.eval()
    test_loss_sum = 0.0
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for s2_batch, s1_batch, targets_batch in tqdm.tqdm(test_loader, desc="Test"):
            s2_batch = s2_batch.to(device)
            s1_batch = s1_batch.to(device)
            targets_batch = targets_batch.to(device)
            preds = downstream_model(s2_batch, s1_batch)
            loss = criterion(preds, targets_batch)
            test_loss_sum += loss.item() * s2_batch.size(0)
            test_preds.append(preds.detach().cpu().numpy())
            test_targets.append(targets_batch.detach().cpu().numpy())
    avg_test_loss = test_loss_sum / len(test_dataset)
    test_preds = np.concatenate(test_preds, axis=0).flatten()
    test_targets = np.concatenate(test_targets, axis=0).flatten()
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds)
    wandb.log({"test_loss": avg_test_loss, "test_r2": test_r2,
               "test_mae": test_mae, "test_rmse": test_rmse})
    logging.info(f"Test Loss: {avg_test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
    
    wandb_run.finish()

if __name__ == "__main__":
    main()
