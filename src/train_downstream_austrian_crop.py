# src/train_downstream.py

import os
import time
import math
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import wandb
import tqdm

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from datasets.downstream_dataset import AustrianCrop, austrian_crop_collate_fn
from models.modules import TransformerEncoder, ProjectionHead
from models.downstream_model import ClassificationHead, MultimodalDownstreamModel

class_names = [
    "Legume",
    "Soy",
    "Summer Grain",
    "Winter Grain",
    "Corn",
    "Sunflower",
    "Mustard",
    "Potato",
    "Beet",
    "Squash",
    "Grapes",
    "Tree Fruit",
    "Cover Crop",
    "Grass",
    "Fallow",
    "Other (Plants)",
    "Other (Non Plants)"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Downstream Classification Training")
    parser.add_argument('--config', type=str, default="configs/downstream_config.py", help="Path to config file (e.g. configs/downstream_config.py)")
    return parser.parse_args()

def load_encoder_weights_only(checkpoint_path, latent_dim, config, device):
    """
    只加载encoder权重，避免创建大的projector
    """
    # 创建encoder（不包含projector）
    s2_backbone = TransformerEncoder(
        band_num=10,
        latent_dim=latent_dim,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=config['max_seq_len_s2']
    ).to(device)
    
    s1_backbone = TransformerEncoder(
        band_num=2,
        latent_dim=latent_dim,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=config['max_seq_len_s1']
    ).to(device)
    
    # 从checkpoint中推断dim_reducer的正确结构
    # 根据错误信息，输入是1024维，输出是128维（latent_dim）
    dim_reducer = nn.Sequential(nn.Linear(1024, latent_dim)  # 输出维度是latent_dim(128)
    ).to(device)
    
    # 加载checkpoint
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint path: {checkpoint_path}")
    
    state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
    state_dict = checkpoint[state_key]
    
    # 处理FSDP前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # 提取各部分的权重
    s2_state = {k[len('s2_backbone.'):]: v for k, v in new_state_dict.items() if k.startswith('s2_backbone.')}
    s1_state = {k[len('s1_backbone.'):]: v for k, v in new_state_dict.items() if k.startswith('s1_backbone.')}
    dim_reducer_state = {k[len('dim_reducer.'):]: v for k, v in new_state_dict.items() if k.startswith('dim_reducer.')}
    
    # 打印维度信息用于调试
    print("dim_reducer state_dict keys:", list(dim_reducer_state.keys()))
    for key, value in dim_reducer_state.items():
        print(f"  {key}: {value.shape}")
    
    # 加载权重
    logging.info("Before loading checkpoint, s2_backbone weights:")
    logging.info(s2_backbone.embedding[0].weight)
    
    s2_backbone.load_state_dict(s2_state, strict=True)
    s1_backbone.load_state_dict(s1_state, strict=True)
    dim_reducer.load_state_dict(dim_reducer_state, strict=True)
    
    logging.info("After loading checkpoint, s2_backbone weights:")
    logging.info(s2_backbone.embedding[0].weight)
    
    # 冻结权重
    for param in s2_backbone.parameters():
        param.requires_grad = False
    for param in s1_backbone.parameters():
        param.requires_grad = False
    for param in dim_reducer.parameters():
        param.requires_grad = False
    
    return s2_backbone, s1_backbone, dim_reducer

def main():
    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = wandb.init(project="btfm-downstream", config=config)
    
    # Step 1: 加载标签和field_ids
    labels = np.load(config['labels_path']).astype(np.int64)
    field_ids = np.load(config['field_ids_path']).astype(np.int64)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    logging.info(f"Unique labels: {unique_labels}, num_classes={num_classes}")
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    labels_remapped = np.vectorize(label_map.get)(labels)
    labels = labels_remapped

    # Step 2: 根据CSV划分train/val/test
    fielddata_df = pd.read_csv(config['updated_fielddata_csv'])
    area_summary = fielddata_df.groupby('SNAR_CODE')['area_m2'].sum().reset_index()
    area_summary.rename(columns={'area_m2': 'total_area'}, inplace=True)
    all_selected_fids = []
    for _, row in area_summary.iterrows():
        sn_code = row['SNAR_CODE']
        total_area = row['total_area']
        target_area = total_area * config['training_ratio']
        rows_sncode = fielddata_df[fielddata_df['SNAR_CODE'] == sn_code].sort_values(by='area_m2')
        selected_fids = []
        selected_area_sum = 0
        for _, r2 in rows_sncode.iterrows():
            if selected_area_sum < target_area:
                selected_fids.append(int(r2['fid_1']))
                selected_area_sum += r2['area_m2']
            else:
                break
        all_selected_fids.extend(selected_fids)
    all_selected_fids = list(set(all_selected_fids))
    logging.info(f"Number of selected train field IDs: {len(all_selected_fids)}")
    all_fields = fielddata_df['fid_1'].unique()
    set_train = set(all_selected_fids)
    set_all = set(all_fields)
    remaining = list(set_all - set_train)
    remaining = np.array(remaining)
    np.random.shuffle(remaining)
    val_count = int(len(remaining) * config['val_test_split_ratio'])
    val_fids = remaining[:val_count]
    test_fids = remaining[val_count:]
    train_fids = np.array(all_selected_fids)
    logging.info(f"Train fields: {len(train_fids)}, Val fields: {len(val_fids)}, Test fields: {len(test_fids)}")

    # Step 3: 构建数据集和DataLoader
    train_dataset = AustrianCrop(
        s2_bands_file_path=config['s2_bands_file_path'],
        s2_masks_file_path=config['s2_masks_file_path'],
        s2_doy_file_path=config['s2_doy_file_path'],
        s1_asc_bands_file_path=config['s1_asc_bands_file_path'],
        s1_asc_doy_file_path=config['s1_asc_doy_file_path'],
        s1_desc_bands_file_path=config['s1_desc_bands_file_path'],
        s1_desc_doy_file_path=config['s1_desc_doy_file_path'],
        labels_path=config['labels_path'],
        field_ids_path=config['field_ids_path'],
        train_fids=train_fids,
        val_fids=val_fids,
        test_fids=test_fids,
        lon_lat_path=config['lon_lat_path'],
        t2m_path=config['t2m_path'],
        lai_hv_path=config['lai_hv_path'],
        lai_lv_path=config['lai_lv_path'],
        split='train',
        shuffle=True,
        is_training=True,
        standardize=True,
        min_valid_timesteps=0,
        sample_size_s2=config['max_seq_len_s2'],
        sample_size_s1=config['max_seq_len_s1']
    )
    val_dataset = AustrianCrop(
        s2_bands_file_path=config['s2_bands_file_path'],
        s2_masks_file_path=config['s2_masks_file_path'],
        s2_doy_file_path=config['s2_doy_file_path'],
        s1_asc_bands_file_path=config['s1_asc_bands_file_path'],
        s1_asc_doy_file_path=config['s1_asc_doy_file_path'],
        s1_desc_bands_file_path=config['s1_desc_bands_file_path'],
        s1_desc_doy_file_path=config['s1_desc_doy_file_path'],
        labels_path=config['labels_path'],
        field_ids_path=config['field_ids_path'],
        train_fids=train_fids,
        val_fids=val_fids,
        test_fids=test_fids,
        lon_lat_path=config['lon_lat_path'],
        t2m_path=config['t2m_path'],
        lai_hv_path=config['lai_hv_path'],
        lai_lv_path=config['lai_lv_path'],
        split='val',
        shuffle=False,
        is_training=False,
        standardize=True,
        min_valid_timesteps=0,
        sample_size_s2=config['max_seq_len_s2'],
        sample_size_s1=config['max_seq_len_s1']
    )
    test_dataset = AustrianCrop(
        s2_bands_file_path=config['s2_bands_file_path'],
        s2_masks_file_path=config['s2_masks_file_path'],
        s2_doy_file_path=config['s2_doy_file_path'],
        s1_asc_bands_file_path=config['s1_asc_bands_file_path'],
        s1_asc_doy_file_path=config['s1_asc_doy_file_path'],
        s1_desc_bands_file_path=config['s1_desc_bands_file_path'],
        s1_desc_doy_file_path=config['s1_desc_doy_file_path'],
        labels_path=config['labels_path'],
        field_ids_path=config['field_ids_path'],
        train_fids=train_fids,
        val_fids=val_fids,
        test_fids=test_fids,
        lon_lat_path=config['lon_lat_path'],
        t2m_path=config['t2m_path'],
        lai_hv_path=config['lai_hv_path'],
        lai_lv_path=config['lai_lv_path'],
        split='test',
        shuffle=False,
        is_training=False,
        standardize=True,
        min_valid_timesteps=0,
        sample_size_s2=config['max_seq_len_s2'],
        sample_size_s1=config['max_seq_len_s1']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], collate_fn=austrian_crop_collate_fn, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], collate_fn=austrian_crop_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], collate_fn=austrian_crop_collate_fn)

    # Step 4: 只加载encoder权重，避免创建大的projector
    latent_dim = config['latent_dim']
    s2_backbone, s1_backbone, dim_reducer = load_encoder_weights_only(
        config['checkpoint_path'], latent_dim, config, device
    )

    # Step 5: 构建下游分类头和完整模型
    if config['fusion_method'] == 'concat':
        classification_in_dim = latent_dim
    else:
        classification_in_dim = latent_dim
    clf_head = ClassificationHead(input_dim=classification_in_dim, num_classes=num_classes).to(device)
    downstream_model = MultimodalDownstreamModel(s2_backbone=s2_backbone,
                                                  s1_backbone=s1_backbone,
                                                  head=clf_head,
                                                  dim_reducer=dim_reducer,
                                                  fusion_method=config['fusion_method']).to(device)
    
    # 打印模型构建后backbone权重
    logging.info("After constructing downstream model, s2_backbone weights:")
    logging.info(downstream_model.s2_backbone.embedding[0].weight)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, downstream_model.parameters()),
                      lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    checkpoint_folder = os.path.join("checkpoints", "downstream")
    os.makedirs(checkpoint_folder, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_folder, f"downstream_best.pt")

    for epoch in range(config['epochs']):
        downstream_model.train()
        train_preds, train_targets = [], []
        train_loss_sum = 0.0
        for s2_batch, s1_batch, labels_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Train"):
            s2_batch = s2_batch.to(device)
            s1_batch = s1_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            logits = downstream_model(s2_batch, s1_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * s2_batch.size(0)
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_targets.extend(labels_batch.cpu().numpy())
        avg_train_loss = train_loss_sum / len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        logging.info(f"[Epoch {epoch+1}] Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}")
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1
        }, step=epoch)
        
        # ---- Validation
        downstream_model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for s2_batch, s1_batch, labels_batch in tqdm.tqdm(val_loader, desc="Validation"):
                s2_batch = s2_batch.to(device)
                s1_batch = s1_batch.to(device)
                labels_batch = labels_batch.to(device)
                logits = downstream_model(s2_batch, s1_batch)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(labels_batch.cpu().numpy())
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        wandb.log({
            "epoch": epoch+1,
            "val_acc": val_acc,
            "val_f1": val_f1
        }, step=epoch)
        logging.info(f"[Epoch {epoch+1}] Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch+1
            torch.save(downstream_model.state_dict(), best_ckpt_path)
            logging.info(f"Saved best model at epoch {best_epoch}")
    
    # ---- Test
    logging.info(f"Loading best checkpoint from epoch {best_epoch} for test evaluation.")
    downstream_model.load_state_dict(torch.load(best_ckpt_path))
    downstream_model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for s2_batch, s1_batch, labels_batch in tqdm.tqdm(test_loader, desc="Test"):
            s2_batch = s2_batch.to(device)
            s1_batch = s1_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits = downstream_model(s2_batch, s1_batch)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_targets.extend(labels_batch.cpu().numpy())
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='weighted')
    report = classification_report(test_targets, test_preds, digits=4)
    conf_mat = confusion_matrix(test_targets, test_preds)
    logging.info(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    logging.info(f"Classification Report:\n{report}")
    cm_df = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
    logging.info(f"\nConfusion Matrix:\n{cm_df}")
    
    # 保存测试结果到txt文件
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "austrian_crop_results.txt"), "w") as f:
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"\nConfusion Matrix:\n{cm_df}\n")
    
    wandb.log({"test_acc": test_acc, "test_f1": test_f1})
    wandb_run.finish()

if __name__ == "__main__":
    main()