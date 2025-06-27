# src/train_downstream_austrian_crop_by_class.py

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
from collections import Counter, defaultdict

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from datasets.downstream_dataset import AustrianCropSelectByClass, select_pixels_by_class, austrian_crop_class_based_collate_fn
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
    parser = argparse.ArgumentParser(description="Downstream Classification Training with Class-Based Pixel Selection")
    parser.add_argument('--config', type=str, default="configs/downstream_config.py", help="Path to config file (e.g. configs/downstream_config.py)")
    return parser.parse_args()

def main():
    # np.random.seed(42)
    
    args_cli = parse_args()
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']

    # Add PIXELS_PER_CLASS and VAL_TEST_SPLIT_RATIO parameters if not in config
    if 'PIXELS_PER_CLASS' not in config:
        config['PIXELS_PER_CLASS'] = 1  # Default value
    if 'VAL_TEST_SPLIT_RATIO' not in config:
        config['VAL_TEST_SPLIT_RATIO'] = 1/7  # Default value

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = wandb.init(project="btfm-downstream", config=config)
    
    # Step 1: Load labels
    logging.info("Loading labels...")
    labels = np.load(config['labels_path']).astype(np.int64)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    logging.info(f"Unique labels: {unique_labels}, num_classes={num_classes}")
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    labels_remapped = np.vectorize(label_map.get)(labels)
    labels = labels_remapped

    # Step 2: Select pixels for training, validation, and testing
    logging.info(f"Selecting pixels with {config['PIXELS_PER_CLASS']} pixels per class...")
    train_indices, val_indices, test_indices = select_pixels_by_class(
        labels, 
        config['PIXELS_PER_CLASS'], 
        config['VAL_TEST_SPLIT_RATIO']
    )
    logging.info(f"Selected {len(train_indices)} training pixels, {len(val_indices)} validation pixels, {len(test_indices)} test pixels")

    # Count and log pixels per class in training set
    train_class_counts = defaultdict(int)
    for y, x in train_indices:
        train_class_counts[labels[y, x]] += 1
    
    for cls in sorted(train_class_counts.keys()):
        if cls == 0:  # Skip background
            continue
        class_name = class_names[cls-1] if cls-1 < len(class_names) else f"Class {cls}"
        logging.info(f"Class {cls} ({class_name}): {train_class_counts[cls]} training pixels")

    # Step 3: Create custom datasets using pixel indices
    train_dataset = AustrianCropSelectByClass(
        s2_bands_file_path=config['s2_bands_file_path'],
        s2_masks_file_path=config['s2_masks_file_path'],
        s2_doy_file_path=config['s2_doy_file_path'],
        s1_asc_bands_file_path=config['s1_asc_bands_file_path'],
        s1_asc_doy_file_path=config['s1_asc_doy_file_path'],
        s1_desc_bands_file_path=config['s1_desc_bands_file_path'],
        s1_desc_doy_file_path=config['s1_desc_doy_file_path'],
        labels_path=config['labels_path'],
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split='train',
        shuffle=True,
        is_training=True,
        standardize=True,
        min_valid_timesteps=0,
        sample_size_s2=config['max_seq_len_s2'],
        sample_size_s1=config['max_seq_len_s1']
    )
    
    val_dataset = AustrianCropSelectByClass(
        s2_bands_file_path=config['s2_bands_file_path'],
        s2_masks_file_path=config['s2_masks_file_path'],
        s2_doy_file_path=config['s2_doy_file_path'],
        s1_asc_bands_file_path=config['s1_asc_bands_file_path'],
        s1_asc_doy_file_path=config['s1_asc_doy_file_path'],
        s1_desc_bands_file_path=config['s1_desc_bands_file_path'],
        s1_desc_doy_file_path=config['s1_desc_doy_file_path'],
        labels_path=config['labels_path'],
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split='val',
        shuffle=False,
        is_training=False,
        standardize=True,
        min_valid_timesteps=0,
        sample_size_s2=config['max_seq_len_s2'],
        sample_size_s1=config['max_seq_len_s1']
    )
    
    test_dataset = AustrianCropSelectByClass(
        s2_bands_file_path=config['s2_bands_file_path'],
        s2_masks_file_path=config['s2_masks_file_path'],
        s2_doy_file_path=config['s2_doy_file_path'],
        s1_asc_bands_file_path=config['s1_asc_bands_file_path'],
        s1_asc_doy_file_path=config['s1_asc_doy_file_path'],
        s1_desc_bands_file_path=config['s1_desc_bands_file_path'],
        s1_desc_doy_file_path=config['s1_desc_doy_file_path'],
        labels_path=config['labels_path'],
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split='test',
        shuffle=False,
        is_training=False,
        standardize=True,
        min_valid_timesteps=0,
        sample_size_s2=config['max_seq_len_s2'],
        sample_size_s1=config['max_seq_len_s1']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'], 
        collate_fn=austrian_crop_class_based_collate_fn, 
        pin_memory=True, 
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'], 
        collate_fn=austrian_crop_class_based_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'], 
        collate_fn=austrian_crop_class_based_collate_fn
    )

    # Step 4: Build backbone network and load checkpoint
    latent_dim = config['latent_dim']
    s2_backbone_ssl = TransformerEncoder(
        band_num=10,
        latent_dim=latent_dim,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=config['max_seq_len_s2']
    )
    s1_backbone_ssl = TransformerEncoder(
        band_num=2,
        latent_dim=latent_dim,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=config['max_seq_len_s1']
    )
    if config['fusion_method'] == 'concat':
        input_dim_for_projector = latent_dim
    else:
        input_dim_for_projector = latent_dim
    projector_ssl = ProjectionHead(input_dim_for_projector, config['projector_hidden_dim'], config['projector_out_dim'])
    
    # Build SSL model to load checkpoint
    from models.ssl_model import MultimodalBTModel
    ssl_model = MultimodalBTModel(s2_backbone_ssl, s1_backbone_ssl, projector_ssl,
                                  fusion_method=config['fusion_method'], return_repr=True).to(device)
    logging.info(f"Loading checkpoint from {config['checkpoint_path']}")
    
    # Print pre-loading weights
    logging.info("Before loading checkpoint, s2_backbone weights:")
    logging.info(ssl_model.s2_backbone.embedding[0].weight)
    
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
    
    ################### For FSDP processing ###################
    state_dict = checkpoint[state_key]
    # Create new state_dict, removing "_orig_mod." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]  # Remove prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    # Load processed state_dict
    ssl_model.load_state_dict(new_state_dict, strict=True)
    #####################################################
    
    # Print post-loading weights
    logging.info("After loading checkpoint, s2_backbone weights:")
    logging.info(ssl_model.s2_backbone.embedding[0].weight)
    
    # Freeze backbone
    for param in ssl_model.s2_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.s1_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.dim_reducer.parameters():
        param.requires_grad = False

    # Step 5: Build downstream classification head and complete model
    if config['fusion_method'] == 'concat':
        classification_in_dim = latent_dim
    else:
        classification_in_dim = latent_dim
    clf_head = ClassificationHead(input_dim=classification_in_dim, num_classes=num_classes).to(device)
    downstream_model = MultimodalDownstreamModel(s2_backbone=ssl_model.s2_backbone,
                                                 s1_backbone=ssl_model.s1_backbone,
                                                 head=clf_head,
                                                 dim_reducer=ssl_model.dim_reducer,
                                                 fusion_method=config['fusion_method']).to(device)
    
    # Print post-construction weights
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
        # 只在30个epoch后进行验证
        if epoch < 30:
            continue
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
    # Save test results to txt file
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "austrian_crop_results_by_class.txt"), "w") as f:
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"\nConfusion Matrix:\n{cm_df}\n")
    wandb.log({"test_acc": test_acc, "test_f1": test_f1})
    wandb_run.finish()

if __name__ == "__main__":
    main()