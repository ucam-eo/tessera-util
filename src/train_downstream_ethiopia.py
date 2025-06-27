import os
import time
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import tqdm

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from datasets.downstream_dataset import EthiopiaDataset, ethiopia_crop_collate_fn
from models.modules import TransformerEncoder, ProjectionHead
from models.downstream_model import ClassificationHead, MultimodalDownstreamModel, LinearProbeHead
from models.ssl_model import MultimodalBTModel

def parse_args():
    parser = argparse.ArgumentParser(description="Ethiopia Downstream Classification Training")
    parser.add_argument('--config', type=str, default="configs/ethiopia_config.py", help="Path to config file")
    parser.add_argument('--no_wandb', action='store_true', help="Disable wandb logging")
    return parser.parse_args()

def main():
    args_cli = parse_args()
    
    # 设置随机种子
    # np.random.seed(args_cli.seed)
    # torch.manual_seed(args_cli.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args_cli.seed)
    
    # 加载配置
    config_module = {}
    with open(args_cli.config, "r") as f:
        exec(f.read(), config_module)
    config = config_module['config']
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置wandb
    use_wandb = False
    if not args_cli.no_wandb:
        try:
            import wandb
            run_name = f"ethiopia_crop_{time.strftime('%Y%m%d_%H%M%S')}"
            wandb_run = wandb.init(project="btfm-downstream", name=run_name, config=config)
            use_wandb = True
        except ImportError:
            logging.warning("wandb未安装，将不使用wandb记录")
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], "logs"), exist_ok=True)
    
    # 创建数据集和数据加载器
    train_dataset = EthiopiaDataset(
        polygons_locations_csv=config['polygons_locations_csv'],
        data_processed_dir=config['data_processed_dir'],
        data_raw_dir=config['data_raw_dir'],
        class_names=config['class_names'],
        split='train',
        samples_per_class=config['samples_per_class'],
        standardize=True,
        max_seq_len_s2=config['max_seq_len_s2'],
        max_seq_len_s1=config['max_seq_len_s1']
    )
    
    val_dataset = EthiopiaDataset(
        polygons_locations_csv=config['polygons_locations_csv'],
        data_processed_dir=config['data_processed_dir'],
        data_raw_dir=config['data_raw_dir'],
        class_names=config['class_names'],
        split='val',
        samples_per_class=config['samples_per_class'],
        standardize=True,
        max_seq_len_s2=config['max_seq_len_s2'],
        max_seq_len_s1=config['max_seq_len_s1']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'], 
        collate_fn=ethiopia_crop_collate_fn, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'], 
        collate_fn=ethiopia_crop_collate_fn
    )
    
    logging.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 构建模型
    num_classes = len(config['class_names'])
    logging.info(f"构建具有 {num_classes} 个类别的模型: {config['class_names']}")
    
    # 构建骨干网络
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
    
    input_dim_for_projector = latent_dim
    projector_ssl = ProjectionHead(
        input_dim_for_projector, 
        config['projector_hidden_dim'], 
        config['projector_out_dim']
    )
    
    # 构建SSL模型
    ssl_model = MultimodalBTModel(
        s2_backbone_ssl, 
        s1_backbone_ssl, 
        projector_ssl,
        fusion_method=config['fusion_method'], 
        return_repr=True
    ).to(device)
    
    # 加载checkpoint
    logging.info(f"从 {config['checkpoint_path']} 加载checkpoint")
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
    
    # 处理FSDP状态字典
    state_dict = checkpoint[state_key]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]  # 移除前缀
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # 加载状态字典
    ssl_model.load_state_dict(new_state_dict, strict=True)
    
    # 冻结骨干网络
    for param in ssl_model.s2_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.s1_backbone.parameters():
        param.requires_grad = False
    for param in ssl_model.dim_reducer.parameters():
        param.requires_grad = False
    
    # 构建分类头和下游模型
    classification_in_dim = latent_dim
    # clf_head = ClassificationHead(input_dim=classification_in_dim, num_classes=num_classes).to(device)
    clf_head = LinearProbeHead(input_dim=classification_in_dim, num_classes=num_classes).to(device)
    downstream_model = MultimodalDownstreamModel(
        s2_backbone=ssl_model.s2_backbone,
        s1_backbone=ssl_model.s1_backbone,
        head=clf_head,
        dim_reducer=ssl_model.dim_reducer,
        fusion_method=config['fusion_method']
    ).to(device)
    
    # 优化器和损失函数
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, downstream_model.parameters()),
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 训练循环
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join(config['output_dir'], "checkpoints", "ethiopia_crop_best.pt")
    
    for epoch in range(config['epochs']):
        # 训练
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
        
        # 训练指标
        avg_train_loss = train_loss_sum / len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        
        logging.info(f"[Epoch {epoch+1}] Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1
            }, step=epoch)
        
        # 验证
        # Epoch大于10时才进行验证
        if epoch < 10:
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
        
        # 验证指标
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        logging.info(f"[Epoch {epoch+1}] Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch+1,
                "val_acc": val_acc,
                "val_f1": val_f1
            }, step=epoch)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch+1
            torch.save(downstream_model.state_dict(), checkpoint_path)
            logging.info(f"在epoch {best_epoch}保存了最佳模型")
    
    # 使用最佳模型进行最终评估
    logging.info(f"从epoch {best_epoch}加载最佳checkpoint进行最终评估")
    downstream_model.load_state_dict(torch.load(checkpoint_path))
    downstream_model.eval()
    
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for s2_batch, s1_batch, labels_batch in tqdm.tqdm(val_loader, desc="Final Evaluation"):
            s2_batch = s2_batch.to(device)
            s1_batch = s1_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            logits = downstream_model(s2_batch, s1_batch)
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            val_targets.extend(labels_batch.cpu().numpy())
    
    # 最终指标
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average='weighted')
    report = classification_report(val_targets, val_preds, target_names=config['class_names'], digits=4)
    conf_mat = confusion_matrix(val_targets, val_preds)
    
    logging.info(f"最终验证 Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    logging.info(f"分类报告:\n{report}")
    
    cm_df = pd.DataFrame(conf_mat, index=config['class_names'], columns=config['class_names'])
    logging.info(f"\n混淆矩阵:\n{cm_df}")
    
    # 保存结果
    results_path = os.path.join(config['output_dir'], "logs", "ethiopia_crop_results.txt")
    with open(results_path, "w") as f:
        f.write(f"最佳epoch: {best_epoch}\n")
        f.write(f"最终验证 Acc: {val_acc:.4f}\n")
        f.write(f"最终验证 F1: {val_f1:.4f}\n")
        f.write(f"分类报告:\n{report}\n")
        f.write(f"\n混淆矩阵:\n{cm_df}\n")
    
    logging.info(f"结果已保存到 {results_path}")
    
    if use_wandb:
        wandb.log({
            "final_val_acc": val_acc, 
            "final_val_f1": val_f1,
            "best_epoch": best_epoch
        })
        wandb_run.finish()

if __name__ == "__main__":
    main()