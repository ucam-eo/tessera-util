import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score, balanced_accuracy_score
from tqdm import tqdm
import logging
from einops import rearrange
import pandas as pd
from datetime import datetime
import argparse
import xgboost as xgb  # Added XGBoost import
from scipy.ndimage import zoom  # For resizing arrays

# -----------------------------------------------------------------------------
# Dataset类：支持训练和测试划分
# -----------------------------------------------------------------------------
class LandClassificationDataset(Dataset):
    def __init__(self, representations, labels):
        """
        Args:
            representations (ndarray): 特征表示，形状 (N, D)
            labels (ndarray): 标签，形状 (N,)
        """
        self.representations = representations
        self.labels = labels

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.representations[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# -----------------------------------------------------------------------------
# 分类头网络定义
# -----------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.ln1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.ln2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu1(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# -----------------------------------------------------------------------------
# 对每个类别进行采样，创建训练和测试数据集
# -----------------------------------------------------------------------------
def create_train_test_split(representations, labels, sample_per_pixel):
    """
    为每个类别随机选择sample_per_pixel个像素作为训练样本，其余作为测试样本
    
    Args:
        representations (ndarray): 特征表示，形状 (H, W, D)
        labels (ndarray): 标签，形状 (H, W)
        sample_per_pixel (int): 每个类别选择的样本数
        
    Returns:
        train_reps, train_labels, test_reps, test_labels (tuple): 训练和测试数据
    """
    # 重新整形为(N, D)
    h, w, d = representations.shape
    flat_reps = representations.reshape(-1, d)
    flat_labels = labels.reshape(-1)
    
    # 排除标签为-1的无效像素
    valid_mask = flat_labels != -1
    valid_reps = flat_reps[valid_mask]
    valid_labels = flat_labels[valid_mask]
    
    unique_classes = np.unique(valid_labels)
    train_indices = []
    test_indices = []
    
    # 为每个类别选择训练和测试样本
    for cls in unique_classes:
        cls_indices = np.where(valid_labels == cls)[0]
        np.random.shuffle(cls_indices)
        
        # 如果该类别的样本数小于sample_per_pixel，则全部用于训练
        if len(cls_indices) <= sample_per_pixel:
            train_indices.extend(cls_indices)
        else:
            train_indices.extend(cls_indices[:sample_per_pixel])
            test_indices.extend(cls_indices[sample_per_pixel:])
    
    # 创建训练和测试数据集
    train_reps = valid_reps[train_indices]
    train_labels = valid_labels[train_indices]
    test_reps = valid_reps[test_indices]
    test_labels = valid_labels[test_indices]
    
    return train_reps, train_labels, test_reps, test_labels

# -----------------------------------------------------------------------------
# 形状对齐函数：调整representations以匹配labels的空间维度
# -----------------------------------------------------------------------------
def align_shapes(representations, labels):
    """
    调整representations的空间维度以匹配labels
    
    Args:
        representations (ndarray): 特征表示，形状 (H1, W1, D)
        labels (ndarray): 标签，形状 (H2, W2)
        
    Returns:
        aligned_representations (ndarray): 调整后的特征表示，形状 (H2, W2, D)
    """
    rep_h, rep_w, rep_d = representations.shape
    label_h, label_w = labels.shape
    
    logging.info(f"Original representations shape: ({rep_h}, {rep_w}, {rep_d})")
    logging.info(f"Labels shape: ({label_h}, {label_w})")
    
    if rep_h == label_h and rep_w == label_w:
        logging.info("Shapes already match, no alignment needed")
        return representations
    
    # 计算缩放因子
    zoom_h = label_h / rep_h
    zoom_w = label_w / rep_w
    
    logging.info(f"Zoom factors: height={zoom_h:.4f}, width={zoom_w:.4f}")
    
    # 使用scipy的zoom函数进行调整
    # order=1 使用双线性插值，对于连续值较合适
    aligned_representations = zoom(representations, (zoom_h, zoom_w, 1), order=1)
    
    logging.info(f"Aligned representations shape: {aligned_representations.shape}")
    
    # 验证形状
    assert aligned_representations.shape[0] == label_h, f"Height mismatch after alignment: {aligned_representations.shape[0]} != {label_h}"
    assert aligned_representations.shape[1] == label_w, f"Width mismatch after alignment: {aligned_representations.shape[1]} != {label_w}"
    
    return aligned_representations

# -----------------------------------------------------------------------------
# 评估函数：计算各种性能指标
# -----------------------------------------------------------------------------
def evaluate_model(model, data_loader, device, num_classes, class_names=None):
    """
    评估模型性能并返回详细的指标
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 运行设备
        num_classes: 类别数量
        class_names: 类别名称列表
        
    Returns:
        metrics (dict): 包含各种性能指标的字典
        report_str (str): 格式化的分类报告
        cm_str (str): 格式化的混淆矩阵
    """
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for reps, labels_batch in data_loader:
            reps, labels_batch = reps.to(device), labels_batch.to(device)
            outputs = model(reps)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(labels_batch.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    return calculate_metrics(all_preds, all_targets, num_classes, class_names)

# -----------------------------------------------------------------------------
# XGBoost评估函数：计算各种性能指标
# -----------------------------------------------------------------------------
def evaluate_xgboost(model, X_test, y_test, num_classes, class_names=None):
    """
    评估XGBoost模型性能并返回详细的指标
    
    Args:
        model: 训练好的XGBoost模型
        X_test: 测试特征
        y_test: 测试标签
        num_classes: 类别数量
        class_names: 类别名称列表
        
    Returns:
        metrics (dict): 包含各种性能指标的字典
        report_str (str): 格式化的分类报告
        cm_str (str): 格式化的混淆矩阵
    """
    # 将测试数据转换为DMatrix
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 预测测试集
    all_preds = model.predict(dtest)
    all_targets = y_test
    
    return calculate_metrics(all_preds, all_targets, num_classes, class_names)

# -----------------------------------------------------------------------------
# 通用指标计算函数
# -----------------------------------------------------------------------------
def calculate_metrics(all_preds, all_targets, num_classes, class_names=None):
    """
    计算预测结果的各种性能指标
    
    Args:
        all_preds: 预测标签
        all_targets: 真实标签
        num_classes: 类别数量
        class_names: 类别名称列表
        
    Returns:
        metrics (dict): 包含各种性能指标的字典
        report_str (str): 格式化的分类报告
        cm_str (str): 格式化的混淆矩阵
    """
    # 计算整体性能指标
    overall_accuracy = accuracy_score(all_targets, all_preds)
    overall_f1 = f1_score(all_targets, all_preds, average='weighted')
    overall_precision = precision_score(all_targets, all_preds, average='weighted')
    overall_recall = recall_score(all_targets, all_preds, average='weighted')
    
    # 计算平衡准确率（宏平均召回率）
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    
    # 计算每个类别的性能指标
    class_report = classification_report(all_targets, all_preds, output_dict=True)
    
    # 创建结果字典
    metrics = {
        'overall_accuracy': overall_accuracy,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'balanced_acc': balanced_acc
    }
    
    # 添加每个类别的指标
    for i in range(num_classes):
        if str(i) in class_report:
            cls_metrics = class_report[str(i)]
            metrics[f'class_{i+1}_precision'] = cls_metrics['precision']
            metrics[f'class_{i+1}_recall'] = cls_metrics['recall']
            metrics[f'class_{i+1}_f1'] = cls_metrics['f1-score']
    
    # 生成易读的分类报告
    if class_names:
        target_names = class_names
        report_str = classification_report(all_targets, all_preds, target_names=target_names)
    else:
        report_str = classification_report(all_targets, all_preds)
    
    # 生成易读的混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 格式化混淆矩阵以提高可读性
    if class_names:
        cm_str = "Confusion Matrix:\n"
        # 添加列标签
        header = "       "
        for i in range(len(class_names)):
            header += f"{i:<8}"
        cm_str += header + "\n"
        
        # 添加行和数据
        for i in range(len(class_names)):
            row_str = f"{i:<6} "
            for j in range(len(class_names)):
                row_str += f"{cm[i, j]:<8}"
            cm_str += row_str + f" | {class_names[i]}\n"
    else:
        cm_str = "Confusion Matrix:\n" + str(cm)
    
    return metrics, report_str, cm_str

# -----------------------------------------------------------------------------
# 主函数：运行多次实验并保存结果
# -----------------------------------------------------------------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Land Classification with Multiple Experiments')
    parser.add_argument('--method', type=str, default='mlp', choices=['mlp', 'xgboost'], 
                        help='Classification method: mlp or xgboost')
    parser.add_argument('--sample_per_pixel', type=int, default=10, help='Number of samples per class')
    parser.add_argument('--num_experiment', type=int, default=200, help='Number of experiments to run')
    parser.add_argument('--result_dir', type=str, default='/mnt/e/Codes/btfm4rs/data/downstream/austrian_crop/logs', help='Directory to save results')
    parser.add_argument('--representation_path', type=str, 
                        # default="data/representation/representations_fsdp_20250427_084307_repreat_1_downsample_100.npy", 
                        default="data/representation/austria_Presto_embeddings_100m.npy", 
                        # default="data/representation/austrian_crop_EFM_v1.0_Embeddings_2022_100x_downsampled.npy", 
                        help='Path to representation file')
    parser.add_argument('--label_path', type=str, 
                        default="data/downstream/austrian_crop/fieldtype_17classes_downsample_100.npy", 
                        help='Path to label file')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/downstream", 
                        help='Directory to save checkpoints')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size')
    
    # XGBoost参数
    parser.add_argument('--xgb_max_depth', type=int, default=6, help='Maximum depth of XGBoost trees')
    parser.add_argument('--xgb_eta', type=float, default=0.3, help='Learning rate for XGBoost')
    parser.add_argument('--xgb_subsample', type=float, default=0.8, help='Subsample ratio for XGBoost')
    parser.add_argument('--xgb_colsample_bytree', type=float, default=0.8, help='Column sample ratio for XGBoost')
    parser.add_argument('--xgb_num_round', type=int, default=100, help='Number of boosting rounds for XGBoost')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    
    # 创建结果和检查点目录
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 定义类别名称
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
    
    # 定义CSV和TXT文件名
    representation_file_name = os.path.basename(args.representation_path)
    representation_file_name = os.path.splitext(representation_file_name)[0]
    method_name = args.method  # 添加方法名到文件名
    checkpoint_name = f"{representation_file_name}_{method_name}_{timestamp}"
    result_csv_path = os.path.join(args.result_dir, 
                                f"{args.sample_per_pixel}_{args.num_experiment}_{checkpoint_name}.csv")
    report_txt_path = os.path.join(args.result_dir, 
                                f"{args.sample_per_pixel}_{args.num_experiment}_{checkpoint_name}.txt")
    
    # -------------------------
    # 加载representation和标签
    # -------------------------
    logging.info(f"Loading data from {args.representation_path} and {args.label_path}")
    logging.info(f"Using method: {args.method}")
    representations = np.load(args.representation_path)  # (H, W, D)
    labels = np.load(args.label_path).astype(np.int64)   # (H, W)
    
    # -------------------------
    # 检查并对齐空间维度
    # -------------------------
    logging.info("Checking spatial dimensions...")
    representations = align_shapes(representations, labels)
    
    # 再次验证形状
    logging.info(f"Final representations shape: {representations.shape}")
    logging.info(f"Final labels shape: {labels.shape}")
    assert representations.shape[:2] == labels.shape, f"Shape mismatch after alignment: {representations.shape[:2]} != {labels.shape}"
    
    # 记录原始标签信息
    original_unique_labels = np.unique(labels)
    logging.info(f"Original unique labels: {original_unique_labels}")
    
    # 重映射标签：将背景类排除，有效类从0开始重新编号
    # 首先创建一个映射字典，将0映射到-1（后续会被过滤掉），其他标签依次映射为0到16
    label_map = {0: -1}  # 背景类映射为-1
    valid_class_counter = 0
    for label in sorted(original_unique_labels):
        if label != 0:  # 如果不是背景类
            label_map[label] = valid_class_counter
            valid_class_counter += 1
    
    # 应用映射
    labels_remapped = np.vectorize(label_map.get)(labels)
    
    # 记录有效类的数量
    num_valid_classes = valid_class_counter
    logging.info(f"Number of valid classes after excluding background: {num_valid_classes}")
    logging.info(f"Label mapping (original->new): {label_map}")
    
    # 检查重映射后的数据统计
    valid_pixels = np.sum(labels_remapped != -1)
    total_pixels = labels_remapped.size
    logging.info(f"Valid pixels: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.2f}%)")
    
    # -------------------------
    # 模型基本配置
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    input_dim = representations.shape[-1]
    logging.info(f"Input dimension (number of features): {input_dim}")
    
    # 用于存储所有实验结果的列表
    all_results = []
    
    # -------------------------
    # 运行多次实验
    # -------------------------
    for exp_id in range(args.num_experiment):
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting experiment {exp_id+1}/{args.num_experiment}")
        logging.info(f"{'='*60}")
        
        # 创建训练和测试集 - 使用重映射后的标签
        train_reps, train_labels, test_reps, test_labels = create_train_test_split(
            representations, labels_remapped, args.sample_per_pixel)
        
        logging.info(f"Exp {exp_id+1}: Train size: {len(train_reps)}, Test size: {len(test_reps)}")
        
        # 统计每个类别的训练样本数
        unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
        logging.info(f"Exp {exp_id+1}: Training samples per class:")
        for label, count in zip(unique_train_labels, train_counts):
            if label < len(class_names):
                logging.info(f"  Class {label} ({class_names[label]}): {count} samples")
        
        # 根据所选方法执行不同的训练和评估流程
        if args.method == 'mlp':
            # -------------------------
            # MLP方法：使用PyTorch模型
            # -------------------------
            # 创建数据集和数据加载器
            train_dataset = LandClassificationDataset(train_reps, train_labels)
            test_dataset = LandClassificationDataset(test_reps, test_labels)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            # 创建模型、损失函数、优化器
            model = ClassificationHead(input_dim, num_valid_classes).to(device)
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            
            # 训练循环
            logging.info(f"Exp {exp_id+1}: Starting MLP training...")
            for epoch in range(args.num_epochs):
                model.train()
                running_loss = 0.0
                train_preds, train_targets = [], []
                
                for reps, labels_batch in train_loader:
                    reps, labels_batch = reps.to(device), labels_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(reps)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    train_targets.extend(labels_batch.cpu().numpy())
                
                # 每个epoch结束后计算训练损失和指标
                epoch_loss = running_loss / len(train_loader)
                train_accuracy = accuracy_score(train_targets, train_preds)
                train_f1 = f1_score(train_targets, train_preds, average='weighted')
                
                # 只打印每10个epoch的信息，避免日志过长
                if (epoch + 1) % 10 == 0:
                    logging.info(f"Exp {exp_id+1} - Epoch [{epoch+1}/{args.num_epochs}] - "
                                f"Train Loss: {epoch_loss:.4f}, "
                                f"Train Accuracy: {train_accuracy:.4f}, "
                                f"Train F1: {train_f1:.4f}")
            
            # 最终测试评估
            logging.info(f"Exp {exp_id+1} - Evaluating on test set")
            test_metrics, report_str, cm_str = evaluate_model(
                model, test_loader, device, num_valid_classes, class_names)
            
        else:
            # -------------------------
            # XGBoost方法：使用XGBoost库
            # -------------------------
            # 设置XGBoost参数
            xgb_params = {
                'max_depth': args.xgb_max_depth,
                'eta': args.xgb_eta,
                'subsample': args.xgb_subsample,
                'colsample_bytree': args.xgb_colsample_bytree,
                'objective': 'multi:softmax',
                'num_class': num_valid_classes,
                'eval_metric': ['merror', 'mlogloss'],
                'tree_method': 'auto',
                'verbosity': 0  # 减少XGBoost的输出
            }
            
            # 创建DMatrix
            dtrain = xgb.DMatrix(train_reps, label=train_labels)
            dtest = xgb.DMatrix(test_reps, label=test_labels)
            
            # 训练XGBoost模型
            logging.info(f"Exp {exp_id+1} - Training XGBoost model...")
            
            # 定义评估列表
            evals = [(dtrain, 'train'), (dtest, 'test')]
            evals_result = {}
            
            # 训练模型
            xgb_model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=args.xgb_num_round,
                evals=evals,
                evals_result=evals_result,
                verbose_eval=False  # 禁用详细评估输出
            )
            
            # 打印最终的训练结果
            final_train_error = evals_result['train']['merror'][-1]
            final_test_error = evals_result['test']['merror'][-1]
            logging.info(f"Exp {exp_id+1} - XGBoost Final - Train Error: {final_train_error:.4f}, Test Error: {final_test_error:.4f}")
            
            # 最终测试评估
            logging.info(f"Exp {exp_id+1} - Evaluating XGBoost on test set")
            test_metrics, report_str, cm_str = evaluate_xgboost(
                xgb_model, test_reps, test_labels, num_valid_classes, class_names)
        
        # 添加实验ID和基本信息
        result_row = {
            'experiment_id': exp_id + 1,
            'method': args.method,  # 添加方法到结果中
            'sample_per_pixel': args.sample_per_pixel,
            'train_size': len(train_reps),
            'test_size': len(test_reps),
            **test_metrics  # 使用解包操作添加所有测试指标
        }
        
        all_results.append(result_row)
        
        # 打印当前实验的结果
        logging.info(f"Exp {exp_id+1} - Test Accuracy: {test_metrics['overall_accuracy']:.4f}, "
                     f"Test F1: {test_metrics['overall_f1']:.4f}, "
                     f"Balanced Acc: {test_metrics['balanced_acc']:.4f}")
        
        # 保存第一次实验的分类报告和混淆矩阵
        if exp_id == 0:
            with open(report_txt_path, 'w') as f:
                f.write(f"Classification Report and Confusion Matrix for Experiment 1\n")
                f.write(f"Method: {args.method}\n")
                f.write(f"================================================================\n\n")
                f.write(f"Sample per pixel: {args.sample_per_pixel}\n")
                f.write(f"Train size: {len(train_reps)}, Test size: {len(test_reps)}\n\n")
                f.write(f"Classification Report:\n")
                f.write(f"---------------------\n")
                f.write(report_str)
                f.write(f"\n\n")
                f.write(f"Confusion Matrix:\n")
                f.write(f"----------------\n")
                f.write(cm_str)
                f.write(f"\n\n")
                f.write(f"Note: Class indices are 0-indexed and correspond to:\n")
                for i, name in enumerate(class_names):
                    f.write(f"  {i}: {name}\n")
                
            logging.info(f"Saved classification report and confusion matrix to {report_txt_path}")
    
    # -------------------------
    # 将所有结果保存到CSV
    # -------------------------
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(result_csv_path, index=False)
    logging.info(f"\nAll experiment results saved to {result_csv_path}")
    
    # -------------------------
    # 打印结果的统计信息
    # -------------------------
    logging.info("\n" + "="*60)
    logging.info("EXPERIMENT STATISTICS:")
    logging.info("="*60)
    for metric in ['overall_accuracy', 'overall_f1', 'overall_precision', 'overall_recall', 'balanced_acc']:
        values = results_df[metric].values
        logging.info(f"{metric}:")
        logging.info(f"  Mean: {np.mean(values):.4f}")
        logging.info(f"  Std:  {np.std(values):.4f}")
        logging.info(f"  Min:  {np.min(values):.4f}")
        logging.info(f"  Max:  {np.max(values):.4f}")
    
    # 计算并打印每个类别的平均性能
    logging.info("\nPER-CLASS AVERAGE PERFORMANCE:")
    for i in range(num_valid_classes):
        if f'class_{i+1}_f1' in results_df.columns:
            class_f1_values = results_df[f'class_{i+1}_f1'].values
            logging.info(f"Class {i} ({class_names[i]}): F1={np.mean(class_f1_values):.4f} ± {np.std(class_f1_values):.4f}")

if __name__ == "__main__":
    main()