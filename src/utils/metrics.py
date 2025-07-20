# src/utils/metrics.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pandas as pd
from collections import Counter, defaultdict
import torch
from sklearn.decomposition import PCA
import os


def rankme(z, eps=1e-7):
    # Convert to float32 for SVD
    if z.dtype == torch.float16:
        z = z.to(torch.float32)
    # Perform SVD
    s = z.svd(compute_uv=False)[1]
    # Calculate rank metrics
    p = s / (s.sum() + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    rankme_score = entropy / torch.log(torch.tensor(float(len(s))))
    return rankme_score

def linear_probe_evaluate(model, val_loader, field_ids=None, field_data_path=None, 
                         training_ratio=0.3, val_test_split_ratio=0.5, 
                         classifier_type='lr', num_inference=1, device='cuda'):
    """
    使用验证集计算模型嵌入后训练出的分类器的表现，
    返回 accuracy, weighted F1 score 以及混淆矩阵。
    
    参数:
    - model: 训练好的模型
    - val_loader: 验证数据加载器
    - field_ids: 可选，用于基于字段的拆分
    - field_data_path: 可选，字段数据的 CSV 文件路径
    - training_ratio: 用于训练的字段面积比例，默认为 0.3
    - val_test_split_ratio: 验证集与测试集分割比例，默认为 0.5
    - classifier_type: 分类器类型，'lr' 或 'rf'，默认为 'lr'
    - num_inference: 推理次数，取平均，默认为 1
    - device: 计算设备，默认为 'cuda'
    
    返回值:
    - accuracy, f1_score, 混淆矩阵, 分类报告
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    model.eval()
    
    # 初始化存储嵌入和标签的容器
    embeddings_dict = {}  # {position: embedding}
    labels_dict = {}      # {position: label}
    field_id_dict = {}    # {position: field_id}
    
    # 收集每个位置的嵌入
    with torch.no_grad():
        for s2_sample, s1_sample, label, field_id, pos in val_loader:
            # 将tensor转为CPU上的numpy数组以便处理
            batch_s2 = s2_sample.to(device)
            batch_s1 = s1_sample.to(device)
            batch_label = label.cpu().numpy()
            if isinstance(field_id, torch.Tensor):
                batch_field_id = field_id.cpu().numpy()
            batch_pos = pos.cpu().numpy()
            
            # 执行多次推理并平均
            embeddings_sum = None
            for _ in range(num_inference):
                out = model(batch_s2, batch_s1)
                if isinstance(out, tuple):
                    out = out[1]  # 获取representation而非projection
                
                if embeddings_sum is None:
                    embeddings_sum = out.cpu().numpy()
                else:
                    embeddings_sum += out.cpu().numpy()
            
            # 平均嵌入
            emb = embeddings_sum / num_inference
            
            # 存储结果，包括位置和field_id
            for i in range(len(batch_pos)):
                position = tuple(batch_pos[i])
                embeddings_dict[position] = emb[i]
                labels_dict[position] = batch_label[i]
                if isinstance(field_id, torch.Tensor):
                    field_id_dict[position] = batch_field_id[i]
    
    # 如果有字段数据，则根据字段进行分割
    if field_data_path is not None and os.path.exists(field_data_path) and field_ids is not None:
        # 加载字段数据
        field_data_df = pd.read_csv(field_data_path)
        area_summary = field_data_df.groupby('SNAR_CODE')['area_m2'].sum().reset_index()
        area_summary.rename(columns={'area_m2': 'total_area'}, inplace=True)
        
        # 收集训练集的field ID
        train_fids = []
        for _, row in area_summary.iterrows():
            sn_code = row['SNAR_CODE']
            total_area = row['total_area']
            target_area = total_area * training_ratio
            rows_sncode = field_data_df[field_data_df['SNAR_CODE'] == sn_code].sort_values(by='area_m2')
            selected_fids = []
            selected_area_sum = 0
            for _, r2 in rows_sncode.iterrows():
                if selected_area_sum < target_area:
                    selected_fids.append(int(r2['fid_1']))
                    selected_area_sum += r2['area_m2']
                else:
                    break
            train_fids.extend(selected_fids)
        
        train_fids = set(train_fids)
        
        # 将剩余的field ID分为验证集和测试集
        all_fields = set(field_data_df['fid_1'].unique().astype(int))
        remaining = list(all_fields - train_fids)
        remaining = np.array(remaining)
        np.random.shuffle(remaining)
        val_count = int(len(remaining) * val_test_split_ratio)
        val_fids = set(remaining[:val_count])
        test_fids = set(remaining[val_count:])
        
        # 基于field ID分配数据
        train_positions = []
        val_positions = []
        test_positions = []
        
        for pos, fid in field_id_dict.items():
            if fid in train_fids:
                train_positions.append(pos)
            elif fid in val_fids:
                val_positions.append(pos)
            elif fid in test_fids:
                test_positions.append(pos)
        
        # 随机打乱各集合中的位置
        # np.random.shuffle(train_positions)
        # np.random.shuffle(val_positions)
        # np.random.shuffle(test_positions)
        
        # 创建训练和测试数据集
        X_train = np.array([embeddings_dict[pos] for pos in train_positions])
        y_train = np.array([labels_dict[pos] for pos in train_positions])
        X_test = np.array([embeddings_dict[pos] for pos in test_positions])
        y_test = np.array([labels_dict[pos] for pos in test_positions])
        
        print(f"Field-based split: Train: {len(train_positions)}, Val: {len(val_positions)}, Test: {len(test_positions)}")
    else:
        # 如果没有字段数据，就随机分割
        positions = list(embeddings_dict.keys())
        np.random.shuffle(positions)
        
        embeddings = np.array([embeddings_dict[pos] for pos in positions])
        labels_arr = np.array([labels_dict[pos] for pos in positions])
        
        # 随机选择training_ratio%的数据用于训练
        N = len(embeddings)
        split = int(training_ratio * N)
        X_train = embeddings[:split]
        y_train = labels_arr[:split]
        X_test = embeddings[split:]
        y_test = labels_arr[split:]
        
        print(f"Random split: Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 选择分类器
    if classifier_type.lower() == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:  # 默认为LR
        clf = LogisticRegression(max_iter=10000, n_jobs=-1)
    
    # 训练分类器
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    # 计算评估指标
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, digits=4)
    
    return acc, f1, cm, cr




def linear_probe_evaluate_64_fixed(model, val_loader, field_ids=None, field_data_path=None, 
                             training_ratio=0.3, val_test_split_ratio=0.5, 
                             classifier_type='lr', num_inference=1, device='cuda'):
    """
    Evaluate model using linear probe with attention masks for 64 timesteps data structure.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        field_ids: Optional, for field-based splitting
        field_data_path: Optional, CSV file path for field data
        training_ratio: Proportion of field area for training, default 0.3
        val_test_split_ratio: Validation/test split ratio, default 0.5
        classifier_type: Classifier type, 'lr' or 'rf', default 'lr'
        num_inference: Number of inference passes to average, default 1
        device: Computation device, default 'cuda'
    
    Returns:
        accuracy, f1_score, confusion matrix, classification report
    """
    import numpy as np
    import torch
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    model.eval()
    
    # Initialize containers to store embeddings and labels
    embeddings_dict = {}  # {position: embedding}
    labels_dict = {}      # {position: label}
    field_id_dict = {}    # {position: field_id}
    
    # Collect embeddings for each position
    with torch.no_grad():
        for batch_data in val_loader:
            # Unpack data - now includes masks
            s2_sample, s1_sample, label, field_id, pos, s2_mask, s1_mask = batch_data
            
            # Convert tensors to CPU numpy arrays for processing
            batch_s2 = s2_sample.to(device)
            batch_s1 = s1_sample.to(device)
            batch_label = label.cpu().numpy()
            batch_s2_mask = s2_mask.to(device)
            batch_s1_mask = s1_mask.to(device)
            
            if isinstance(field_id, torch.Tensor):
                batch_field_id = field_id.cpu().numpy()
            batch_pos = pos.cpu().numpy()
            
            # Perform multiple inferences and average
            embeddings_sum = None
            for _ in range(num_inference):
                # Forward pass with masks
                out = model(batch_s2, batch_s1, batch_s2_mask, batch_s1_mask)
                if isinstance(out, tuple):
                    out = out[1]  # Get representation instead of projection
                
                if embeddings_sum is None:
                    embeddings_sum = out.cpu().numpy()
                else:
                    embeddings_sum += out.cpu().numpy()
            
            # Average embeddings
            emb = embeddings_sum / num_inference
            
            # Store results, including position and field_id
            for i in range(len(batch_pos)):
                position = tuple(batch_pos[i])
                embeddings_dict[position] = emb[i]
                labels_dict[position] = batch_label[i]
                if isinstance(field_id, torch.Tensor):
                    field_id_dict[position] = batch_field_id[i]
    
    # If field data is available, split based on fields
    if field_data_path is not None and os.path.exists(field_data_path) and field_ids is not None:
        # Load field data
        field_data_df = pd.read_csv(field_data_path)
        area_summary = field_data_df.groupby('SNAR_CODE')['area_m2'].sum().reset_index()
        area_summary.rename(columns={'area_m2': 'total_area'}, inplace=True)
        
        # Collect field IDs for training set
        train_fids = []
        for _, row in area_summary.iterrows():
            sn_code = row['SNAR_CODE']
            total_area = row['total_area']
            target_area = total_area * training_ratio
            rows_sncode = field_data_df[field_data_df['SNAR_CODE'] == sn_code].sort_values(by='area_m2')
            selected_fids = []
            selected_area_sum = 0
            for _, r2 in rows_sncode.iterrows():
                if selected_area_sum < target_area:
                    selected_fids.append(int(r2['fid_1']))
                    selected_area_sum += r2['area_m2']
                else:
                    break
            train_fids.extend(selected_fids)
        
        train_fids = set(train_fids)
        
        # Split remaining field IDs into validation and test sets
        all_fields = set(field_data_df['fid_1'].unique().astype(int))
        remaining = list(all_fields - train_fids)
        remaining = np.array(remaining)
        np.random.shuffle(remaining)
        val_count = int(len(remaining) * val_test_split_ratio)
        val_fids = set(remaining[:val_count])
        test_fids = set(remaining[val_count:])
        
        # Allocate data based on field IDs
        train_positions = []
        val_positions = []
        test_positions = []
        
        for pos, fid in field_id_dict.items():
            if fid in train_fids:
                train_positions.append(pos)
            elif fid in val_fids:
                val_positions.append(pos)
            elif fid in test_fids:
                test_positions.append(pos)
        
        # Create training and test datasets
        X_train = np.array([embeddings_dict[pos] for pos in train_positions])
        y_train = np.array([labels_dict[pos] for pos in train_positions])
        
        X_test = np.array([embeddings_dict[pos] for pos in test_positions])
        y_test = np.array([labels_dict[pos] for pos in test_positions])
        print(f"Field-based split: Train: {len(train_positions)}, Test: {len(test_positions)}")
    else:
        # If no field data, use random splitting
        positions = list(embeddings_dict.keys())
        np.random.shuffle(positions)
        
        embeddings = np.array([embeddings_dict[pos] for pos in positions])
        labels_arr = np.array([labels_dict[pos] for pos in positions])
        
        # Randomly select training_ratio% of data for training
        N = len(embeddings)
        split = int(training_ratio * N)
        X_train = embeddings[:split]
        y_train = labels_arr[:split]
        X_test = embeddings[split:]
        y_test = labels_arr[split:]
        
        print(f"Random split: Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Choose classifier
    if classifier_type.lower() == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:  # Default to LogisticRegression
        clf = LogisticRegression(max_iter=10000, n_jobs=-1)
    
    # Train classifier
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    # Calculate evaluation metrics
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, digits=4)
    
    return acc, f1, cm, cr

