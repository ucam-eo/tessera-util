# src/utils/metrics.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from sklearn.ensemble import RandomForestClassifier

import csv
from sklearn.ensemble import RandomForestClassifier


def rankme(z, eps=1e-7):
    # Convert to float32 for SVD
    # if z.dtype == torch.float16:
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
    
    # ---- 1) 提取原始模型以避免使用编译 (torch.compile) 版本 ----
    # 如果你用的是 FSDP，可以先获取 FSDP 的内部模型:
    if isinstance(model, FSDP):
        eval_model = model.module
    else:
        eval_model = model

    # 如果模型被 torch.compile(...) 包装过，则会有 _orig_mod
    if hasattr(eval_model, '_orig_mod'):
        eval_model = eval_model._orig_mod

    eval_model.eval()
    
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
                out = eval_model(batch_s2, batch_s1)
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
    field_split_success = False
    
    if field_data_path is not None and os.path.exists(field_data_path) and field_ids is not None:
        try:
            # 加载字段数据
            field_data = []
            with open(field_data_path, 'r') as f:
                csv_reader = csv.reader(f)
                header = next(csv_reader)
                
                # 找到必要列 (SNAR_CODE, fid_1, area_m2)
                snar_code_idx = header.index('SNAR_CODE') if 'SNAR_CODE' in header else None
                fid_1_idx = header.index('fid_1') if 'fid_1' in header else None
                area_m2_idx = header.index('area_m2') if 'area_m2' in header else None
                
                if snar_code_idx is None or fid_1_idx is None or area_m2_idx is None:
                    print("Error: Required columns missing in CSV. Found:", header)
                    raise ValueError("CSV missing required columns")
                
                for row in csv_reader:
                    if len(row) <= max(snar_code_idx, fid_1_idx, area_m2_idx):
                        continue
                    snar_code = row[snar_code_idx]
                    try:
                        fid_1 = int(float(row[fid_1_idx]))
                        area_m2 = float(row[area_m2_idx])
                        field_data.append({
                            'SNAR_CODE': snar_code,
                            'fid_1': fid_1,
                            'area_m2': area_m2
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Warning: skip row: {row}, Error: {e}")
                        continue
            
            # 按SNAR_CODE分组并计算总面积
            snar_code_areas = {}
            for entry in field_data:
                sn_code = entry['SNAR_CODE']
                area = entry['area_m2']
                snar_code_areas[sn_code] = snar_code_areas.get(sn_code, 0) + area
            
            area_summary = [
                {'SNAR_CODE': sn_code, 'total_area': total_area}
                for sn_code, total_area in snar_code_areas.items()
            ]
            
            # 逐个作物类型，按面积选择一部分field_id进入训练集
            train_fids = []
            for area_entry in area_summary:
                sn_code = area_entry['SNAR_CODE']
                total_area = area_entry['total_area']
                target_area = total_area * training_ratio
                
                # 找出当前SNAR_CODE的所有字段并按面积排序
                rows_sncode = [x for x in field_data if x['SNAR_CODE'] == sn_code]
                rows_sncode.sort(key=lambda x: x['area_m2'])
                
                selected_fids = []
                selected_area_sum = 0
                for row in rows_sncode:
                    if selected_area_sum < target_area:
                        selected_fids.append(row['fid_1'])
                        selected_area_sum += row['area_m2']
                    else:
                        break
                train_fids.extend(selected_fids)
            
            train_fids = set(train_fids)
            
            # 将剩余的field ID分为验证集和测试集
            all_fields = set(entry['fid_1'] for entry in field_data)
            remaining = list(all_fields - train_fids)
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
            
            # 判断是否足够做训练/验证
            if len(train_positions) > 0 and len(test_positions) > 0:
                # 创建训练和测试数据集
                X_train = np.array([embeddings_dict[pos] for pos in train_positions])
                y_train = np.array([labels_dict[pos] for pos in train_positions])
                X_test = np.array([embeddings_dict[pos] for pos in test_positions])
                y_test = np.array([labels_dict[pos] for pos in test_positions])
                
                print(f"Field-based split: Train: {len(train_positions)}, Val: {len(val_positions)}, Test: {len(test_positions)}")
                field_split_success = True
            else:
                print("Warning: Field-based splitting yielded empty training or test set.")
                field_split_success = False
                
        except Exception as e:
            print(f"Error during field-based splitting: {e}")
            field_split_success = False
    
    # 如果字段拆分不成功，就随机拆分
    if not field_split_success:
        print("Using random splitting as fallback...")
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
    
    # 检查数据合法性
    if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(y_test) == 0:
        print("Error: Empty training or test set after splitting")
        return 0.0, 0.0, np.zeros((1, 1)), "Error: Empty dataset"

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("Warning: Not enough classes in training or test set")
        print(f"Classes in train: {np.unique(y_train)}")
        print(f"Classes in test: {np.unique(y_test)}")
        return 0.0, 0.0, np.zeros((1, 1)), "Error: Not enough classes"
    
    # 选择分类器
    if classifier_type.lower() == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:  # 默认为LR
        clf = LogisticRegression(max_iter=100000, n_jobs=-1)
    
    # 训练分类器
    print(f"Training {classifier_type} classifier with {len(X_train)} samples...")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    # 计算评估指标
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, digits=4)
    
    return acc, f1, cm, cr



def linear_probe_evaluate_64_fixed(
    model,
    val_loader,
    field_ids=None,
    field_data_path=None,
    training_ratio=0.3,
    val_test_split_ratio=0.5,
    classifier_type='lr',
    num_inference=1,
    device='cuda'
):
    """
    Evaluate model using linear probe with attention masks for 64 timesteps data structure.

    Args:
        model: Trained (possibly compiled/FSDP/DDP) model
        val_loader: Validation DataLoader
        field_ids: Optional, for field-based splitting
        field_data_path: Optional, CSV file path for field data
        training_ratio: Proportion of field area for training, default 0.3
        val_test_split_ratio: Validation/test split ratio, default 0.5
        classifier_type: Classifier type, 'lr' or 'rf', default 'lr'
        num_inference: Number of inference passes to average, default 1
        device: Computation device, default 'cuda'

    Returns:
        (acc, f1, cm, cr):
            - accuracy (float)
            - F1 score (float)
            - confusion matrix (numpy array)
            - classification report (string)
    """

    # ---- 1) 提取原始模型以避免使用编译 (torch.compile) 版本 ----
    # 如果你用的是 DDP，可以先获取 DDP 的内部模型:
    if isinstance(model, DDP):
        eval_model = model.module
    else:
        eval_model = model

    # 如果模型被 torch.compile(...) 包装过，则会有 _orig_mod
    if hasattr(eval_model, '_orig_mod'):
        eval_model = eval_model._orig_mod

    # 如果你使用了 FSDP，也可以类似地判断并获取 eval_model.module
    # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    # if isinstance(eval_model, FSDP):
    #     eval_model = eval_model.module
    #
    # 同理如果还嵌套了 compile，需要再次检查 _orig_mod，可视情况处理

    eval_model.eval()

    # ---- 2) 收集验证集样本的 embedding ----
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize containers to store embeddings and labels
    embeddings_dict = {}  # {position: embedding}
    labels_dict = {}      # {position: label}
    field_id_dict = {}    # {position: field_id}

    with torch.no_grad():
        for batch_data in val_loader:
            # Unpack data (including masks)
            s2_sample, s1_sample, label, field_id, pos, s2_mask, s1_mask = batch_data

            batch_s2 = s2_sample.to(device)
            batch_s1 = s1_sample.to(device)
            batch_label = label.cpu().numpy()
            batch_s2_mask = s2_mask.to(device)
            batch_s1_mask = s1_mask.to(device)

            if isinstance(field_id, torch.Tensor):
                batch_field_id = field_id.cpu().numpy()

            batch_pos = pos.cpu().numpy()

            # 多次 forward，取平均 embedding
            embeddings_sum = None
            for _ in range(num_inference):
                out = eval_model(batch_s2, batch_s1, batch_s2_mask, batch_s1_mask)
                # 若模型 forward 返回 (z, repr) 这种 tuple，就取第二项 repr
                if isinstance(out, tuple):
                    out = out[1]  # representation

                emb_this = out.cpu().numpy()
                if embeddings_sum is None:
                    embeddings_sum = emb_this
                else:
                    embeddings_sum += emb_this
            
            emb = embeddings_sum / num_inference

            # 存储到字典
            for i in range(len(batch_pos)):
                position = tuple(batch_pos[i])
                embeddings_dict[position] = emb[i]
                labels_dict[position] = batch_label[i]
                if isinstance(field_id, torch.Tensor):
                    field_id_dict[position] = batch_field_id[i]
        print(f"Embedding shape: {emb_this.shape}")

    print(f"Collected embeddings for {len(embeddings_dict)} positions")
    if field_id_dict:
        print(f"Number of unique field IDs: {len(set(field_id_dict.values()))}")
    else:
        print("No field IDs found (field_id_dict is empty)")

    # ---- 3) 如果指定了 field_data_path，尝试基于地块做拆分，否则随机拆分 ----
    field_split_success = False

    if field_data_path is not None and os.path.exists(field_data_path) and field_ids is not None:
        try:
            # Load CSV field data
            field_data = []
            with open(field_data_path, 'r') as f:
                csv_reader = csv.reader(f)
                header = next(csv_reader)
                print(f"CSV headers: {header}")

                # 找到必要列 (SNAR_CODE, fid_1, area_m2)
                snar_code_idx = header.index('SNAR_CODE') if 'SNAR_CODE' in header else None
                fid_1_idx = header.index('fid_1') if 'fid_1' in header else None
                area_m2_idx = header.index('area_m2') if 'area_m2' in header else None

                if snar_code_idx is None or fid_1_idx is None or area_m2_idx is None:
                    print("Error: Required columns missing in CSV. Found:", header)
                    raise ValueError("CSV missing required columns")

                for row in csv_reader:
                    if len(row) <= max(snar_code_idx, fid_1_idx, area_m2_idx):
                        continue
                    snar_code = row[snar_code_idx]
                    try:
                        fid_1 = int(float(row[fid_1_idx]))
                        area_m2 = float(row[area_m2_idx])
                        field_data.append({
                            'SNAR_CODE': snar_code,
                            'fid_1': fid_1,
                            'area_m2': area_m2
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Warning: skip row: {row}, Error: {e}")
                        continue

            print(f"Loaded {len(field_data)} entries from field data CSV")

            csv_field_ids = set(entry['fid_1'] for entry in field_data)
            dataset_field_ids = set(field_id_dict.values()) if field_id_dict else set()
            common_field_ids = csv_field_ids.intersection(dataset_field_ids)

            print(f"Field IDs in CSV: {len(csv_field_ids)}")
            print(f"Field IDs in dataset: {len(dataset_field_ids)}")
            print(f"Common field IDs: {len(common_field_ids)}")

            if len(common_field_ids) == 0:
                raise ValueError("No common field IDs between CSV and dataset")

            # 统计每种 SNAR_CODE 的总面积
            snar_code_areas = {}
            for entry in field_data:
                sn_code = entry['SNAR_CODE']
                area = entry['area_m2']
                snar_code_areas[sn_code] = snar_code_areas.get(sn_code, 0) + area

            area_summary = [
                {'SNAR_CODE': sn_code, 'total_area': total_area}
                for sn_code, total_area in snar_code_areas.items()
            ]
            print(f"Found {len(area_summary)} unique crop types (SNAR_CODE)")

            # 逐个作物类型，按面积选择一部分 field_id 进入训练集
            train_fids = []
            for area_entry in area_summary:
                sn_code = area_entry['SNAR_CODE']
                total_area = area_entry['total_area']
                target_area = total_area * training_ratio

                rows_sncode = [x for x in field_data if x['SNAR_CODE'] == sn_code]
                rows_sncode.sort(key=lambda x: x['area_m2'])

                selected_fids = []
                selected_area_sum = 0
                for row in rows_sncode:
                    if selected_area_sum < target_area:
                        selected_fids.append(row['fid_1'])
                        selected_area_sum += row['area_m2']
                    else:
                        break
                train_fids.extend(selected_fids)

            train_fids = set(train_fids)
            print(f"Selected {len(train_fids)} field IDs for training")

            # 剩余 ID 再分成 val/test
            all_fields = set(x['fid_1'] for x in field_data)
            remaining = list(all_fields - train_fids)
            np.random.shuffle(remaining)
            val_count = int(len(remaining) * val_test_split_ratio)
            val_fids = set(remaining[:val_count])
            test_fids = set(remaining[val_count:])

            # 根据 fid 分配 position
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

            print(f"Field-based split: Train={len(train_positions)}, Val={len(val_positions)}, Test={len(test_positions)}")

            # 判断是否足够做训练/验证
            if len(train_positions) > 0 and (len(val_positions) > 0 or len(test_positions) > 0):
                X_train = np.array([embeddings_dict[pos] for pos in train_positions])
                y_train = np.array([labels_dict[pos] for pos in train_positions])
                # 验证集直接抛弃即可
                X_test = np.array([embeddings_dict[pos] for pos in test_positions])
                y_test = np.array([labels_dict[pos] for pos in test_positions])

                field_split_success = True
            else:
                print("Warning: Field-based splitting yielded empty training or test set.")
                field_split_success = False

        except Exception as e:
            print(f"Error during field-based splitting: {e}")
            field_split_success = False

    # ---- 4) 如果字段拆分不成功，就随机拆分 ----
    if not field_split_success:
        print("Using random splitting as fallback...")
        positions = list(embeddings_dict.keys())
        np.random.shuffle(positions)
        if len(positions) == 0:
            raise ValueError("No valid positions found in dataset")

        embeddings = np.array([embeddings_dict[pos] for pos in positions])
        labels_arr = np.array([labels_dict[pos] for pos in positions])

        N = len(embeddings)
        split = int(training_ratio * N)

        X_train = embeddings[:split]
        y_train = labels_arr[:split]
        X_test = embeddings[split:]
        y_test = labels_arr[split:]

        print(f"Random split: Train={len(X_train)}, Test={len(X_test)}")

    # ---- 5) 检查数据合法性 ----
    if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(y_test) == 0:
        print("Error: Empty training or test set after splitting")
        return 0.0, 0.0, np.zeros((1, 1)), "Error: Empty dataset"

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("Warning: Not enough classes in training or test set")
        print(f"Classes in train: {np.unique(y_train)}")
        print(f"Classes in test: {np.unique(y_test)}")
        return 0.0, 0.0, np.zeros((1, 1)), "Error: Not enough classes"

    # ---- 6) 训练线性分类器 (RF 或 LR) ----
    if classifier_type.lower() == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=10000, n_jobs=-1)

    print(f"Training {classifier_type} classifier with {len(X_train)} samples...")
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, digits=4)

    return acc, f1, cm, cr



