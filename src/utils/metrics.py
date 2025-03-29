# src/utils/metrics.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from torch.nn.parallel import DistributedDataParallel as DDP
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

def linear_probe_evaluate(model, val_loader, device='cuda'):
    """
    使用验证集计算模型嵌入后训练出的 logistic regression 分类器的表现，
    返回 accuracy, weighted F1 score 以及混淆矩阵。
    """
    
    # Create a non-compiled copy of the model for evaluation
    if isinstance(model, DDP):
        # Get the underlying model from DDP wrapper
        eval_model = model.module
    else:
        eval_model = model
        
    # Make sure we're using the original model, not the compiled one
    # We can access the original model through _orig_mod if it was compiled
    if hasattr(eval_model, '_orig_mod'):
        eval_model = eval_model._orig_mod
    
    eval_model.eval()
    embeddings_list = []
    labels_list = []
    # max_samples = 20000
    with torch.no_grad():
        for s2_sample, s1_sample, label in val_loader:
            s2_sample = s2_sample.to(device)
            s1_sample = s1_sample.to(device)
            out = eval_model(s2_sample, s1_sample)
            if isinstance(out, tuple):
                out = out[1]
            emb = out.cpu().numpy()
            embeddings_list.append(emb)
            # 注意：label 可能需要 .cpu() 后再转换成 numpy 数组
            labels_list.extend(label.cpu().numpy())
            # if len(labels_list) >= max_samples:
            #     break
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels_arr = np.array(labels_list, dtype=np.int64)
    N = embeddings.shape[0]
    if N < 2:
        return 1.0, 1.0, None
    np.random.seed(42)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.3 * N)
    train_idx = idx[:split]
    test_idx = idx[split:]
    X_train = embeddings[train_idx]
    y_train = labels_arr[train_idx]
    X_test = embeddings[test_idx]
    y_test = labels_arr[test_idx]
    clf = LogisticRegression(max_iter=10000, n_jobs=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)
    # 生成report
    cr = classification_report(y_test, pred, digits=4)
    return acc, f1, cm, cr

def rf_probe_evaluate(model, val_loader, device='cuda'):
    """
    使用验证集计算模型嵌入后使用 RandomForestClassifier 的分类性能，
    返回 accuracy, weighted F1 score, 混淆矩阵，以及分类报告(字符串)。
    """
    # 1) 获取原始模型 (若使用DDP或torch.compile)
    if isinstance(model, DDP):
        eval_model = model.module
    else:
        eval_model = model
    if hasattr(eval_model, '_orig_mod'):
        eval_model = eval_model._orig_mod

    eval_model.eval()

    # 2) 前向推理，收集 embeddings
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for s2_sample, s1_sample, label in val_loader:
            s2_sample = s2_sample.to(device)
            s1_sample = s1_sample.to(device)
            out = eval_model(s2_sample, s1_sample)
            # 若 model(...) 返回 (z, representation) 二元组，就取第2个即可
            if isinstance(out, tuple):
                out = out[1]
            emb = out.cpu().numpy()
            embeddings_list.append(emb)
            labels_list.extend(label.cpu().numpy())

    # 拼接全部 embeddings / labels
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels_arr = np.array(labels_list, dtype=np.int64)

    # 若数据过少，则直接返回
    N = embeddings.shape[0]
    if N < 2:
        return 1.0, 1.0, None, "Not enough data"

    # 3) 同样的随机种子、同样的 30%:70% 划分
    np.random.seed(42)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.3 * N)
    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = embeddings[train_idx]
    y_train = labels_arr[train_idx]
    X_test = embeddings[test_idx]
    y_test = labels_arr[test_idx]

    # 4) 随机森林分类器
    rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    rf_clf.fit(X_train, y_train)
    pred = rf_clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)
    # digits=4 可以让分类报告保留 4 位小数
    cr = classification_report(y_test, pred, digits=4)

    return acc, f1, cm, cr