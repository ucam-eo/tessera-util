# src/utils/metrics.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
from sklearn.decomposition import PCA

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

def linear_probe_evaluate(model, val_loader, device='cuda'):
    """
    使用验证集计算模型嵌入后训练出的 logistic regression 分类器的表现，
    返回 accuracy, weighted F1 score 以及混淆矩阵。
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    # max_samples = 20000
    with torch.no_grad():
        for s2_sample, s1_sample, label in val_loader:
            s2_sample = s2_sample.to(device)
            s1_sample = s1_sample.to(device)
            out = model(s2_sample, s1_sample)
            if isinstance(out, tuple):
                out = out[1]
            emb = out.cpu().numpy() # shape: (batch_size, num_features)
            embeddings_list.append(emb)
            # 注意：label 可能需要 .cpu() 后再转换成 numpy 数组
            labels_list.extend(label.cpu().numpy())
            # if len(labels_list) >= max_samples:
            #     break
    embeddings = np.concatenate(embeddings_list, axis=0) # shape: (num_samples, num_features)
    labels_arr = np.array(labels_list, dtype=np.int64)
    N, dim = embeddings.shape
    if N < 2:
        return 1.0, 1.0, None
    
    # ========== 使用 PCA 将特征维降 ==========
    # 如果维度大于256，则用 PCA 将整个数据集的 embeddings 降维到128
    # if dim > 256:
    # pca = PCA(n_components=64)
    # embeddings = pca.fit_transform(embeddings)  # 在整个数据集上fit+transform
    
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
    
    
    clf = LogisticRegression(max_iter=10000, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, digits=4)
    return acc, f1, cm, cr
