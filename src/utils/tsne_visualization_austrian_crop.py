#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# —— 一定要在导入 numpy 之前设置 —— 
# 限制 OpenBLAS/OMP/MKL 的最大线程数，避免过多线程导致内存区分配失败
os.environ["OPENBLAS_NUM_THREADS"] = "32"  # 增加到32以利用更多核心
os.environ["OMP_NUM_THREADS"]       = "32"
os.environ["MKL_NUM_THREADS"]       = "32"

import time
import numpy as np
import matplotlib
# 不使用 GUI 后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

# 尝试导入更快的 t-SNE 实现
try:
    import openTSNE
    USE_OPENDATA_TSNE = True
    print("Using openTSNE for better performance")
except ImportError:
    USE_OPENDATA_TSNE = False
    print("openTSNE not available, using sklearn TSNE")

def load_and_preprocess_data(file_path):
    """加载并预处理影像特征数据"""
    print("Loading feature data...")
    t0 = time.time()
    data = np.load(file_path)
    print(f"  Feature data shape: {data.shape}  loaded in {time.time()-t0:.2f}s")
    H, W, C = data.shape
    
    # 转换为 float32 以节省内存
    if data.dtype != np.float32:
        print("  Converting to float32 for memory efficiency...")
        data = data.astype(np.float32)
    
    data_flat = data.reshape(-1, C)
    print(f"  Reshaped to {data_flat.shape} ({data_flat.shape[0]:,} samples)")
    return data_flat, (H, W, C)

def load_and_preprocess_labels(label_path):
    """加载并扁平化标签数据"""
    print("Loading label data...")
    t0 = time.time()
    labels = np.load(label_path)
    print(f"  Label data shape: {labels.shape}  loaded in {time.time()-t0:.2f}s")
    flat = labels.reshape(-1)
    return flat

def stratified_sampling(data, labels, max_samples_per_class=8000, min_samples_per_class=1000):
    """
    对每个类别进行分层采样，确保每个类别都有足够的代表性
    """
    print("Performing stratified sampling...")
    t0 = time.time()
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]  # 排除背景
    
    sampled_indices = []
    sampling_info = {}
    
    for label in unique_labels:
        label_mask = labels == label
        label_indices = np.where(label_mask)[0]
        n_available = len(label_indices)
        
        # 动态决定采样数量
        if n_available <= min_samples_per_class:
            n_samples = n_available  # 如果样本太少，全部保留
        elif n_available <= max_samples_per_class:
            n_samples = n_available  # 如果样本适中，全部保留
        else:
            n_samples = max_samples_per_class  # 如果样本太多，采样到上限
        
        # 随机采样
        if n_samples < n_available:
            sampled_label_indices = np.random.choice(label_indices, size=n_samples, replace=False)
        else:
            sampled_label_indices = label_indices
        
        sampled_indices.extend(sampled_label_indices)
        sampling_info[int(label)] = {'original': n_available, 'sampled': n_samples}
        
        print(f"  Class {label:2d}: {n_available:8,} → {n_samples:6,} samples")
    
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)  # 打乱顺序
    
    sampled_data = data[sampled_indices]
    sampled_labels = labels[sampled_indices]
    
    total_original = sum(info['original'] for info in sampling_info.values())
    total_sampled = len(sampled_indices)
    
    print(f"  Stratified sampling completed in {time.time()-t0:.2f}s")  
    print(f"  Total: {total_original:,} → {total_sampled:,} samples ({total_sampled/total_original*100:.1f}%)")
    
    return sampled_data, sampled_labels, sampling_info

def apply_pca_preprocessing(data, n_components=50):
    """使用 PCA 将维度降至 n_components"""
    print(f"Applying PCA (n_components={n_components})...")
    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data)
    ev = pca.explained_variance_ratio_.sum()
    print(f"  PCA done in {time.time()-t0:.2f}s; shape={data_pca.shape}; explained variance={ev:.4f}")
    return data_pca, pca

def perform_tsne_openTSNE(data, n_jobs=None):
    """使用 openTSNE 执行 t-SNE（更快的实现）"""
    if n_jobs is None:
        n_jobs = min(cpu_count(), 32)  # 使用更多核心
    
    print(f"Running openTSNE (n_jobs={n_jobs})...")
    t0 = time.time()
    
    n = data.shape[0]
    if n > 100000:
        perp, early_iter, final_iter = 50, 250, 500
    elif n > 50000:
        perp, early_iter, final_iter = 40, 250, 750  
    elif n > 10000:
        perp, early_iter, final_iter = 30, 250, 750
    else:
        perp, early_iter, final_iter = 30, 250, 1000
    
    print(f"  params → perplexity={perp}, early_exaggeration_iter={early_iter}, final_iter={final_iter}")
    
    # 使用 openTSNE，它使用不同的参数结构
    # 首先进行初始化
    init = openTSNE.initialization.pca(data, n_components=2, random_state=42)
    
    # 创建亲和力矩阵（近邻）
    affinities = openTSNE.affinity.PerplexityBasedNN(
        data,
        perplexity=perp,
        n_jobs=n_jobs,
        random_state=42,
        verbose=True
    )
    
    # 执行早期夸大阶段
    embedding_early = openTSNE.TSNEEmbedding(
        init,
        affinities,
        n_jobs=n_jobs,
        verbose=True,
        random_state=42
    )
    
    # 早期夸大优化
    embedding_early = embedding_early.optimize(
        n_iter=early_iter,
        exaggeration=12.0,
        momentum=0.5,
        learning_rate="auto"
    )
    
    # 最终优化阶段
    embedding_final = embedding_early.optimize(
        n_iter=final_iter,
        exaggeration=1.0,
        momentum=0.8,
        learning_rate="auto"
    )
    
    Y = np.array(embedding_final)
    print(f"  openTSNE done in {time.time()-t0:.2f}s")
    return Y

def perform_tsne(data, n_jobs=None):
    """执行 t-SNE，并限制并行数"""
    # 读取 BLAS 线程上限
    max_blas = int(os.environ.get("OPENBLAS_NUM_THREADS", "1"))
    if n_jobs is None:
        n_jobs = min(cpu_count(), max_blas)
    else:
        n_jobs = min(n_jobs, cpu_count(), max_blas)
    print(f"Running t-SNE (n_jobs={n_jobs})...")
    t0 = time.time()

    n = data.shape[0]
    if n > 100000:
        perp, iters = 50, 750  # 减少迭代次数以加快速度
    elif n > 50000:
        perp, iters = 40, 1000
    elif n > 10000:
        perp, iters = 30, 1000
    else:
        perp, iters = 30, 1000

    print(f"  params → perplexity={perp}, max_iter={iters}, learning_rate=200")
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=200,
        init='pca',  # 使用 PCA 初始化而不是随机初始化
        random_state=42,
        n_jobs=n_jobs,
        verbose=2,
        metric='euclidean',
        max_iter=iters,
        method='barnes_hut',  # 使用 Barnes-Hut 近似以加快速度
        angle=0.5  # Barnes-Hut 的角度参数
    )
    Y = tsne.fit_transform(data)
    print(f"  t-SNE done in {time.time()-t0:.2f}s; KL div={tsne.kl_divergence_:.6f}")
    return Y

def create_visualization(tsne_Y, labels, output_path, title="t-SNE Visualization", sampling_info=None):
    """按标签着色并保存 t-SNE 可视化结果"""
    print("Creating t-SNE plot...")
    plt.figure(figsize=(16,12))

    n_pts = tsne_Y.shape[0]
    if n_pts > 100000:
        sz, alpha = 0.1, 0.3
    elif n_pts > 50000:
        sz, alpha = 0.3, 0.4
    elif n_pts > 10000:
        sz, alpha = 0.5, 0.5
    else:
        sz, alpha = 1.0, 0.6

    # 使用 tab20 作为离散的 17 类 colormap
    norm = plt.Normalize(vmin=1, vmax=17)
    scatter = plt.scatter(
        tsne_Y[:,0], tsne_Y[:,1],
        c=labels,
        cmap='tab20',
        norm=norm,
        s=sz,
        alpha=alpha,
        rasterized=True
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)

    # Colorbar 标注 1–17
    cbar = plt.colorbar(scatter, ticks=list(range(1,18)))
    cbar.set_label('Class Label', fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {output_path}")
    
    # 如果有采样信息，保存到文件
    if sampling_info:
        info_path = output_path.replace('.png', '_sampling_info.txt')
        with open(info_path, 'w') as f:
            f.write("Stratified Sampling Information:\n")
            f.write("="*50 + "\n")
            for label, info in sampling_info.items():
                f.write(f"Class {label:2d}: {info['original']:8,} → {info['sampled']:6,} samples\n")
            total_original = sum(info['original'] for info in sampling_info.values())
            total_sampled = sum(info['sampled'] for info in sampling_info.values())
            f.write("="*50 + "\n")
            f.write(f"Total: {total_original:,} → {total_sampled:,} samples ({total_sampled/total_original*100:.1f}%)\n")
        print(f"  Sampling info saved to {info_path}")

def main():
    # —— 配置区域 —— 
    # input_file  = '/scratch/zf281/btfm_representation/austrian_crop/whole_year_representations_fsdp_20250427_084307_repreat_1.npy'
    # input_file  = '/scratch/zf281/btfm_representation/austrian_crop/austrian_crop_EFM_Embeddings_2022_downsample_100.npy'
    input_file  = '/scratch/zf281/btfm_representation/austrian_crop/austrian_crop_EFM_Embeddings_2022.npy'
    # input_file  = '/scratch/zf281/btfm_representation/austrian_crop/whole_year_representations_fsdp_20250427_084307_repreat_1_downsample_100.npy'
    # input_file  = '/scratch/zf281/btfm_representation/austrian_crop/representations_fsdp_20250427_084307_downsample_100.npy'
    # label_file  = '/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes_downsample_100.npy'
    label_file  = '/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes.npy'
    output_dir  = '/maps/zf281/btfm4rs/src/utils/'
    
    # 采样参数配置
    MAX_SAMPLES_PER_CLASS = 1000000000   # 每个类别最多采样的样本数
    MIN_SAMPLES_PER_CLASS = 1000   # 每个类别最少保留的样本数
    FORCE_PCA_THRESHOLD = 1500000000   # 超过这个样本数强制使用PCA
    # ——————————

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # 1. 加载数据与标签
    data_flat, (H, W, C) = load_and_preprocess_data(input_file)
    labels_flat = load_and_preprocess_labels(label_file)

    # 确保空间大小一致
    assert labels_flat.shape[0] == data_flat.shape[0], \
        "Feature 和标签在 H*W 上必须一致"

    # 2. 排除背景（标签0）
    mask = labels_flat > 0
    data_sel   = data_flat[mask]
    labels_sel = labels_flat[mask]
    print(f"Filtered out background → {data_sel.shape[0]:,} samples remain")

    # 3. 智能分层采样（如果数据量太大）
    sampling_info = None
    if data_sel.shape[0] > 100000:  # 如果超过10万个样本就进行采样
        data_sampled, labels_sampled, sampling_info = stratified_sampling(
            data_sel, labels_sel, 
            max_samples_per_class=MAX_SAMPLES_PER_CLASS,
            min_samples_per_class=MIN_SAMPLES_PER_CLASS
        )
        data_for_tsne = data_sampled
        labels_for_tsne = labels_sampled
        print(f"Using sampled data: {data_for_tsne.shape[0]:,} samples")
    else:
        data_for_tsne = data_sel
        labels_for_tsne = labels_sel
        print(f"Using all filtered data: {data_for_tsne.shape[0]:,} samples")

    # 4. 对特别大的数据先 PCA（阈值提高到15万）
    used_pca = False
    if data_for_tsne.shape[0] > FORCE_PCA_THRESHOLD:
        data_for_tsne, _ = apply_pca_preprocessing(data_for_tsne, n_components=50)
        used_pca = True

    # 5. 执行 t-SNE
    if USE_OPENDATA_TSNE:
        tsne_Y = perform_tsne_openTSNE(data_for_tsne)
    else:
        tsne_Y = perform_tsne(data_for_tsne)

    # 6. 可视化并保存
    out_png = os.path.join(output_dir, 'tsne_by_label_optimized.png')
    title_suffix = f', sampled' if sampling_info else ', full'
    create_visualization(
        tsne_Y,
        labels_for_tsne,
        out_png,
        title=f't-SNE (classes 1–17), samples={tsne_Y.shape[0]:,}, orig={H}×{W}×{C}{title_suffix}',
        sampling_info=sampling_info
    )

    # 7. 保存结果
    np.save(os.path.join(output_dir, 'tsne_result_optimized.npy'), tsne_Y)
    meta = {
        'orig_shape':        (H, W, C),
        'n_total':           labels_flat.shape[0],
        'n_filtered':        data_sel.shape[0],
        'n_final':           data_for_tsne.shape[0],
        'used_sampling':     sampling_info is not None,
        'used_pca':          used_pca,
        'tsne_shape':        tsne_Y.shape,
        'sampling_info':     sampling_info
    }
    np.save(os.path.join(output_dir, 'tsne_metadata_optimized.npy'), meta)

    print(f"全部完成，总耗时 {time.time()-t_start:.2f}s")
    
    # 输出最终统计信息
    print("\n" + "="*60)
    print("FINAL SUMMARY:")
    print("="*60)
    print(f"Original data: {H}×{W}×{C} = {labels_flat.shape[0]:,} total samples")
    print(f"After filtering background: {data_sel.shape[0]:,} samples")
    if sampling_info:
        print(f"After stratified sampling: {data_for_tsne.shape[0]:,} samples")
        total_original = sum(info['original'] for info in sampling_info.values())
        sampling_ratio = data_for_tsne.shape[0] / total_original * 100
        print(f"Sampling ratio: {sampling_ratio:.1f}%")
    if used_pca:
        print(f"PCA applied: {data_for_tsne.shape[1]} components")
    print(f"Final t-SNE input: {data_for_tsne.shape[0]:,} samples × {data_for_tsne.shape[1]} features")
    print(f"Output saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()