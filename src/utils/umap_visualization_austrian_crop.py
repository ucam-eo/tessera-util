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
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP  # <--- 修改点：直接从 umap 库导入 UMAP 类
from multiprocessing import cpu_count
import warnings
from scipy import ndimage
warnings.filterwarnings('ignore')

# 设置matplotlib参数以符合Nature标准
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['pdf.fonttype'] = 42  # 确保PDF中的字体可编辑
plt.rcParams['ps.fonttype'] = 42

def resize_feature_data(data, target_shape):
    """
    将特征数据的空间维度resize到目标形状
    
    Args:
        data: shape (H, W, C) 的特征数据
        target_shape: (target_H, target_W) 目标空间维度
    
    Returns:
        resized_data: shape (target_H, target_W, C) 的调整后数据
    """
    H, W, C = data.shape
    target_H, target_W = target_shape
    
    if (H, W) == (target_H, target_W):
        return data
    
    print(f"  Resizing feature data from ({H}, {W}) to ({target_H}, {target_W})...")
    
    # 首先检查输入数据是否有NaN
    if np.any(np.isnan(data)):
        print("  Warning: Input data contains NaN values, replacing with 0...")
        data = np.nan_to_num(data, nan=0.0)
    
    # 计算缩放因子
    zoom_factors = [target_H / H, target_W / W, 1.0]  # C维度不变
    
    # 使用scipy的zoom进行插值
    # order=1 表示双线性插值，mode='nearest'避免边界产生NaN
    resized_data = ndimage.zoom(data, zoom_factors, order=1, mode='nearest')
    
    # 检查并处理可能的NaN值
    if np.any(np.isnan(resized_data)):
        print("  Warning: Resize produced NaN values, replacing with 0...")
        resized_data = np.nan_to_num(resized_data, nan=0.0)
    
    # 确保输出形状正确
    assert resized_data.shape == (target_H, target_W, C), \
        f"Resize failed: expected {(target_H, target_W, C)}, got {resized_data.shape}"
    
    return resized_data

def load_and_preprocess_data(file_path, target_shape=None):
    """
    加载并预处理影像特征数据
    
    Args:
        file_path: 特征数据文件路径
        target_shape: 可选的目标空间形状 (H, W)
    """
    print("Loading feature data...")
    t0 = time.time()
    data = np.load(file_path)
    print(f"  Feature data shape: {data.shape} loaded in {time.time()-t0:.2f}s")
    H, W, C = data.shape
    
    # 检查原始数据是否有NaN
    nan_count = np.sum(np.isnan(data))
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaN values in original data, replacing with 0...")
        data = np.nan_to_num(data, nan=0.0)
    
    # 如果需要，调整空间维度
    if target_shape is not None and (H, W) != target_shape:
        data = resize_feature_data(data, target_shape)
        H, W = target_shape
        print(f"  Resized to shape: {data.shape}")
    
    # 转换为 float32 以节省内存
    if data.dtype != np.float32:
        print("  Converting to float32 for memory efficiency...")
        data = data.astype(np.float32)
    
    data_flat = data.reshape(-1, C)
    
    # 最终NaN检查
    nan_count_flat = np.sum(np.isnan(data_flat))
    if nan_count_flat > 0:
        print(f"  Warning: Found {nan_count_flat} NaN values after flattening, replacing with 0...")
        data_flat = np.nan_to_num(data_flat, nan=0.0)
    
    print(f"  Reshaped to {data_flat.shape} ({data_flat.shape[0]:,} samples)")
    return data_flat, (H, W, C)

def load_and_preprocess_labels(label_path):
    """加载并扁平化标签数据"""
    print("Loading label data...")
    t0 = time.time()
    labels = np.load(label_path)
    print(f"  Label data shape: {labels.shape} loaded in {time.time()-t0:.2f}s")
    flat = labels.reshape(-1)
    return flat, labels.shape[:2]  # 返回扁平化数据和空间形状

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
    
    # 在PCA之前再次检查NaN
    nan_count = np.sum(np.isnan(data))
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaN values before PCA, replacing with 0...")
        data = np.nan_to_num(data, nan=0.0)
    
    # 检查是否有无穷值
    inf_count = np.sum(np.isinf(data))
    if inf_count > 0:
        print(f"  Warning: Found {inf_count} infinite values before PCA, replacing with 0...")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data)
    ev = pca.explained_variance_ratio_.sum()
    print(f"  PCA done in {time.time()-t0:.2f}s; shape={data_pca.shape}; explained variance={ev:.4f}")
    return data_pca, pca

def perform_umap(data, n_neighbors=15, min_dist=0.1, n_jobs=None):
    """执行 UMAP 降维"""
    if n_jobs is None:
        n_jobs = min(cpu_count(), 32)
    
    print(f"Running UMAP (n_jobs={n_jobs})...")
    t0 = time.time()
    
    n = data.shape[0]
    # 根据数据量调整参数
    if n > 100000:
        n_neighbors = 30
        min_dist = 0.05
    elif n > 50000:
        n_neighbors = 20
        min_dist = 0.1
    else:
        n_neighbors = 15
        min_dist = 0.1
    
    print(f"  params → n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    # <--- 修改点：直接使用 UMAP(...) 而不是 umap.UMAP(...)
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        n_jobs=n_jobs,
        random_state=42,
        verbose=True,
        low_memory=False  # 使用更多内存以加快速度
    )
    
    Y = reducer.fit_transform(data)
    print(f"  UMAP done in {time.time()-t0:.2f}s")
    return Y

def compute_clustering_metrics(umap_Y, labels, sample_size=10000):
    """
    计算聚类评估指标
    
    Args:
        umap_Y: UMAP降维后的2D数据
        labels: 类别标签
        sample_size: 如果数据太大，采样计算以加快速度
    
    Returns:
        dict: 包含Silhouette Score和Davies-Bouldin Index
    """
    print("Computing clustering metrics...")
    t0 = time.time()
    
    n_samples = umap_Y.shape[0]
    
    # 如果数据太大，进行采样
    if n_samples > sample_size:
        print(f"  Sampling {sample_size} points for metric computation...")
        indices = np.random.choice(n_samples, size=sample_size, replace=False)
        Y_sample = umap_Y[indices]
        labels_sample = labels[indices]
    else:
        Y_sample = umap_Y
        labels_sample = labels
    
    # 计算Silhouette Score（轮廓系数）
    try:
        silhouette = silhouette_score(Y_sample, labels_sample, metric='euclidean')
        print(f"  Silhouette Score: {silhouette:.4f}")
    except Exception as e:
        print(f"  Warning: Failed to compute Silhouette Score: {e}")
        silhouette = np.nan
    
    # 计算Davies-Bouldin Index（戴维斯-布尔丁指数）
    try:
        db_index = davies_bouldin_score(Y_sample, labels_sample)
        print(f"  Davies-Bouldin Index: {db_index:.4f}")
    except Exception as e:
        print(f"  Warning: Failed to compute Davies-Bouldin Index: {e}")
        db_index = np.nan
    
    print(f"  Metrics computed in {time.time()-t0:.2f}s")
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': db_index
    }

def create_nature_visualization(umap_Y, labels, output_path, sampling_info=None, metrics=None):
    """创建符合Nature期刊标准的UMAP可视化图"""
    print("Creating UMAP plot (Nature style)...")
    
    # 创建图形，尺寸为Nature单栏宽度（89mm ≈ 3.5 inches）
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    
    n_pts = umap_Y.shape[0]
    # 根据点数调整点的大小
    if n_pts > 100000:
        sz, alpha = 0.1, 0.3
    elif n_pts > 50000:
        sz, alpha = 0.2, 0.4
    elif n_pts > 10000:
        sz, alpha = 0.4, 0.5
    else:
        sz, alpha = 1, 0.6
    
    # 使用专业的配色方案
    # 为17个类别创建自定义颜色映射
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    
    # 使用tab20配色，但调整为17个类别
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
    # 移除一些相似的颜色，保留17个
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18]
    custom_colors = colors[selected_indices]
    custom_cmap = ListedColormap(custom_colors)
    
    # 绘制散点图
    scatter = ax.scatter(
        umap_Y[:, 0], umap_Y[:, 1],
        c=labels - 1,  # 调整为0-16的索引
        cmap=custom_cmap,
        s=sz,
        alpha=alpha,
        rasterized=True,  # 栅格化以减小文件大小
        edgecolors='none'
    )
    
    # 设置轴标签
    ax.set_xlabel('UMAP 1', fontsize=8)
    ax.set_ylabel('UMAP 2', fontsize=8)
    
    # 设置1:1的纵横比
    # ax.set_aspect('equal', adjustable='box')
    
    # 移除顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # 添加聚类指标到右上角
    if metrics is not None:
        # 获取坐标轴范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # 计算文本位置（右上角）
        text_x = xlim[1] - 0.05 * (xlim[1] - xlim[0])
        text_y = ylim[1] - 0.05 * (ylim[1] - ylim[0])
        
        # 准备文本内容
        metric_text = []
        if not np.isnan(metrics['silhouette']):
            metric_text.append(f"Silhouette: {metrics['silhouette']:.3f}")
        if not np.isnan(metrics['davies_bouldin']):
            metric_text.append(f"Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
        
        if metric_text:
            # 添加半透明背景框
            bbox_props = dict(boxstyle="round,pad=0.3", 
                            facecolor='white', 
                            edgecolor='gray',
                            alpha=0.8,
                            linewidth=0.5)
            
            ax.text(text_x, text_y, '\n'.join(metric_text),
                   transform=ax.transData,
                   fontsize=7,
                   ha='right',
                   va='top',
                   bbox=bbox_props,
                   zorder=100)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(17))
    cbar.set_ticklabels(np.arange(1, 18))
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Class', fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存高质量图像
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                format='png', facecolor='white', edgecolor='none')
    
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
            
            # 添加聚类指标信息
            if metrics is not None:
                f.write("\n" + "="*50 + "\n")
                f.write("Clustering Metrics:\n")
                f.write("="*50 + "\n")
                f.write(f"Silhouette Score: {metrics['silhouette']:.4f}\n")
                f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}\n")
                f.write("\nNote: Higher Silhouette Score (range: -1 to 1) indicates better clustering.\n")
                f.write("Lower Davies-Bouldin Index (range: 0 to ∞) indicates better clustering.\n")
        
        print(f"  Sampling info saved to {info_path}")

def main():
    # —— 配置区域 —— 
    # input_file  = '/scratch/zf281/btfm_representation/austrian_crop/austrian_crop_EFM_Embeddings_2022.npy'
    input_file  = '/scratch/zf281/btfm_representation/austrian_crop/mpc_pipeline_fsdp_20250604_100313_downsample_100.npy'
    # input_file  = '/scratch/zf281/btfm_representation/austrian_crop/austria_Presto_embeddings_100m.npy'
    
    # label_file  = '/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes.npy'
    label_file  = '/scratch/zf281/austrian_crop_v1.0_pipeline/fieldtype_17classes_downsample_100.npy'
    output_dir  = '/maps/zf281/btfm4rs/src/utils/'
    
    # 采样参数配置
    MAX_SAMPLES_PER_CLASS = 1000000000   # 每个类别最多采样的样本数
    MIN_SAMPLES_PER_CLASS = 1000   # 每个类别最少保留的样本数
    FORCE_PCA_THRESHOLD = 1500000000   # 超过这个样本数强制使用PCA
    # ——————————

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # 1. 先加载标签数据以获取目标空间形状
    labels_flat, label_shape = load_and_preprocess_labels(label_file)
    
    # 2. 加载特征数据，如果需要则resize到标签的空间形状
    data_flat, (H, W, C) = load_and_preprocess_data(input_file, target_shape=label_shape)

    # 确保空间大小一致
    assert labels_flat.shape[0] == data_flat.shape[0], \
        f"Feature ({data_flat.shape[0]}) 和标签 ({labels_flat.shape[0]}) 在 H*W 上必须一致"

    # 3. 排除背景（标签0）
    mask = labels_flat > 0
    data_sel   = data_flat[mask]
    labels_sel = labels_flat[mask]
    print(f"Filtered out background → {data_sel.shape[0]:,} samples remain")
    
    # 检查过滤后的数据是否有NaN
    nan_count = np.sum(np.isnan(data_sel))
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaN values after filtering, replacing with 0...")
        data_sel = np.nan_to_num(data_sel, nan=0.0)

    # 4. 智能分层采样（如果数据量太大）
    sampling_info = None
    if data_sel.shape[0] > 100000:  # 如果超过10万个样本就进行采样
        data_sampled, labels_sampled, sampling_info = stratified_sampling(
            data_sel, labels_sel, 
            max_samples_per_class=MAX_SAMPLES_PER_CLASS,
            min_samples_per_class=MIN_SAMPLES_PER_CLASS
        )
        data_for_umap = data_sampled
        labels_for_umap = labels_sampled
        print(f"Using sampled data: {data_for_umap.shape[0]:,} samples")
    else:
        data_for_umap = data_sel
        labels_for_umap = labels_sel
        print(f"Using all filtered data: {data_for_umap.shape[0]:,} samples")

    # 5. 对特别大的数据先 PCA（阈值提高到15万）
    used_pca = False
    if data_for_umap.shape[0] > FORCE_PCA_THRESHOLD or data_for_umap.shape[1] > 100:
        # 对于高维数据，建议先用PCA降维
        data_for_umap, _ = apply_pca_preprocessing(data_for_umap, n_components=50)
        used_pca = True

    # 6. 执行 UMAP
    umap_Y = perform_umap(data_for_umap)

    # 7. 计算聚类指标
    metrics = compute_clustering_metrics(umap_Y, labels_for_umap)

    # 8. 可视化并保存（Nature风格）
    out_png = os.path.join(output_dir, 'umap_by_label_nature.png')
    create_nature_visualization(
        umap_Y,
        labels_for_umap,
        out_png,
        sampling_info=sampling_info,
        metrics=metrics
    )

    print(f"全部完成，总耗时 {time.time()-t_start:.2f}s")
    
    # 输出最终统计信息
    print("\n" + "="*60)
    print("FINAL SUMMARY:")
    print("="*60)
    print(f"Original data: {H}×{W}×{C} = {labels_flat.shape[0]:,} total samples")
    print(f"After filtering background: {data_sel.shape[0]:,} samples")
    if sampling_info:
        print(f"After stratified sampling: {data_for_umap.shape[0]:,} samples")
        total_original = sum(info['original'] for info in sampling_info.values())
        sampling_ratio = data_for_umap.shape[0] / total_original * 100
        print(f"Sampling ratio: {sampling_ratio:.1f}%")
    if used_pca:
        print(f"PCA applied: {data_for_umap.shape[1]} components")
    print(f"Final UMAP input: {data_for_umap.shape[0]:,} samples × {data_for_umap.shape[1]} features")
    
    # 打印聚类指标
    print("\nClustering Performance:")
    print(f"  Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
    
    print(f"\nOutput saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()