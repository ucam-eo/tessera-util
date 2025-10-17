#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NVIDIA GPU优化版本 - 单GPU + 多核CPU混合加速 - 无PCA版本

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# 标准CPU库
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP
from scipy import ndimage
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil

# 设置线程数以充分利用200+核心
CPU_COUNT = os.cpu_count()
OPTIMAL_THREADS = min(CPU_COUNT, 100)  # 使用最多100个线程
os.environ["OPENBLAS_NUM_THREADS"] = str(OPTIMAL_THREADS)
os.environ["OMP_NUM_THREADS"] = str(OPTIMAL_THREADS)
os.environ["MKL_NUM_THREADS"] = str(OPTIMAL_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(OPTIMAL_THREADS)

# 检查GPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # 指定使用第一个GPU
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 设置GPU内存分配策略
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的GPU内存
else:
    device = torch.device('cpu')
    print("GPU not available, using CPU")

# matplotlib设置
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
        return allocated, reserved, free
    return 0, 0, 0

def resize_feature_data_torch(data, target_shape):
    """使用PyTorch在GPU上进行resize，带内存管理"""
    H, W, C = data.shape
    target_H, target_W = target_shape
    
    if (H, W) == (target_H, target_W):
        return data
    
    print(f"  Resizing feature data from ({H}, {W}) to ({target_H}, {target_W})...")
    
    # 检查GPU内存
    allocated, reserved, free = check_gpu_memory()
    print(f"  GPU Memory - Allocated: {allocated:.1f}GB, Free: {free:.1f}GB")
    
    # 根据可用内存决定批处理大小
    estimated_memory_needed = H * W * C * 4 * 2 / 1e9  # float32, 输入输出各一份
    batch_channels = int(min(C, max(1, free * 0.8 * 1e9 / (H * W * 4 * 2))))
    
    if batch_channels < C:
        print(f"  Processing in batches of {batch_channels} channels due to memory constraints")
        resized_data = np.zeros((target_H, target_W, C), dtype=np.float32)
        
        for i in range(0, C, batch_channels):
            end_idx = min(i + batch_channels, C)
            batch = data[:, :, i:end_idx]
            
            # 转换为PyTorch张量并移至GPU
            batch_tensor = torch.from_numpy(batch).float().to(device)
            batch_tensor = batch_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # 使用双线性插值
            resized_batch = F.interpolate(
                batch_tensor, 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 转回CPU
            resized_data[:, :, i:end_idx] = resized_batch.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # 清理GPU内存
            del batch_tensor, resized_batch
            torch.cuda.empty_cache()
    else:
        # 一次性处理所有通道
        data_tensor = torch.from_numpy(data).float().to(device)
        data_tensor = data_tensor.permute(2, 0, 1).unsqueeze(0)
        
        resized_tensor = F.interpolate(
            data_tensor, 
            size=(target_H, target_W), 
            mode='bilinear', 
            align_corners=False
        )
        
        resized_data = resized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        del data_tensor, resized_tensor
        torch.cuda.empty_cache()
    
    return resized_data

def parallel_umap_batch(args):
    """用于并行处理UMAP批次的函数"""
    batch_data, batch_idx, n_neighbors, min_dist = args
    
    print(f"      Processing UMAP batch {batch_idx} with {batch_data.shape[0]} samples...")
    
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, batch_data.shape[0] - 1),
        min_dist=min_dist,
        n_epochs=500,
        metric='euclidean',
        n_jobs=-1,  # 使用所有可用核心
        random_state=42 + batch_idx,  # 不同批次使用不同种子
        verbose=False,
        low_memory=False
    )
    
    embedding = reducer.fit_transform(batch_data)
    return embedding

def optimized_batch_umap(data, labels, batch_size=50000, n_neighbors=30, min_dist=0.1):
    """优化的批处理UMAP，利用多进程并行"""
    print(f"Running optimized batch UMAP (batch_size={batch_size})...")
    print(f"  Input data shape: {data.shape}")
    t0 = time.time()
    
    n_samples = data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    if n_batches == 1:
        # 单批次，直接处理
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=500,
            metric='euclidean',
            n_jobs=OPTIMAL_THREADS,
            random_state=42,
            verbose=True
        )
        full_embedding = reducer.fit_transform(data)
    else:
        # 多批次并行处理
        batch_args = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_data = data[start_idx:end_idx]
            batch_args.append((batch_data, i, n_neighbors, min_dist))
        
        # 使用进程池并行处理批次
        max_workers = min(n_batches, max(1, CPU_COUNT // 50))  # 每个进程至少分配50个核心
        print(f"  Using {max_workers} parallel workers for {n_batches} batches")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(parallel_umap_batch, batch_args))
        
        # 合并所有批次的结果
        full_embedding = np.vstack(embeddings)
    
    print(f"  Batch UMAP done in {time.time()-t0:.2f}s")
    return full_embedding

def parallel_stratified_sampling(data, labels, max_samples_per_class=50000, total_max_samples=500000):
    """优化的分层采样，混合使用GPU和CPU"""
    print("Performing optimized stratified sampling...")
    t0 = time.time()
    
    # 使用NumPy在CPU上进行快速操作
    unique_labels = np.unique(labels[labels > 0])
    
    # 并行计算每个类别的样本数
    def count_class_samples(label):
        return label, np.sum(labels == label)
    
    with ThreadPoolExecutor(max_workers=min(len(unique_labels), OPTIMAL_THREADS)) as executor:
        class_counts = dict(executor.map(count_class_samples, unique_labels))
    
    # 动态分配采样数量
    total_available = sum(class_counts.values())
    if total_available > total_max_samples:
        scale_factor = total_max_samples / total_available
        samples_per_class = {
            label: min(max(int(count * scale_factor), 1000), count)
            for label, count in class_counts.items()
        }
    else:
        samples_per_class = {
            label: min(count, max_samples_per_class)
            for label, count in class_counts.items()
        }
    
    # 并行执行采样
    def sample_class(label):
        label_indices = np.where(labels == label)[0]
        n_available = len(label_indices)
        n_samples = samples_per_class[label]
        
        if n_samples < n_available:
            sampled_indices = np.random.choice(label_indices, size=n_samples, replace=False)
        else:
            sampled_indices = label_indices
        
        return sampled_indices, label, n_available, n_samples
    
    with ThreadPoolExecutor(max_workers=min(len(unique_labels), OPTIMAL_THREADS)) as executor:
        results = list(executor.map(sample_class, unique_labels))
    
    # 收集结果
    sampled_indices = []
    sampling_info = {}
    
    for indices, label, n_available, n_samples in results:
        sampled_indices.extend(indices)
        sampling_info[int(label)] = {'original': n_available, 'sampled': n_samples}
        print(f"  Class {label:2d}: {n_available:8,} → {n_samples:6,} samples")
    
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)
    
    sampled_data = data[sampled_indices]
    sampled_labels = labels[sampled_indices]
    
    total_sampled = len(sampled_indices)
    print(f"  Parallel sampling completed in {time.time()-t0:.2f}s")
    print(f"  Total: {total_available:,} → {total_sampled:,} samples")
    
    return sampled_data, sampled_labels, sampling_info

def compute_clustering_metrics(umap_Y, labels, sample_size=50000):
    """计算聚类指标，使用多线程加速"""
    print("  Computing clustering metrics...")
    t0 = time.time()
    
    n_samples = umap_Y.shape[0]
    if n_samples > sample_size:
        indices = np.random.choice(n_samples, size=sample_size, replace=False)
        Y_sample = umap_Y[indices]
        labels_sample = labels[indices]
    else:
        Y_sample = umap_Y
        labels_sample = labels
    
    try:
        # 使用多线程计算
        from joblib import parallel_backend
        with parallel_backend('threading', n_jobs=OPTIMAL_THREADS):
            silhouette = silhouette_score(Y_sample, labels_sample, metric='euclidean')
            db_index = davies_bouldin_score(Y_sample, labels_sample)
        
        print(f"    Silhouette Score: {silhouette:.4f}")
        print(f"    Davies-Bouldin Index: {db_index:.4f}")
    except Exception as e:
        print(f"    Warning: Failed to compute metrics: {e}")
        silhouette = np.nan
        db_index = np.nan
    
    print(f"  Metrics computed in {time.time()-t0:.2f}s")
    return {'silhouette': silhouette, 'davies_bouldin': db_index}

def create_nature_visualization(umap_Y, labels, output_path, sampling_info=None, metrics=None, no_pca=True):
    """创建Nature风格的可视化"""
    print("  Creating UMAP plot...")
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    
    n_pts = umap_Y.shape[0]
    if n_pts > 200000:
        sz, alpha = 0.1, 0.4
    elif n_pts > 100000:
        sz, alpha = 0.2, 0.5
    elif n_pts > 50000:
        sz, alpha = 0.3, 0.6
    else:
        sz, alpha = 0.5, 0.7
    
    # 颜色映射
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18]
    custom_colors = colors[selected_indices]
    custom_cmap = ListedColormap(custom_colors)
    
    scatter = ax.scatter(
        umap_Y[:, 0], umap_Y[:, 1],
        c=labels - 1,
        cmap=custom_cmap,
        s=sz,
        alpha=alpha,
        rasterized=True,
        edgecolors='none'
    )
    
    ax.set_xlabel('UMAP 1', fontsize=8)
    ax.set_ylabel('UMAP 2', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # 添加指标
    if metrics is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        text_x = xlim[1] - 0.05 * (xlim[1] - xlim[0])
        text_y = ylim[1] - 0.05 * (ylim[1] - ylim[0])
        
        metric_text = []
        if not np.isnan(metrics['silhouette']):
            metric_text.append(f"Silhouette: {metrics['silhouette']:.3f}")
        if not np.isnan(metrics['davies_bouldin']):
            metric_text.append(f"Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
        
        if metric_text:
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
    
    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(17))
    cbar.set_ticklabels(np.arange(1, 18))
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Class', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                format='png', facecolor='white', edgecolor='none')
    
    # 保存信息
    if sampling_info:
        info_path = output_path.replace('.png', '_info.txt')
        with open(info_path, 'w') as f:
            f.write("Sampling Information:\n")
            f.write("="*50 + "\n")
            for label, info in sampling_info.items():
                f.write(f"Class {label:2d}: {info['original']:8,} → {info['sampled']:6,} samples\n")
            if metrics:
                f.write("\nMetrics:\n")
                f.write(f"Silhouette Score: {metrics['silhouette']:.4f}\n")
                f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}\n")
            f.write(f"\nNote: This visualization was created WITHOUT PCA reduction\n")

def main():
    # 配置
    # input_file = '/scratch/zf281/btfm_representation/austrian_crop/austrian_crop_efm_v1.0.npy'
    # input_file = '/scratch/zf281/btfm_representation/austrian_crop/austria_Presto_embeddings.npy'
    input_file = '/scratch/zf281/btfm_representation/austrian_crop/mpc_pipeline_fsdp_20250604_100313.npy'
    label_file = '/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes.npy'
    output_dir = '/maps/zf281/btfm4rs/src/utils/'
    
    # 参数设置（针对16G显存优化）
    MAX_SAMPLES_PER_CLASS = 50000
    TOTAL_MAX_SAMPLES = 500000
    BATCH_SIZE_UMAP = 50000  # 减小批处理大小以适应单GPU
    
    # 对于高维数据，可能需要调整UMAP参数
    UMAP_N_NEIGHBORS = 50  # 增加邻居数以更好地捕捉高维结构
    UMAP_MIN_DIST = 0.1
    
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    
    print("="*60)
    print("UMAP VISUALIZATION - NVIDIA GPU OPTIMIZED VERSION (NO PCA)")
    print("="*60)
    print(f"Device: {device}")
    print(f"CPU Cores: {CPU_COUNT} (using {OPTIMAL_THREADS} threads)")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"GPU Compute Capability: {props.major}.{props.minor}")
    
    # 显示系统内存信息
    mem = psutil.virtual_memory()
    print(f"System Memory: {mem.total / 1e9:.1f} GB (Available: {mem.available / 1e9:.1f} GB)")
    print("="*60)

    # 1. 加载标签
    print("\n[Step 1] Loading labels...")
    labels = np.load(label_file)
    label_shape = labels.shape[:2]
    labels_flat = labels.reshape(-1)
    print(f"  Label shape: {labels.shape}")
    
    # 2. 加载特征数据
    print("\n[Step 2] Loading features...")
    t0 = time.time()
    
    # 根据文件大小决定是否使用内存映射
    file_size = os.path.getsize(input_file)
    use_mmap = file_size > 10e9  # 大于10GB使用内存映射
    
    if use_mmap:
        print(f"  Using memory mapping for large file ({file_size/1e9:.1f} GB)")
        data = np.load(input_file, mmap_mode='r')
    else:
        data = np.load(input_file)
    
    print(f"  Feature shape: {data.shape} loaded in {time.time()-t0:.2f}s")
    
    # 检查是否需要resize
    if data.shape[:2] != label_shape:
        print("  Resizing features to match label dimensions...")
        if isinstance(data, np.memmap):
            # 对于内存映射，分批加载和处理
            print("  Loading memmap data into memory for resizing...")
            data = np.array(data)
        data = resize_feature_data_torch(data, label_shape)
    
    # 转换为float32并展平
    if data.dtype != np.float32:
        print("  Converting to float32...")
        data = data.astype(np.float32)
    
    H, W, C = data.shape
    data_flat = data.reshape(-1, C)
    data_flat = np.nan_to_num(data_flat, nan=0.0)
    
    # 3. 过滤背景
    print("\n[Step 3] Filtering background...")
    mask = labels_flat > 0
    data_sel = data_flat[mask]
    labels_sel = labels_flat[mask]
    print(f"  Filtered: {data_sel.shape[0]:,} samples remain")
    print(f"  Feature dimension: {data_sel.shape[1]}")
    
    # 清理内存
    del data_flat, labels_flat, mask
    if 'data' in locals():
        del data
    
    # 4. 分层采样
    print("\n[Step 4] Stratified sampling...")
    data_sampled, labels_sampled, sampling_info = parallel_stratified_sampling(
        data_sel, labels_sel,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        total_max_samples=TOTAL_MAX_SAMPLES
    )
    
    del data_sel, labels_sel
    
    # 5. 直接使用原始数据进行UMAP（跳过PCA）
    print("\n[Step 5] Skipping PCA - Using original features for UMAP")
    print(f"  Feature dimension for UMAP: {data_sampled.shape[1]}")
    data_for_umap = data_sampled
    
    # 6. UMAP（优化版本）
    print("\n[Step 6] UMAP embedding...")
    print("  Note: High-dimensional UMAP may take longer than PCA-reduced version")
    umap_Y = optimized_batch_umap(
        data_for_umap, 
        labels_sampled, 
        batch_size=BATCH_SIZE_UMAP,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST
    )
    
    # 7. 计算指标
    print("\n[Step 7] Computing metrics...")
    metrics = compute_clustering_metrics(umap_Y, labels_sampled)
    
    # 8. 可视化
    print("\n[Step 8] Creating visualization...")
    png_prefix = input_file.split('/')[-1].replace('.npy', '')
    out_png = os.path.join(output_dir, f"{png_prefix}_umap_no_pca_nvidia.png")
    create_nature_visualization(
        umap_Y,
        labels_sampled,
        out_png,
        sampling_info=sampling_info,
        metrics=metrics,
        no_pca=True
    )
    
    # 最终统计
    total_time = time.time() - t_start
    print("\n" + "="*60)
    print("FINAL SUMMARY:")
    print("="*60)
    print(f"Processing completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Output saved to: {out_png}")
    print(f"Visualization created WITHOUT PCA (using {data_for_umap.shape[1]}-dimensional features)")
    if torch.cuda.is_available():
        allocated, reserved, free = check_gpu_memory()
        print(f"Final GPU Memory - Allocated: {allocated:.1f}GB, Free: {free:.1f}GB")
    print("="*60)

if __name__ == "__main__":
    # 确保使用所有CPU核心
    torch.set_num_threads(OPTIMAL_THREADS)
    
    # 运行主程序
    main()