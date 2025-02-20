#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

def main(args):
    # 输入文件路径
    repr_path = "data/downstream/austrian_crop/inference_repr_33UXP.npy"
    label_path = "data/downstream/austrian_crop/fieldtype_17classes.npy"
    
    # 输出文件夹
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 直接加载整个数据到内存
    print("加载 representation 数据 ...")
    representation = np.load(repr_path)  # shape: (H, W, 128)
    print("加载 fieldtype 数据 ...")
    fieldtype = np.load(label_path)        # shape: (H, W)
    
    H, W, C = representation.shape
    if fieldtype.shape[0] != H or fieldtype.shape[1] != W:
        raise ValueError("representation 和 fieldtype 的尺寸不匹配！")
    
    patch_size = args.patch_size
    
    # 预先获取 fieldtype 中所有类别，并建立颜色映射（颜色顺序固定，确保所有 patch 中同一类别颜色一致）
    unique_labels = np.unique(fieldtype)
    unique_labels = np.sort(unique_labels)
    print("全图中发现类别：", unique_labels)
    
    # 使用 matplotlib 的 tab20 作为颜色盘（最多20种颜色，如果类别数超过20，请自行调整 colormap）
    cmap = plt.get_cmap("tab20", len(unique_labels))
    label_to_color = {}
    for idx, label in enumerate(unique_labels):
        rgba = cmap(idx)
        rgb = tuple(int(255 * x) for x in rgba[:3])
        label_to_color[label] = rgb

    print("类别到颜色的映射：")
    for label, color in label_to_color.items():
        print(f"  类别 {label}: 颜色 {color}")
    
    count = 0

    # 判断是否采用顺序采样模式（当 max_overlap_ratio >= 0 时使用顺序采样，忽略 num_patches 参数）
    if args.max_overlap_ratio >= 0:
        print(f"使用顺序采样模式，最大重叠比例为 {args.max_overlap_ratio}")
        # 计算步长：允许的重叠像素数 = int(patch_size * max_overlap_ratio)
        overlap_pixels = int(patch_size * args.max_overlap_ratio)
        step = patch_size - overlap_pixels
        if step < 1:
            step = 1
        print(f"patch_size={patch_size}, overlap_pixels={overlap_pixels}, 采样步长 step={step}")
        
        for i in range(0, H - patch_size + 1, step):
            for j in range(0, W - patch_size + 1, step):
                # 裁剪 representation 和 label patch
                repr_patch = representation[i:i + patch_size, j:j + patch_size, :]
                label_patch = fieldtype[i:i + patch_size, j:j + patch_size]
                
                # 如果 label_patch 全为 0，则跳过此 patch
                if np.all(label_patch == 0):
                    continue
                
                patch_index = count + 1
                # 保存 npy 文件
                repr_npy_path = os.path.join(output_dir, f"patch_{patch_index}.npy")
                label_npy_path = os.path.join(output_dir, f"label_{patch_index}.npy")
                np.save(repr_npy_path, repr_patch)
                np.save(label_npy_path, label_patch)
                
                # --- 对 representation patch 进行 PCA 降维及可视化 ---
                reshaped_repr = repr_patch.reshape(-1, C)
                pca = PCA(n_components=3)
                repr_pca = pca.fit_transform(reshaped_repr)
                repr_pca = repr_pca.reshape(patch_size, patch_size, 3)
                # 归一化到 [0,1] 便于显示
                min_val = repr_pca.min()
                max_val = repr_pca.max()
                if max_val > min_val:
                    repr_norm = (repr_pca - min_val) / (max_val - min_val)
                else:
                    repr_norm = np.zeros_like(repr_pca)
                repr_png_path = os.path.join(output_dir, f"patch_{patch_index}.png")
                plt.imsave(repr_png_path, repr_norm)
                
                # --- 对 label patch 进行彩色可视化 ---
                label_vis = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                for label, color in label_to_color.items():
                    mask = (label_patch == label)
                    label_vis[mask] = color
                label_png_path = os.path.join(output_dir, f"label_{patch_index}.png")
                plt.imsave(label_png_path, label_vis)
                
                print(f"已保存 patch {patch_index}，位置：({i},{j})")
                count += 1
        print(f"顺序采样模式下，共生成 {count} 个有效 patch。")
    else:
        # 随机采样模式：使用 num_patches 参数
        num_patches = args.num_patches
        rng = np.random.default_rng(args.seed)
        attempts = 0
        max_attempts = num_patches * 100  # 防止死循环的上限
        
        while count < num_patches and attempts < max_attempts:
            attempts += 1
            # 随机选择左上角坐标
            i = rng.integers(0, H - patch_size + 1)
            j = rng.integers(0, W - patch_size + 1)
            
            # 裁剪 representation 和 label patch
            repr_patch = representation[i:i + patch_size, j:j + patch_size, :]
            label_patch = fieldtype[i:i + patch_size, j:j + patch_size]
            
            # 如果 label_patch 全为 0，则跳过此 patch
            if np.all(label_patch == 0):
                continue
            
            patch_index = count + 1
            # 保存 npy 文件
            repr_npy_path = os.path.join(output_dir, f"patch_{patch_index}.npy")
            label_npy_path = os.path.join(output_dir, f"label_{patch_index}.npy")
            np.save(repr_npy_path, repr_patch)
            np.save(label_npy_path, label_patch)
            
            # --- 对 representation patch 进行 PCA 降维及可视化 ---
            reshaped_repr = repr_patch.reshape(-1, C)
            pca = PCA(n_components=3)
            repr_pca = pca.fit_transform(reshaped_repr)
            repr_pca = repr_pca.reshape(patch_size, patch_size, 3)
            # 归一化到 [0,1] 便于显示
            min_val = repr_pca.min()
            max_val = repr_pca.max()
            if max_val > min_val:
                repr_norm = (repr_pca - min_val) / (max_val - min_val)
            else:
                repr_norm = np.zeros_like(repr_pca)
            repr_png_path = os.path.join(output_dir, f"patch_{patch_index}.png")
            plt.imsave(repr_png_path, repr_norm)
            
            # --- 对 label patch 进行彩色可视化 ---
            label_vis = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            for label, color in label_to_color.items():
                mask = (label_patch == label)
                label_vis[mask] = color
            label_png_path = os.path.join(output_dir, f"label_{patch_index}.png")
            plt.imsave(label_png_path, label_vis)
            
            print(f"已保存 patch {patch_index}/{num_patches} (随机采样)")
            count += 1

        if attempts >= max_attempts:
            print("警告：尝试次数达到上限，生成的 patch 数量可能不足预期。")
        else:
            print(f"随机采样模式下，共生成 {count} 个有效 patch。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="裁剪 representation 与 fieldtype patch 并生成可视化图片（跳过 label 全为 0 的 patch）"
    )
    parser.add_argument("--num_patches", type=int, default=3000, help="生成的 patch 数量（仅在随机采样模式下有效）")
    parser.add_argument("--patch_size", type=int, default=64, help="patch 的尺寸（正方形边长）")
    parser.add_argument("--output_dir", type=str, default="data/downstream/austrian_crop_patch", help="保存生成文件的输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # 如果 max_overlap_ratio >= 0，则启用顺序采样模式，忽略 num_patches 参数；否则使用随机采样
    parser.add_argument("--max_overlap_ratio", type=float, default=-1,
                        help="最大重叠比例（0～1）；若设置为>=0，则顺序采样，否则随机采样")
    args = parser.parse_args()
    main(args)
