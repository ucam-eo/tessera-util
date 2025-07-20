import os
import numpy as np
import argparse
import cv2  # 用于保存png图像和resize

def normalize_channel(channel):
    """对单个通道进行最小—最大归一化至 [0, 255]，返回 uint8 数组"""
    if channel.size == 0:
        # 如果通道为空，则返回与channel形状相同的全0数组
        return np.zeros_like(channel, dtype=np.uint8)
    c_min = channel.min()
    c_max = channel.max()
    if c_max > c_min:
        norm = (channel - c_min) / (c_max - c_min) * 255
    else:
        norm = np.zeros_like(channel)
    return norm.astype(np.uint8)

def process_rgb_patch(bands_data, masks_data, x, y, patch_size):
    """
    对于给定的patch区域，从 bands_data 和 masks_data 中选取有效性最高的时间步，
    提取前三个波段（原始顺序为 [r, b, g]，调整为 [r, g, b]），并对各通道归一化。
    返回归一化后的RGB patch，数据类型为 uint8，形状 (patch_size, patch_size, 3)。
    """
    # 提取该区域所有时间步的数据
    bands_patch = bands_data[:, x: x+patch_size, y: y+patch_size, :]  # shape: (T, patch_size, patch_size, B)
    masks_patch = masks_data[:, x: x+patch_size, y: y+patch_size]      # shape: (T, patch_size, patch_size)
    
    # 计算每个时间步有效（未被云雾遮挡）像素的数量
    valid_counts = np.sum(masks_patch == 1, axis=(1, 2))
    t_idx = np.argmax(valid_counts)
    
    # 取出该时间步的前三个波段（原始顺序为 [r, b, g]），并调整为 [r, g, b]
    rgb_patch = bands_patch[t_idx, :, :, :3]
    rgb_patch = rgb_patch[..., [0, 2, 1]]  # 调整顺序
    
    # 对每个通道归一化
    norm_rgb = np.zeros_like(rgb_patch, dtype=np.uint8)
    for ch in range(3):
        norm_rgb[..., ch] = normalize_channel(rgb_patch[..., ch])
        
    return norm_rgb

def generate_patches(rep_file, target_file, bands_file, masks_file, output_dir, patch_size, num_patches, max_overlap):
    """
    从 representation、target 数据中抽取 patch_size x patch_size 大小的 patch，
    同时从多时相的RGB图像数据中提取对应的RGB patch（选择云雾遮挡最少的时刻），
    分别保存到 output_dir 下的 representation、target、rgb 文件夹中。

    参数:
        rep_file: representation数据的npy文件路径，形状为 (H, W, 128)
        target_file: 标签数据的npy文件路径，形状为 (1, H, W) 或 (H, W)
        bands_file: RGB图像数据的npy文件路径，形状为 (T, H, W, B)，前三个波段为 r, b, g
        masks_file: 云雾遮挡掩膜的npy文件路径，形状为 (T, H, W)，值为1表示未被遮挡
        output_dir: 保存patch的根目录
        patch_size: 每个patch的尺寸（例如 100 表示 100x100 的patch）
        num_patches: 需要生成的patch数量（仅在随机采样模式下使用）
        max_overlap: 最大允许的重叠比例，取值范围 [0, 1)。
                     如果指定该值，则采用顺序扫描模式（从上到下、从左到右），忽略 num_patches 参数，
                     例如 0.5 表示相邻patch在宽或高方向最多有50%的重叠。
    """
    # 创建保存 patch 的文件夹
    rep_out_dir = os.path.join(output_dir, 'representation')
    target_out_dir = os.path.join(output_dir, 'target')
    rgb_out_dir = os.path.join(output_dir, 'rgb')
    os.makedirs(rep_out_dir, exist_ok=True)
    os.makedirs(target_out_dir, exist_ok=True)
    os.makedirs(rgb_out_dir, exist_ok=True)

    # 如果文件夹中已有 patch，则先清空
    for d in [rep_out_dir, target_out_dir, rgb_out_dir]:
        for file in os.listdir(d):
            os.remove(os.path.join(d, file))
    
    # 加载数据
    rep_data = np.load(rep_file)      # 形状 (H, W, 128)
    
    # 如果数据形状为int，则归一化
    if rep_data.dtype == np.int8:
        scale_data = np.load(rep_file.replace('.npy', '_scale.npy'))
        rep_data = rep_data.astype(np.float32) / scale_data
    
    target_data = np.load(target_file)  # 形状可能为 (1, H, W) 或 (H, W)
    bands_data = np.load(bands_file)    # 形状 (T, H, W, B)
    masks_data = np.load(masks_file)    # 形状 (T, H, W)
    
    # 去除 target 数据多余的维度（如果有）
    if target_data.ndim == 3 and target_data.shape[0] == 1:
        target_data = target_data[0]
        
    # --- 检查并resize representation数据 ---
    rep_h, rep_w, _ = rep_data.shape
    target_h, target_w = target_data.shape

    if (rep_h, rep_w) != (target_h, target_w):
        print(f"Representation ({rep_h}, {rep_w}) 和 Target ({target_h}, {target_w}) 空间尺寸不匹配。")
        print(f"正在将 Representation resize 为 ({target_h}, {target_w})...")
        rep_data = cv2.resize(rep_data, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        print("Resize 完成。")

    H, W, _ = rep_data.shape
    if target_data.shape != (H, W):
        raise ValueError(f"Representation 数据 resize 后形状为 {rep_data.shape}，但 target 数据形状为 {target_data.shape}，尺寸仍然不匹配")
    
    patch_count = 0
    # 如果启用 max_overlap，则采用顺序扫描模式
    if max_overlap is not None:
        if not (0 <= max_overlap < 1):
            raise ValueError("max_overlap 必须在 [0, 1) 的范围内")
        stride = patch_size - int(max_overlap * patch_size)
        if stride <= 0:
            stride = 1
        
        print(f"使用 max_overlap 模式进行顺序采样，步长为 {stride}，忽略 --num_patches 参数")
        for x in range(0, H - patch_size + 1, stride):
            for y in range(0, W - patch_size + 1, stride):
                rep_patch = rep_data[x: x + patch_size, y: y + patch_size, :]
                target_patch = target_data[x: x + patch_size, y: y + patch_size]
                
                # 如果 target patch 全部为 NaN，则跳过
                if np.isnan(target_patch).all():
                    print(f"patch 坐标 ({x}, {y}) 标签全为 NaN，跳过")
                    continue

                # --- 新增代码: 填充含有部分NaN的patch ---
                if np.isnan(target_patch).any():
                    # 计算当前patch内非NaN像素的均值
                    patch_mean = np.nanmean(target_patch)
                    # 如果patch_mean因为某种原因还是NaN（例如，被上面检查遗漏的全NaN patch），则用0填充
                    if np.isnan(patch_mean):
                        patch_mean = 0.0
                    # 使用计算出的均值填充NaN值
                    target_patch = np.nan_to_num(target_patch, nan=patch_mean)
                # --- 新增代码结束 ---

                # 处理RGB patch
                rgb_patch = process_rgb_patch(bands_data, masks_data, x, y, patch_size)
                
                # 保存三个产品
                rep_patch_path = os.path.join(rep_out_dir, f'rep_patch_{patch_count:04d}.npy')
                target_patch_path = os.path.join(target_out_dir, f'target_patch_{patch_count:04d}.npy')
                rgb_patch_path = os.path.join(rgb_out_dir, f'rgb_patch_{patch_count:04d}.png')
                
                np.save(rep_patch_path, rep_patch)
                np.save(target_patch_path, target_patch)
                bgr_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_patch_path, bgr_patch)
                
                print(f"patch {patch_count:04d} 已保存，位置: ({x}, {y})")
                patch_count += 1
        print(f"共生成有效 patch 数量: {patch_count}")
    
    else:
        # 随机采样模式
        print("使用随机采样模式")
        attempts = 0
        max_attempts = num_patches * 20  # 增加尝试次数以应对可能的大量NaN区域
        while patch_count < num_patches and attempts < max_attempts:
            attempts += 1
            x = np.random.randint(0, H - patch_size + 1)
            y = np.random.randint(0, W - patch_size + 1)
            
            rep_patch = rep_data[x: x + patch_size, y: y + patch_size, :]
            target_patch = target_data[x: x + patch_size, y: y + patch_size]
            
            # 如果 target patch 全部为 NaN，则跳过
            if np.isnan(target_patch).all():
                continue

            # --- 新增代码: 填充含有部分NaN的patch ---
            if np.isnan(target_patch).any():
                # 计算当前patch内非NaN像素的均值
                patch_mean = np.nanmean(target_patch)
                # 如果patch_mean因为某种原因还是NaN，则用0填充
                if np.isnan(patch_mean):
                    patch_mean = 0.0
                # 使用计算出的均值填充NaN值
                target_patch = np.nan_to_num(target_patch, nan=patch_mean)
            # --- 新增代码结束 ---

            # 处理RGB patch
            rgb_patch = process_rgb_patch(bands_data, masks_data, x, y, patch_size)
            
            rep_patch_path = os.path.join(rep_out_dir, f'rep_patch_{patch_count:04d}.npy')
            target_patch_path = os.path.join(target_out_dir, f'target_patch_{patch_count:04d}.npy')
            rgb_patch_path = os.path.join(rgb_out_dir, f'rgb_patch_{patch_count:04d}.png')
            
            np.save(rep_patch_path, rep_patch)
            np.save(target_patch_path, target_patch)
            bgr_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_patch_path, bgr_patch)
            
            print(f"patch {patch_count:04d} 已保存，位置: ({x}, {y})")
            patch_count += 1
        
        if patch_count < num_patches:
            print(f"警告: 仅生成 {patch_count} 个有效 patch，未达到指定的 {num_patches} 个。可能是源数据中NaN区域过多。")
        else:
            print(f"共生成有效 patch 数量: {patch_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从npy数据生成固定大小的patches，并生成对应的RGB图像")
    parser.add_argument('--rep_file', type=str, default='/mnt/e/Codes/btfm4rs/data/representation/presto_embedding_asset_borneo.npy',
                        help="representation数据的npy文件路径")
    parser.add_argument('--target_file', type=str, default='/mnt/e/Codes/btfm4rs/data/downstream/borneo/Danum_2020_chm_lspikefree_subsampled.npy',
                        help="标签数据的npy文件路径")
    parser.add_argument('--bands_file', type=str, default='/mnt/e/Codes/btfm4rs/data/downstream/borneo/50NNL_subset/bands.npy',
                        help="RGB图像数据的npy文件路径")
    parser.add_argument('--masks_file', type=str, default='/mnt/e/Codes/btfm4rs/data/downstream/borneo/50NNL_subset/masks.npy',
                        help="云雾遮挡掩膜的npy文件路径")
    parser.add_argument('--output_dir', type=str, default='/mnt/e/Codes/btfm4rs/data/downstream/borneo_patch',
                        help="保存patch的根目录")
    parser.add_argument('--patch_size', type=int, default=64,
                        help="patch的尺寸，例如100表示100x100的patch")
    parser.add_argument('--num_patches', type=int, default=500,
                        help="需要生成的patch数量（仅在随机采样模式下使用）")
    parser.add_argument('--max_overlap', type=float, default=None,
                        help="最大允许的重叠比例，取值范围 [0, 1)。如果指定该值，则采用顺序扫描模式，忽略 --num_patches 参数")
    
    args = parser.parse_args()
    generate_patches(args.rep_file, args.target_file, args.bands_file, args.masks_file,
                     args.output_dir, args.patch_size, args.num_patches, args.max_overlap)