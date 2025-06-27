import os
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import tempfile
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

def create_rgb_tiff(grid_folder, tiff_dir, output_dir, downsample_factor=10, target_crs='EPSG:32630'):
    """为每个grid创建一个临时的RGB TIFF文件，统一到目标CRS"""
    try:
        grid_name = os.path.basename(grid_folder)
        npy_path = os.path.join(grid_folder, f"{grid_name}.npy")
        tiff_path = os.path.join(tiff_dir, f"{grid_name}.tiff")
        
        if not os.path.exists(npy_path) or not os.path.exists(tiff_path):
            return None
        
        # 读取地理信息
        with rasterio.open(tiff_path) as src:
            src_transform = src.transform
            src_crs = src.crs
            bounds = src.bounds
        
        # 使用内存映射读取npy
        npy_mmap = np.load(npy_path, mmap_mode='r')
        h, w, c = npy_mmap.shape
        
        # 降采样并提取RGB
        rgb_data = npy_mmap[::downsample_factor, ::downsample_factor, :3] # Assumed input range is [-127, 127]
        
        # Scale to [0, 255]. Using float for calculation and clipping for robustness.
        rgb_data = (rgb_data.astype(np.float32) + 127.0) * (255.0 / 254.0)
        rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
        
        # 创建临时TIFF文件
        temp_tiff_path = os.path.join(output_dir, f"{grid_name}_rgb.tif")
        
        # 调整变换矩阵以适应降采样
        downsample_transform = src_transform * rasterio.Affine.scale(downsample_factor, downsample_factor)
        height, width = rgb_data.shape[:2]
        
        # 如果CRS与目标CRS不同，需要重投影
        if src_crs != target_crs:
            # 计算新的变换和尺寸
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, target_crs, width, height,
                left=bounds.left, bottom=bounds.bottom, 
                right=bounds.right, top=bounds.top
            )
            
            # 创建重投影后的数组
            dst_rgb = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
            
            # 对每个波段进行重投影
            for i in range(3):
                reproject(
                    source=rgb_data[:, :, i],
                    destination=dst_rgb[:, :, i],
                    src_transform=downsample_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
            
            # 使用重投影后的数据
            rgb_data = dst_rgb
            transform = dst_transform
            crs = target_crs
            height, width = dst_height, dst_width
        else:
            transform = downsample_transform
            crs = src_crs
        
        # 写入TIFF文件
        with rasterio.open(
            temp_tiff_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype='uint8',
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            for i in range(3):
                dst.write(rgb_data[:, :, i], i + 1)
        
        del npy_mmap
        return {
            'path': temp_tiff_path,
            'src_crs': str(src_crs),
            'grid_name': grid_name
        }
        
    except Exception as e:
        print(f"Error processing {grid_folder}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 设置路径
    base_dir = "/scratch/zf281/btfm_representation/uk/2024"
    tiff_dir = "/scratch/zf281/global_map_0.1_degree_tiff"
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='uk_map_temp_')
    print(f"Using temporary directory: {temp_dir}")
    
    # 获取所有grid子文件夹
    grid_folders = [d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith("grid_")]
    print(f"Found {len(grid_folders)} grid folders")
    
    # 目标CRS - 使用EPSG:32630 (UTM 30N)作为英国的主要投影
    target_crs = 'EPSG:32630'
    
    # 第一步：并行创建RGB TIFF文件
    print(f"\nStep 1: Creating RGB TIFF files (target CRS: {target_crs})...")
    num_workers = min(96, len(grid_folders))
    
    tiff_info = []
    crs_stats = defaultdict(int)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_grid = {
            executor.submit(create_rgb_tiff, str(folder), tiff_dir, temp_dir, target_crs=target_crs): folder 
            for folder in grid_folders
        }
        
        with tqdm(total=len(grid_folders), desc="Creating RGB TIFFs") as pbar:
            for future in as_completed(future_to_grid):
                result = future.result()
                if result is not None:
                    tiff_info.append(result)
                    crs_stats[result['src_crs']] += 1
                pbar.update(1)
    
    print(f"\nCreated {len(tiff_info)} RGB TIFF files")
    print("CRS distribution:")
    for crs, count in crs_stats.items():
        print(f"  {crs}: {count} files")
    
    if not tiff_info:
        print("No valid TIFF files created")
        return
    
    # 提取路径
    tiff_paths = [info['path'] for info in tiff_info]
    
    # 第二步：分批处理merge（避免文件句柄过多）
    print("\nStep 2: Creating mosaic...")
    
    # 将文件分批处理
    batch_size = 500  # 减小批次大小
    all_mosaics = []
    
    for i in range(0, len(tiff_paths), batch_size):
        batch_paths = tiff_paths[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(tiff_paths) + batch_size - 1)//batch_size}")
        
        # 打开批次中的所有文件
        src_files = []
        for path in tqdm(batch_paths, desc="Opening files", leave=False):
            src_files.append(rasterio.open(path))
        
        # 合并批次
        try:
            mosaic, out_trans = merge(src_files, res=(100, 100))
            
            # 保存批次结果
            batch_mosaic_path = os.path.join(temp_dir, f'batch_mosaic_{i//batch_size}.tif')
            
            height, width = mosaic.shape[1:]
            with rasterio.open(
                batch_mosaic_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=3,
                dtype='uint8',
                crs=target_crs,
                transform=out_trans,
                compress='lzw'
            ) as dst:
                dst.write(mosaic)
            
            all_mosaics.append(batch_mosaic_path)
            
        except Exception as e:
            print(f"Error merging batch: {str(e)}")
            
        finally:
            # 关闭文件
            for src in src_files:
                src.close()
    
    print(f"\nCreated {len(all_mosaics)} batch mosaics")
    
    # 如果有多个批次，再次合并
    if len(all_mosaics) > 1:
        print("\nMerging batch mosaics...")
        src_files = [rasterio.open(path) for path in all_mosaics]
        
        try:
            final_mosaic, final_trans = merge(src_files, res=(100, 100))
        finally:
            for src in src_files:
                src.close()
    else:
        # 只有一个批次，直接读取
        with rasterio.open(all_mosaics[0]) as src:
            final_mosaic = src.read()
            final_trans = src.transform
    
    # 第三步：重投影到WGS84
    print("\nStep 3: Reprojecting to WGS84...")
    
    src_crs = target_crs
    dst_crs = 'EPSG:4326'
    
    # 计算边界
    height, width = final_mosaic.shape[1:]
    left = final_trans.c
    top = final_trans.f
    right = left + width * final_trans.a
    bottom = top + height * final_trans.e
    
    # 计算WGS84中的输出尺寸
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height,
        left=left, bottom=bottom, right=right, top=top
    )
    
    print(f"Final output size (WGS84): {dst_width} x {dst_height}")
    
    # 创建输出数组
    final_output = np.zeros((3, dst_height, dst_width), dtype=np.uint8)
    
    # 重投影
    for i in range(3):
        reproject(
            source=final_mosaic[i],
            destination=final_output[i],
            src_transform=final_trans,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
    
    # 转换为适合显示的格式
    rgb_array = np.transpose(final_output, (1, 2, 0))
    
    # 第四步：可视化
    print("\nStep 4: Creating visualization...")
    
    # 计算地理范围
    west = dst_transform.c
    north = dst_transform.f
    east = west + dst_transform.a * dst_width
    south = north + dst_transform.e * dst_height
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(20, 15))
    
    # 显示图像
    im = ax.imshow(rgb_array, extent=[west, east, south, north], 
                   interpolation='nearest', origin='upper')
    
    # 设置标题和标签
    ax.set_title('UK Map Visualization (100m resolution, WGS84)', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴格式
    ax.ticklabel_format(useOffset=False)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = "uk_map_100m_wgs84_fixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to: {output_path}")
    
    # 保存最终的GeoTIFF
    output_tiff = "uk_map_100m_wgs84_fixed.tiff"
    
    with rasterio.open(
        output_tiff,
        'w',
        driver='GTiff',
        height=dst_height,
        width=dst_width,
        count=3,
        dtype='uint8',
        crs='EPSG:4326',
        transform=dst_transform,
        compress='lzw',
        tiled=True,
        blockxsize=512,
        blockysize=512
    ) as dst:
        dst.write(final_output)
    
    print(f"GeoTIFF saved to: {output_tiff}")
    
    # 清理临时文件
    print("\nCleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir)
    
    # 显示统计信息
    print(f"\nFinal image statistics:")
    print(f"  - Image size: {dst_width} x {dst_height}")
    print(f"  - Min RGB values: {np.min(rgb_array, axis=(0,1))}")
    print(f"  - Max RGB values: {np.max(rgb_array, axis=(0,1))}")
    
    # Fix: Format each RGB channel's mean value separately
    mean_rgb = np.mean(rgb_array, axis=(0,1))
    print(f"  - Mean RGB values: [{mean_rgb[0]:.2f}, {mean_rgb[1]:.2f}, {mean_rgb[2]:.2f}]")
    
    coverage = np.sum(np.any(rgb_array > 0, axis=2)) / (dst_height * dst_width) * 100
    print(f"  - Coverage: {coverage:.2f}%")
    
    plt.show()

if __name__ == "__main__":
    main()