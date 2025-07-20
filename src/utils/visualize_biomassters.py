#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版本：生物质掩膜生成器
1. 读取 biomassters train_agbm / test_agbm 下所有 *_agbm.tif(.tiff)
2. 计算联合包络 + 3 km buffer，按 EPSG:3067 生成二值掩膜栅格
3. 以 WGS-84 世界底图 + 半透明掩膜可视化，保存 PNG

优化特性：
- 添加了详细的日志记录
- 进度条显示处理进度
- 优化的代码结构
- 错误处理和恢复
- 性能优化
- 固定输出分辨率为 10m
"""

# ---------------------------------------------------------------------------
# 导入和配置
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import os
import sys
import glob
import logging
import concurrent.futures as cf
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform_bounds, calculate_default_transform, reproject
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 配置和常量
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """配置类"""
    WORLD_SHP: str = "/mnt/e/Codes/btfm4rs/data/global_map/ne_110m_admin_0_countries.shp"
    ROOT_DIR: str = "/mnt/e/Codes/btfm4rs/data/downstream/biomassters"
    BUFFER_M: int = 3000  # 3 km
    DEST_CRS: CRS = CRS.from_epsg(3067)  # ETRS89 / TM35FIN
    N_WORKERS: int = 16  # 并行读取线程
    MASK_NAME: str = "agbm_footprint_mask.tif"
    FIG_NAME: str = "agbm_footprint_world.png"
    VIZ_RESOLUTION: float = 0.0002  # WGS-84 可视化分辨率
    PIXEL_SIZE_M: float = 10.0  # 固定输出分辨率为 10 米

config = Config()

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
def setup_logging(log_file: str = "biomass_mask.log"):
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ---------------------------------------------------------------------------
# 文件收集和验证
# ---------------------------------------------------------------------------
def collect_tifs(directory: str) -> List[str]:
    """收集目录下的所有TIFF文件"""
    logger.info(f"正在扫描目录：{directory}")
    tifs = sorted(
        glob.glob(os.path.join(directory, "*_agbm.tif")) +
        glob.glob(os.path.join(directory, "*_agbm.tiff"))
    )
    logger.info(f"找到 {len(tifs)} 个TIFF文件")
    return tifs

def validate_tif_file(path: str) -> bool:
    """验证TIFF文件是否可读"""
    try:
        with rasterio.open(path) as src:
            _ = src.bounds, src.crs
        return True
    except Exception as e:
        logger.warning(f"文件损坏或无法读取: {path}, 错误: {e}")
        return False

# ---------------------------------------------------------------------------
# 并行元数据读取
# ---------------------------------------------------------------------------
class TiffMeta:
    """TIFF元数据类"""
    def __init__(self, bounds, crs, pixel_size):
        self.bounds = bounds
        self.crs = crs
        self.pixel_size = pixel_size

def read_tiff_metadata(path: str) -> Optional[TiffMeta]:
    """读取单个TIFF文件的元数据"""
    try:
        with rasterio.open(path) as src:
            bounds = src.bounds
            crs = src.crs
            pixel_size = max(abs(src.transform.a), abs(src.transform.e))
            return TiffMeta(bounds, crs, pixel_size)
    except Exception as e:
        logger.error(f"读取文件元数据失败: {path}, 错误: {e}")
        return None

def read_all_metadata(tiff_paths: List[str]) -> List[TiffMeta]:
    """并行读取所有TIFF文件的元数据"""
    logger.info(f"开始并行读取 {len(tiff_paths)} 个文件的元数据（{config.N_WORKERS} 个工作线程）")
    
    valid_metas = []
    with cf.ThreadPoolExecutor(max_workers=config.N_WORKERS) as executor:
        with tqdm(total=len(tiff_paths), desc="读取元数据", unit="文件") as pbar:
            futures = [executor.submit(read_tiff_metadata, path) for path in tiff_paths]
            
            for future in cf.as_completed(futures):
                meta = future.result()
                if meta:
                    valid_metas.append(meta)
                pbar.update(1)
    
    logger.info(f"成功读取 {len(valid_metas)} 个文件的元数据")
    return valid_metas

# ---------------------------------------------------------------------------
# 几何处理
# ---------------------------------------------------------------------------
def convert_bounds_to_target_crs(meta: TiffMeta) -> box:
    """将边界转换到目标坐标系"""
    try:
        minx, miny, maxx, maxy = transform_bounds(
            meta.crs, config.DEST_CRS, 
            meta.bounds.left, meta.bounds.bottom, 
            meta.bounds.right, meta.bounds.top, 
            densify_pts=0
        )
        return box(minx, miny, maxx, maxy)
    except Exception as e:
        logger.error(f"坐标转换失败: {e}")
        return None

def create_union_polygon(metas: List[TiffMeta]) -> Tuple[box, box]:
    """创建所有几何体的并集，并应用缓冲区"""
    logger.info("转换边界到目标坐标系...")
    
    polys_3067 = []
    with tqdm(metas, desc="转换坐标", unit="文件") as pbar:
        for meta in pbar:
            poly = convert_bounds_to_target_crs(meta)
            if poly:
                polys_3067.append(poly)
    
    logger.info(f"计算 {len(polys_3067)} 个多边形的并集...")
    union_poly = unary_union(polys_3067)
    
    logger.info(f"应用 {config.BUFFER_M}m 缓冲区...")
    buffer_poly = union_poly.buffer(config.BUFFER_M)
    
    # 使用固定分辨率 10m
    logger.info(f"使用固定像素大小: {config.PIXEL_SIZE_M}m")
    
    return union_poly, buffer_poly

# ---------------------------------------------------------------------------
# 掩膜生成
# ---------------------------------------------------------------------------
def create_mask_raster(union_poly: box, buffer_poly: box) -> Tuple[np.ndarray, Affine, Tuple[int, int]]:
    """创建掩膜栅格，使用固定的10m分辨率"""
    # 使用buffer_poly的边界来定义栅格范围
    minx, miny, maxx, maxy = buffer_poly.bounds
    
    # 使用固定的像素大小 (10m)
    pixel_size = config.PIXEL_SIZE_M
    
    # 计算栅格尺寸
    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))
    
    logger.info(f"创建掩膜栅格: {width} x {height} 像素，分辨率: {pixel_size}m")
    
    # 创建仿射变换
    transform = Affine(pixel_size, 0, minx,
                      0, -pixel_size, maxy)
    
    # 栅格化原始的union_poly（不包括buffer）
    logger.info("正在栅格化多边形...")
    mask_arr = rasterize(
        [(union_poly, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )
    
    return mask_arr, transform, (height, width)

def save_mask(mask_arr: np.ndarray, transform: Affine, dimensions: Tuple[int, int], 
              output_path: str) -> None:
    """保存掩膜栅格到文件"""
    height, width = dimensions
    
    logger.info(f"保存掩膜栅格到: {output_path}")
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=config.DEST_CRS,
        transform=transform,
        compress="lzw"  # 添加压缩以减小文件大小
    ) as dst:
        dst.write(mask_arr, 1)
    
    logger.info(f"掩膜栅格保存成功，文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def create_visualization(mask_path: str, world_shp_path: str, output_fig_path: str) -> None:
    """创建可视化图像"""
    logger.info("开始创建可视化...")
    
    # 重投影掩膜到WGS-84
    logger.info("重投影掩膜到WGS-84...")
    with rasterio.open(mask_path) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds, 
            resolution=config.VIZ_RESOLUTION
        )
        mask_wgs84 = np.zeros((dst_height, dst_width), dtype=np.uint8)
        
        reproject(
            source=src.read(1),
            destination=mask_wgs84,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
        
        # 获取边界用于可视化裁剪
        mask_bounds_wgs = transform_bounds(
            config.DEST_CRS, "EPSG:4326", 
            *src.bounds
        )
    
    # 屏蔽0值以获得透明背景
    mask_wgs84 = np.ma.masked_equal(mask_wgs84, 0)
    
    # 读取世界地图
    logger.info(f"读取世界地图: {world_shp_path}")
    if not os.path.exists(world_shp_path):
        logger.warning(f"世界地图文件不存在: {world_shp_path}，跳过底图")
        world = None
    else:
        world = gpd.read_file(world_shp_path)
    
    # 创建图形
    logger.info("生成最终图像...")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    # 绘制世界边界
    if world is not None:
        world.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=1)
    
    # 绘制掩膜栅格：半透明红色
    red_cmap = ListedColormap([(1, 0, 0, 0.5)])  # 单色 colormap，α=0.5
    
    # 使用 matplotlib imshow 直接显示栅格
    im = ax.imshow(mask_wgs84, 
                   extent=[mask_bounds_wgs[0], mask_bounds_wgs[2], 
                          mask_bounds_wgs[1], mask_bounds_wgs[3]],
                   cmap=red_cmap,
                   zorder=2,
                   aspect='auto')
    
    # 设置视窗
    ax.set_xlim(mask_bounds_wgs[0], mask_bounds_wgs[2])
    ax.set_ylim(mask_bounds_wgs[1], mask_bounds_wgs[3])
    
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Biomassters AGBM Footprint (buffered 3 km, 10m resolution)", fontsize=14, fontweight='bold')
    
    ax.legend(handles=[Patch(facecolor="red", alpha=0.5, label="AGBM footprint")],
              loc="upper right", fontsize=11)
    
    # 保存图像
    logger.info(f"保存可视化图像到: {output_fig_path}")
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    img_size = os.path.getsize(output_fig_path) / 1024 / 1024
    logger.info(f"可视化图像保存成功，文件大小: {img_size:.2f} MB")

# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    """主程序入口"""
    logger.info("=" * 60)
    logger.info("开始生成生物质掩膜")
    logger.info("=" * 60)
    
    try:
        # 1. 收集所有TIFF文件
        test_dir = os.path.join(config.ROOT_DIR, "test_agbm")
        train_dir = os.path.join(config.ROOT_DIR, "train_agbm")
        
        test_tifs = collect_tifs(test_dir)
        train_tifs = collect_tifs(train_dir)
        
        if not (test_tifs or train_tifs):
            raise FileNotFoundError("未在 test_agbm / train_agbm 目录找到 *_agbm.tif(.tiff) 文件")
        
        all_tifs = test_tifs + train_tifs
        logger.info(f"总共找到 {len(all_tifs)} 个TIFF文件")
        
        # 2. 验证文件（可选，对于大量文件可能耗时）
        logger.info("跳过文件验证以提高性能...")
        
        # 3. 读取元数据
        metas = read_all_metadata(all_tifs)
        if not metas:
            raise RuntimeError("没有成功读取任何文件的元数据")
        
        # 4. 创建并集多边形
        union_poly, buffer_poly = create_union_polygon(metas)
        
        # 5. 创建掩膜栅格 (使用固定的10m分辨率)
        mask_arr, transform, dimensions = create_mask_raster(union_poly, buffer_poly)
        
        # 6. 保存掩膜
        save_mask(mask_arr, transform, dimensions, config.MASK_NAME)
        
        # 7. 创建可视化
        create_visualization(config.MASK_NAME, config.WORLD_SHP, config.FIG_NAME)
        
        logger.info("=" * 60)
        logger.info("生物质掩膜生成完成！")
        logger.info(f"输出文件：")
        logger.info(f"  - 掩膜文件: {config.MASK_NAME} (10m分辨率)")
        logger.info(f"  - 可视化图像: {config.FIG_NAME}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        logger.exception("详细错误信息:")
        sys.exit(1)

if __name__ == "__main__":
    main()