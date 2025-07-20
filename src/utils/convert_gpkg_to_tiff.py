import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import numpy as np
import pyproj
from pyproj import Transformer
import math

def find_best_utm_zone(geometry):
    """
    根据几何体的中心点确定最佳UTM投影
    """
    # 获取几何体的边界并计算中心点
    bounds = geometry.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    # 计算UTM区域号
    utm_zone = int((center_lon + 180) / 6) + 1
    
    # 判断是北半球还是南半球
    hemisphere = 'north' if center_lat >= 0 else 'south'
    
    # 构建UTM投影的EPSG代码
    if hemisphere == 'north':
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone
    
    return f"EPSG:{epsg_code}"

def gpkg_to_tiff(input_path, output_path, resolution=10):
    """
    将GPKG文件转换为TIFF文件
    
    Parameters:
    - input_path: 输入的GPKG文件路径
    - output_path: 输出的TIFF文件路径
    - resolution: 分辨率（米），默认10米
    """
    
    # 1. 读取GPKG文件
    print("正在读取GPKG文件...")
    gdf = gpd.read_file(input_path)
    
    # 确保原始坐标系存在
    if gdf.crs is None:
        print("警告：GPKG文件没有坐标系信息，假设为WGS84")
        gdf.crs = 'EPSG:4326'
    
    print(f"原始坐标系: {gdf.crs}")
    print(f"几何体数量: {len(gdf)}")
    
    # 2. 确定最佳投影坐标系
    print("\n正在确定最佳投影坐标系...")
    
    # 先转换到WGS84以获取正确的地理坐标
    if gdf.crs != 'EPSG:4326':
        gdf_wgs84 = gdf.to_crs('EPSG:4326')
    else:
        gdf_wgs84 = gdf.copy()
    
    best_crs = find_best_utm_zone(gdf_wgs84)
    print(f"选择的最佳投影: {best_crs}")
    
    # 3. 转换到最佳投影坐标系
    print("\n正在转换坐标系...")
    gdf_projected = gdf.to_crs(best_crs)
    
    # 4. 计算边界和栅格参数
    print("\n正在计算栅格参数...")
    bounds = gdf_projected.total_bounds
    min_x, min_y, max_x, max_y = bounds
    
    # 计算栅格尺寸
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    
    # 调整边界以确保完整覆盖
    max_x = min_x + width * resolution
    max_y = min_y + height * resolution
    
    print(f"栅格尺寸: {width} x {height}")
    print(f"边界: ({min_x:.2f}, {min_y:.2f}) 到 ({max_x:.2f}, {max_y:.2f})")
    
    # 5. 创建变换矩阵
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    # 6. 栅格化
    print("\n正在进行栅格化...")
    
    # 创建空的栅格数组
    raster = np.zeros((height, width), dtype=np.uint8)
    
    # 栅格化几何体（用1填充）
    if len(gdf_projected) > 0:
        # 获取几何体列表
        geometries = [(geom, 1) for geom in gdf_projected.geometry]
        
        # 栅格化
        rasterized = features.rasterize(
            geometries,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        
        raster = rasterized
    
    # 7. 保存为TIFF
    print(f"\n正在保存到: {output_path}")
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=best_crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(raster, 1)
    
    print("\n转换完成！")
    print(f"输出文件: {output_path}")
    print(f"分辨率: {resolution}m")
    print(f"投影坐标系: {best_crs}")
    print(f"栅格尺寸: {width} x {height}")
    
    # 返回一些统计信息
    stats = {
        'output_path': output_path,
        'resolution': resolution,
        'crs': best_crs,
        'width': width,
        'height': height,
        'bounds': (min_x, min_y, max_x, max_y),
        'pixel_count': np.sum(raster > 0),
        'total_pixels': width * height
    }
    
    return stats

# 使用示例
if __name__ == "__main__":
    # 输入和输出文件路径
    input_gpkg = "/mnt/c/Users/Zhengpeng Feng/Downloads/trentino.gpkg"
    output_tiff = "/mnt/c/Users/Zhengpeng Feng/Downloads/trentino_10m.tiff"
    
    # 执行转换
    try:
        stats = gpkg_to_tiff(input_gpkg, output_tiff, resolution=10)
        
        print("n=== 转换统计信息 ===")
        print(f"有效像素数: {stats['pixel_count']:,}")
        print(f"总像素数: {stats['total_pixels']:,}")
        print(f"覆盖率: {stats['pixel_count']/stats['total_pixels']*100:.2f}%")
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()