#!/usr/bin/env python3
import os
import fiona
import rasterio
import logging
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.crs import CRS
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform, unary_union
from pyproj import Transformer

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def determine_utm_zone(lon, lat):
    """
    根据经度和纬度确定最佳UTM区域
    
    Args:
        lon (float): 经度（度）
        lat (float): 纬度（度）
        
    Returns:
        tuple: (epsg代码, 区域编号, 是否北半球)
    """
    # UTM区域宽度为6度
    zone_number = int((lon + 180) / 6) + 1
    
    # 挪威特殊情况
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone_number = 32
        
    # 斯瓦尔巴特特殊情况
    if 72 <= lat < 84:
        if 0 <= lon < 9:
            zone_number = 31
        elif 9 <= lon < 21:
            zone_number = 33
        elif 21 <= lon < 33:
            zone_number = 35
        elif 33 <= lon < 42:
            zone_number = 37
            
    is_northern = lat >= 0
    
    # EPSG代码：北半球EPSG:326xx，南半球EPSG:327xx
    epsg_code = 32600 + zone_number if is_northern else 32700 + zone_number
    
    return epsg_code, zone_number, is_northern

def determine_best_utm_crs(geometries, src_crs):
    """
    基于几何体集合的质心确定最佳UTM坐标参考系统
    
    Args:
        geometries (list): 源CRS下的几何体列表
        src_crs (CRS): 源坐标参考系统
        
    Returns:
        CRS: 目标UTM坐标参考系统
    """
    logger.info("开始确定最佳UTM区域...")
    
    # 如果不是WGS84 (EPSG:4326)，则先转换
    if src_crs.to_epsg() != 4326 and src_crs != CRS.from_epsg(4326):
        logger.info(f"源CRS不是WGS84，正在转换到WGS84...")
        wgs84_crs = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(src_crs, wgs84_crs, always_xy=True).transform
        # 将所有几何体转换为WGS84
        wgs84_geoms = [shp_transform(transformer, shape(geom)) for geom in geometries]
        logger.info(f"完成WGS84转换")
    else:
        logger.info(f"源CRS已经是WGS84，无需转换")
        wgs84_geoms = [shape(geom) for geom in geometries]
    
    # 创建所有几何体的并集并获取其质心
    union_geom = unary_union(wgs84_geoms)
    centroid = union_geom.centroid
    cent_lon, cent_lat = centroid.x, centroid.y
    
    # 基于质心确定UTM区域
    epsg_code, zone_number, is_northern = determine_utm_zone(cent_lon, cent_lat)
    
    hemisphere = "北半球" if is_northern else "南半球"
    logger.info(f"数据质心位置: {cent_lon:.6f}°E, {cent_lat:.6f}°N")
    logger.info(f"选择的UTM区域: {zone_number}{hemisphere[0]} (EPSG:{epsg_code})")
    
    return CRS.from_epsg(epsg_code)

def shp_to_tiff(shp_path, tiff_path=None, pixel_size=100, force_crs=None):
    """
    将shapefile转换为TIFF栅格
    
    Args:
        shp_path (str): 输入shapefile路径
        tiff_path (str, optional): 输出TIFF路径。如果为None，将从shp_path派生
        pixel_size (float, optional): 输出像素大小（米）。默认为100
        force_crs (CRS, optional): 强制指定目标CRS。如果为None，将自动确定最佳UTM区域
        
    Returns:
        tuple: (主TIFF文件路径, 凸包TIFF文件路径)
    """
    # 如果未提供输出路径，则设置默认路径
    if tiff_path is None:
        tiff_path = os.path.splitext(shp_path)[0] + '.tiff'
    
    logger.info(f"开始转换shapefile: {shp_path}")
    logger.info(f"输出TIFF将保存为: {tiff_path}")
    logger.info(f"使用像素大小: {pixel_size}米")
    
    # 打开shapefile并读取几何体
    with fiona.open(shp_path, 'r') as src:
        # 获取shapefile的基本信息
        num_features = len(src)
        src_driver = src.driver
        src_schema = src.schema
        
        logger.info(f"Shapefile信息:")
        logger.info(f"  - 要素数量: {num_features}")
        logger.info(f"  - 驱动: {src_driver}")
        logger.info(f"  - 模式: {src_schema}")
        
        # 读取所有几何体
        geometries = [feature['geometry'] for feature in src]
        logger.info(f"从shapefile中读取了{len(geometries)}个几何体")
        
        # 获取源CRS，如果未定义则默认为EPSG:4326
        if src.crs:
            src_crs = CRS(src.crs)
            logger.info(f"源CRS: {src_crs.to_string()}")
        else:
            src_crs = CRS.from_epsg(4326)
            logger.warning(f"源CRS未定义，默认使用EPSG:4326 (WGS84)")
    
    # 确定目标CRS（如果未强制指定）
    if force_crs:
        target_crs = force_crs
        logger.info(f"使用强制指定的目标CRS: {target_crs.to_string()}")
    else:
        target_crs = determine_best_utm_crs(geometries, src_crs)
        logger.info(f"自动选择的目标CRS: {target_crs.to_string()}")
    
    # 构造从源CRS到目标CRS的转换器
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True).transform
    logger.info("正在将几何体重投影到目标CRS...")
    
    # 将几何体重投影到目标CRS
    try:
        reprojected_geoms = [mapping(shp_transform(transformer, shape(geom))) for geom in geometries]
        logger.info(f"成功重投影{len(reprojected_geoms)}个几何体")
    except Exception as e:
        logger.error(f"重投影几何体时出错: {str(e)}")
        raise
    
    # 计算重投影几何体的并集和凸包
    try:
        union_geom = unary_union([shape(geom) for geom in reprojected_geoms])
        convex_hull = union_geom.convex_hull
        logger.info("创建了几何体的并集和凸包")
    except Exception as e:
        logger.error(f"创建并集或凸包时出错: {str(e)}")
        raise
    
    # 获取凸包的边界
    minx, miny, maxx, maxy = convex_hull.bounds
    logger.info(f"凸包边界: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")
    
    # 计算输出栅格的尺寸
    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))
    
    if width <= 0 or height <= 0:
        error_msg = f"无效的栅格尺寸: width={width}, height={height}。请检查像素大小。"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"输出栅格尺寸: 宽={width}, 高={height}像素")
    
    # 创建仿射变换矩阵
    transform_affine = from_origin(minx, maxy, pixel_size, pixel_size)
    logger.info(f"仿射变换矩阵: {transform_affine}")
    
    # 栅格化几何体
    logger.info("正在栅格化几何体...")
    try:
        raster = rasterize(
            reprojected_geoms,
            out_shape=(height, width),
            transform=transform_affine,
            fill=0,
            default_value=255,
            dtype='uint8'
        )
        logger.info(f"栅格化完成。栅格形状: {raster.shape}")
    except Exception as e:
        logger.error(f"栅格化过程中出错: {str(e)}")
        raise
    
    # 写入TIFF文件
    logger.info(f"正在将栅格写入TIFF文件: {tiff_path}")
    try:
        with rasterio.open(
            tiff_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs=target_crs,
            transform=transform_affine,
        ) as dst:
            dst.write(raster, 1)
        logger.info("成功写入TIFF文件")
    except Exception as e:
        logger.error(f"写入TIFF文件时出错: {str(e)}")
        raise
    
    # 创建凸包TIFF
    hull_tiff_path = os.path.splitext(tiff_path)[0] + '_convex_hull.tiff'
    logger.info(f"正在创建凸包TIFF: {hull_tiff_path}")
    
    try:
        # 将凸包转换为GeoJSON兼容格式
        hull_geometry = mapping(convex_hull)
        
        # 栅格化凸包
        hull_raster = rasterize(
            [(hull_geometry, 255)],
            out_shape=(height, width),
            transform=transform_affine,
            fill=0,
            dtype='uint8'
        )
        
        # 将凸包写入TIFF
        with rasterio.open(
            hull_tiff_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=hull_raster.dtype,
            crs=target_crs,
            transform=transform_affine,
        ) as dst:
            dst.write(hull_raster, 1)
        logger.info("成功写入凸包TIFF文件")
    except Exception as e:
        logger.error(f"创建凸包TIFF时出错: {str(e)}")
        raise
    
    return tiff_path, hull_tiff_path

def main():
    """
    主函数，运行转换过程
    """
    # 输入shapefile路径
    shp_path = '/scratch/zf281/downstream_dataset/jovana/tile_1/tile_10x10_extent.shp'
    
    # 调用转换函数
    try:
        tiff_path, hull_tiff_path = shp_to_tiff(shp_path, pixel_size=10, force_crs=None)
        logger.info(f"转换完成!")
        logger.info(f"TIFF文件保存于: {tiff_path}")
        logger.info(f"凸包TIFF保存于: {hull_tiff_path}")
    except Exception as e:
        logger.error(f"转换失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()