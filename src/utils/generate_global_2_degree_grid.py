import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from shapely.geometry import box
import pyproj
from tqdm import tqdm
import multiprocessing as mp
import concurrent.futures
import glob
import pandas as pd
import random
from functools import partial
import time

def load_countries_shapefile(shapefile_path, exclude_antarctica=True, buffer_distance=3000):
    """
    加载国家边界shapefile，根据需要过滤掉南极洲，并添加3公里缓冲区
    """
    countries = gpd.read_file(shapefile_path)
    
    # 过滤出大陆区域(排除南极洲)
    if exclude_antarctica:
        # countries = countries[countries['SOVEREIGNT'] != 'Antarctica']
        countries = countries[countries['NAME'] != 'Antarctica']
    
    # 添加缓冲区 - 先转换到投影坐标系统进行缓冲
    # 使用Web Mercator (EPSG:3857) 进行缓冲操作
    countries_proj = countries.to_crs("EPSG:3857")
    countries_buffered = countries_proj.buffer(buffer_distance)
    
    # 创建包含缓冲区的新GeoDataFrame
    countries_with_buffer = gpd.GeoDataFrame(
        countries.drop(columns='geometry'), 
        geometry=countries_buffered,
        crs="EPSG:3857"
    )
    
    # 转回WGS84 (EPSG:4326)
    countries_with_buffer = countries_with_buffer.to_crs("EPSG:4326")
    
    print(f"已添加 {buffer_distance}米 缓冲区至国家边界")
    
    return countries_with_buffer

def generate_global_grid(grid_size=2.0, x_min=-180, x_max=180, y_min=-90, y_max=90):
    """
    生成指定大小的全球网格(以度为单位)
    """
    # 创建网格单元
    grid_cells = []
    
    for x in np.arange(x_min, x_max, grid_size):
        for y in np.arange(y_min, y_max, grid_size):
            # 计算中心坐标
            center_lon = x + grid_size/2
            center_lat = y + grid_size/2
            
            # 为网格单元创建一个矩形框
            cell = box(x, y, x + grid_size, y + grid_size)
            grid_cells.append({
                'geometry': cell,
                'lon_min': x,
                'lon_max': x + grid_size,
                'lat_min': y,
                'lat_max': y + grid_size,
                'center_lon': center_lon,
                'center_lat': center_lat
            })
    
    # 从网格单元创建GeoDataFrame
    grid = gpd.GeoDataFrame(grid_cells, crs="EPSG:4326")
    return grid

def get_utm_zone(longitude, latitude):
    """
    获取给定经纬度的UTM区域
    """
    # 计算UTM区域编号
    if longitude >= 180:
        longitude = longitude - 360
    
    zone_number = int((longitude + 180) / 6) + 1
    
    # 确定是北半球还是南半球
    if latitude >= 0:
        # 北半球
        epsg = 32600 + zone_number
    else:
        # 南半球
        epsg = 32700 + zone_number
    
    return f"EPSG:{epsg}"

def create_grid_raster(grid_cell, countries, output_path, resolution=1000):
    """
    为网格单元创建栅格，其中与陆地相交的区域=1，其他区域=0。
    使用自定义分辨率，默认为1000米，所有TIFF具有相同尺寸。
    """
    # 提取网格单元属性
    lon_min = grid_cell['lon_min']
    lon_max = grid_cell['lon_max']
    lat_min = grid_cell['lat_min']
    lat_max = grid_cell['lat_max']
    center_lon = grid_cell['center_lon']
    center_lat = grid_cell['center_lat']
    
    # 根据网格中心坐标创建输出文件名
    filename = f"grid_{center_lon:.2f}_{center_lat:.2f}.tiff"
    output_file = os.path.join(output_path, filename)
    
    # 如果文件已存在，则跳过
    if os.path.exists(output_file):
        return output_file
    
    # 获取适当的UTM投影
    utm_epsg = get_utm_zone(center_lon, center_lat)
    
    # 为此网格单元创建GeoDataFrame
    grid_gdf = gpd.GeoDataFrame([{
        'geometry': box(lon_min, lat_min, lon_max, lat_max),
    }], crs="EPSG:4326")
    
    # 将国家边界裁剪到此网格单元的范围
    grid_geom = grid_gdf.geometry.iloc[0]
    countries_in_grid = countries[countries.intersects(grid_geom)]
    
    # 投影到UTM
    grid_utm = grid_gdf.to_crs(utm_epsg)
    
    # 计算UTM中的网格边界
    bounds_utm = grid_utm.total_bounds
    xmin, ymin, xmax, ymax = bounds_utm
    
    # 计算宽度和高度，使用自定义分辨率
    width = int(round((xmax - xmin) / resolution))
    height = int(round((ymax - ymin) / resolution))
    
    # 打印实际网格和分辨率信息
    print(f"网格 {filename}: 分辨率={resolution}m, 尺寸={width}x{height}像素")
    
    # 创建变换矩阵
    transform = from_origin(xmin, ymax, resolution, resolution)
    
    # 如果此网格单元中有陆地，栅格化陆地区域
    if len(countries_in_grid) > 0:
        countries_utm = countries_in_grid.to_crs(utm_epsg)
        shapes = [(geom, 1) for geom in countries_utm.geometry]
        raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,  # 确保所有相交的像素都被包括
            dtype=np.uint8
        )
    else:
        # 如果没有陆地，创建全0数组
        raster = np.zeros((height, width), dtype=np.uint8)
    
    # 写入压缩的GeoTIFF
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=utm_epsg,
        transform=transform,
        compress='lzw',      # 使用LZW压缩
        predictor=2,         # 水平差分预测器，提高压缩率
        tiled=True,          # 使用分块存储
        blockxsize=256,      # 优化块大小
        blockysize=256,
        zlevel=9,            # 最高压缩级别
        nodata=0             # 设置无数据值
    ) as dst:
        dst.write(raster, 1)
    
    return output_file

def process_grid_cell(grid_cell, countries, output_path, resolution=1000):
    """处理单个网格单元的函数，用于并行处理"""
    return create_grid_raster(grid_cell, countries, output_path, resolution)

def main():
    # 定义路径
    shapefile_path = "/maps/zf281/btfm4rs/data/global_map_shp/detailed_world_map.shp"
    output_path = "/scratch/zf281/global_map_0.1_degree_tiff"
    
    # 自定义分辨率设置 (单位：米)
    # 可以根据需要修改此值
    resolution = 10  # 默认1000米分辨率
    
    # 如果输出目录不存在，则创建
    os.makedirs(output_path, exist_ok=True)
    
    # 加载国家边界shapefile并添加3公里缓冲区
    print("加载国家边界shapefile并添加1公里缓冲区...")
    countries = load_countries_shapefile(shapefile_path, buffer_distance=1000)
    
    # 生成全球网格
    print("生成全球网格...")
    grid = generate_global_grid(grid_size=0.1)  # 0.1度网格大约是11公里
    
    # 筛选只包含与陆地相交的网格单元
    print("筛选与陆地相交的网格单元...")
    land_grid = []
    for _, grid_row in tqdm(grid.iterrows(), total=len(grid), desc="筛选网格"):
        grid_geom = grid_row.geometry
        if countries.intersects(grid_geom).any():
            land_grid.append({
                'lon_min': grid_row.lon_min,
                'lon_max': grid_row.lon_max,
                'lat_min': grid_row.lat_min,
                'lat_max': grid_row.lat_max,
                'center_lon': grid_row.center_lon,
                'center_lat': grid_row.center_lat
            })
    
    print(f"找到 {len(land_grid)} 个与陆地相交的网格单元")
    print(f"使用分辨率: {resolution}米")
    
    # 使用多进程并行处理创建栅格
    num_cores = mp.cpu_count()
    print(f"使用 {num_cores} 个CPU核心进行并行处理")
    
    # 为每个网格单元创建栅格
    print("为网格单元创建栅格...")
    
    # 创建部分函数以传递固定参数
    process_func = partial(
        process_grid_cell, 
        countries=countries, 
        output_path=output_path, 
        resolution=resolution
    )
    
    # 使用进程池并行处理网格
    with mp.Pool(processes=num_cores) as pool:
        # 使用tqdm显示进度
        list(tqdm(pool.imap(process_func, land_grid), total=len(land_grid), desc="创建网格TIFF"))
    
    print("完成!")

if __name__ == "__main__":
    main()