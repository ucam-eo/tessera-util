import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import os

# 读取shapefile文件
base_dir = "data/downstream/Ethiopian_Crop_Type_2020_dataset"
shapefile_path = os.path.join(base_dir, "EthCT2020.shp")
gdf = gpd.read_file(shapefile_path)

# 打印所有属性名
print("所有属性名:")
for column in gdf.columns:
    print(column)

# 打印散点总数
total_points = len(gdf)
print(f"\n散点总数: {total_points}")

# 提取c_class属性的所有唯一值并统计每个类别的点数
if "c_class" in gdf.columns:
    # 获取c_class的唯一值并统计数量
    class_counts = gdf["c_class"].value_counts()
    
    # 获取前4个点最多的类别
    top_4_classes = class_counts.nlargest(4).index.tolist()
    print(f"\n点数最多的前4个类别: {top_4_classes}")
    
    # 提取这4个类别对应的点
    filtered_gdf = gdf[gdf["c_class"].isin(top_4_classes)]
    print(f"提取出的点数: {len(filtered_gdf)}")
    
    # 保存为新的shapefile
    new_shapefile_name = "EthCT2020_top4_classes.shp"
    new_shapefile_path = os.path.join(base_dir, new_shapefile_name)
    filtered_gdf.to_file(new_shapefile_path)
    print(f"已保存新的shapefile: {new_shapefile_path}")
    
    # 创建对应区域的tiff文件
    # 获取区域边界
    bounds = filtered_gdf.total_bounds  # (minx, miny, maxx, maxy)
    print(f"数据边界: {bounds}")
    
    # 计算适当的分辨率，使得栅格大小合理（目标大小约2000x2000像素）
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]
    target_pixels = 2000
    resolution = max(x_range, y_range) / target_pixels
    
    # 计算栅格大小
    width = int(x_range / resolution)
    height = int(y_range / resolution)
    
    # 打印栅格信息
    print(f"计算的分辨率: {resolution}")
    print(f"栅格尺寸: {width} x {height} 像素")
    
    # 检查栅格大小，防止内存溢出
    max_pixels = 1e8  # 最大允许的像素数（约100MB，uint8类型）
    if width * height > max_pixels:
        print(f"警告: 栅格尺寸过大! 调整分辨率...")
        scale_factor = np.sqrt((width * height) / max_pixels)
        width = int(width / scale_factor)
        height = int(height / scale_factor)
        resolution = max(x_range / width, y_range / height)
        print(f"调整后的分辨率: {resolution}")
        print(f"调整后的栅格尺寸: {width} x {height} 像素")
    
    # 创建仿射变换
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    # 创建所有值为1的栅格数据
    raster_data = np.ones((height, width), dtype=np.uint8)
    
    # 获取原始shapefile的坐标系
    crs = filtered_gdf.crs
    
    # 保存为tiff文件
    new_tiff_name = "EthCT2020_top4_classes.tiff"
    new_tiff_path = os.path.join(base_dir, new_tiff_name)
    
    with rasterio.open(
        new_tiff_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=raster_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raster_data, 1)
    
    print(f"已保存tiff文件: {new_tiff_path}")
else:
    print("\n在shapefile中未找到'c_class'属性")