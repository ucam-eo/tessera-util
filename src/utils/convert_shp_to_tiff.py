#!/usr/bin/env python3
import os
import fiona
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.crs import CRS

# 输入 shapefile 文件路径
shp_path = 'data/Jovana/Sierra ROI/sierras_convex_hull.shp'
# 输出 TIFF 文件路径，与 shp 同目录，文件名相同，扩展名改为 .tiff
tiff_path = os.path.splitext(shp_path)[0] + '.tiff'

with fiona.open(shp_path, 'r') as src:
    # 读取所有几何体
    geometries = [feature['geometry'] for feature in src]
    # 获取空间参考信息，如果为空，则默认设置为 EPSG:4326
    crs = src.crs if src.crs else CRS.from_epsg(4326)
    # 获取图层边界 [xmin, ymin, xmax, ymax]
    bounds = src.bounds

x_min, y_min, x_max, y_max = bounds
print("Bounds:", bounds)

# 针对经纬度数据，建议选择较小的像元大小（单位：度），例如 0.001 度
pixel_size = 0.001

# 计算栅格的行列数
width = int((x_max - x_min) / pixel_size)
height = int((y_max - y_min) / pixel_size)

if width <= 0 or height <= 0:
    raise ValueError(f"计算得到的宽度或高度不合法：width={width}, height={height}。请检查像元大小设置。")

print(f"Output raster size: width={width}, height={height}")

# 构造仿射变换矩阵，from_origin 的参数为左上角坐标（xmin, ymax）
transform = from_origin(x_min, y_max, pixel_size, pixel_size)

# 利用 rasterize 将矢量数据转换为栅格
# burn_value 为烧写的数值，此处设为 255，其余区域填充为 0
raster = rasterize(
    geometries,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    default_value=255,
    dtype='uint8'
)

# 写入 TIFF 文件，同时包含投影信息（crs）和地理变换信息
with rasterio.open(
    tiff_path,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=raster.dtype,
    crs=crs,
    transform=transform,
) as dst:
    dst.write(raster, 1)

print("转换成功，tiff 文件保存于：", tiff_path)
