# #!/usr/bin/env python3
# import os
# import fiona
# import rasterio
# from rasterio.features import rasterize
# from rasterio.transform import from_origin
# from rasterio.crs import CRS

# # 输入 shapefile 文件路径
# shp_path = '/mnt/e/Codes/btfm4rs/data/Jovana/SEKI ROI/seki_convex_hull.shp'
# # 输出 TIFF 文件路径，与 shp 同目录，文件名相同，扩展名改为 .tiff
# tiff_path = os.path.splitext(shp_path)[0] + '.tiff'

# with fiona.open(shp_path, 'r') as src:
#     # 读取所有几何体
#     geometries = [feature['geometry'] for feature in src]
#     # 获取空间参考信息，如果为空，则默认设置为 EPSG:4326
#     crs = src.crs if src.crs else CRS.from_epsg(4326)
#     # 获取图层边界 [xmin, ymin, xmax, ymax]
#     bounds = src.bounds

# x_min, y_min, x_max, y_max = bounds # 经度在前，纬度在后
# print("Bounds:", bounds)

# # 针对经纬度数据，建议选择较小的像元大小（单位：度），例如 0.001 度
# pixel_size = 0.001

# # 计算栅格的行列数
# width = int((x_max - x_min) / pixel_size)
# height = int((y_max - y_min) / pixel_size)

# if width <= 0 or height <= 0:
#     raise ValueError(f"计算得到的宽度或高度不合法：width={width}, height={height}。请检查像元大小设置。")

# print(f"Output raster size: width={width}, height={height}")

# # 构造仿射变换矩阵，from_origin 的参数为左上角坐标（xmin, ymax）
# transform = from_origin(x_min, y_max, pixel_size, pixel_size)

# # 利用 rasterize 将矢量数据转换为栅格
# # burn_value 为烧写的数值，此处设为 255，其余区域填充为 0
# raster = rasterize(
#     geometries,
#     out_shape=(height, width),
#     transform=transform,
#     fill=0,
#     default_value=255,
#     dtype='uint8'
# )

# # 写入 TIFF 文件，同时包含投影信息（crs）和地理变换信息
# with rasterio.open(
#     tiff_path,
#     'w',
#     driver='GTiff',
#     height=height,
#     width=width,
#     count=1,
#     dtype=raster.dtype,
#     crs=crs,
#     transform=transform,
# ) as dst:
#     dst.write(raster, 1)

# print("转换成功，tiff 文件保存于：", tiff_path)

#!/usr/bin/env python3
import os
import fiona
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.crs import CRS
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform, unary_union
from pyproj import Transformer

# 输入 shapefile 文件路径
shp_path = '/mnt/e/Codes/btfm4rs/data/Yihang/WythamWoods/perimeter poly with clearings_region.shp'
# 输出 TIFF 文件路径，与 shp 同目录，文件名相同，扩展名改为 .tiff
tiff_path = os.path.splitext(shp_path)[0] + '.tiff'

# 设定目标 CRS 为 UTM，例如 EPSG:32650（WGS84 / UTM zone 50N）
target_crs = CRS.from_epsg(32650)

# 打开 shapefile
with fiona.open(shp_path, 'r') as src:
    # 读取所有几何体
    geometries = [feature['geometry'] for feature in src]
    # 获取输入数据的 CRS，如果未定义，则默认使用 EPSG:4326
    src_crs = CRS(src.crs) if src.crs else CRS.from_epsg(4326)

# 构造从源 CRS 到目标 CRS 的转换器（注意 always_xy=True 确保先 x 后 y）
transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True).transform

# 将几何体重投影到目标 CRS
reprojected_geoms = [mapping(shp_transform(transformer, shape(geom))) for geom in geometries]

# 计算所有重投影后几何体的整体边界
union_geom = unary_union([shape(geom) for geom in reprojected_geoms])
minx, miny, maxx, maxy = union_geom.bounds
print("重投影后的边界：", (minx, miny, maxx, maxy))

# 设置输出分辨率为 10 米
pixel_size = 10

# 计算输出栅格的宽度和高度（注意单位为目标 CRS 下的米）
width = int((maxx - minx) / pixel_size)
height = int((maxy - miny) / pixel_size)
if width <= 0 or height <= 0:
    raise ValueError(f"计算得到的宽度或高度不合法：width={width}, height={height}。请检查像元大小设置。")
print(f"输出栅格尺寸：宽度={width}, 高度={height}")

# 构造仿射变换矩阵，from_origin 的参数为左上角坐标（minx, maxy）
transform_affine = from_origin(minx, maxy, pixel_size, pixel_size)
print("仿射变换矩阵：", transform_affine)
# 说明：在 transform 中，前两个参数为左上角的 x 和 y 坐标，
# 后两个参数分别为 x 分辨率和 y 分辨率（通常 y 分辨率为负数表示从上到下递减）。

# 利用 rasterize 将矢量数据转换为栅格，烧写值设为 255，其它区域填充 0
raster = rasterize(
    reprojected_geoms,
    out_shape=(height, width),
    transform=transform_affine,
    fill=0,
    default_value=255,
    dtype='uint8'
)

# 将栅格数据写入 TIFF 文件，同时写入投影和仿射变换信息
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

print("转换成功，tiff 文件保存于：", tiff_path)