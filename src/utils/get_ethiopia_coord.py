import os
import glob
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point
import json
from datetime import datetime

# 路径设置
data_raw_dir = "/scratch/zf281/ethiopia/data_raw"
data_processed_dir = "/scratch/zf281/ethiopia/data_processed"
shapefile_path = "/maps/zf281/btfm4rs/data/downstream/ethiopia/shp/EthCT2020_top4_classes.shp"
output_dir = "/maps/zf281/btfm4rs/data/downstream/ethiopia"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取shapefile
print("读取shapefile...")
polygons_gdf = gpd.read_file(shapefile_path)
print(f"Shapefile中的对象总数: {len(polygons_gdf)}")
print(f"几何类型: {polygons_gdf.geometry.iloc[0].geom_type}")

# 获取所有MGRS瓦片列表
mgrs_tiles = [os.path.basename(d) for d in glob.glob(os.path.join(data_raw_dir, "*")) if os.path.isdir(d)]
print(f"找到{len(mgrs_tiles)}个MGRS瓦片")

# 建立MGRS瓦片到TIFF文件的映射，以加快访问速度
print("创建MGRS到TIFF的映射...")
mgrs_to_tiff = {}
for mgrs_tile in mgrs_tiles:
    tiff_files = glob.glob(os.path.join(data_raw_dir, mgrs_tile, "red", "*.tiff"))
    if tiff_files:
        mgrs_to_tiff[mgrs_tile] = tiff_files[0]

print(f"为{len(mgrs_to_tiff)}个MGRS瓦片找到了TIFF文件")

# 用于存储结果的字典
results = []

# 处理每个多边形
print("处理多边形坐标...")
for idx, polygon in tqdm(polygons_gdf.iterrows(), total=len(polygons_gdf)):
    # 使用多边形的质心（centroid）作为其代表点
    centroid = polygon.geometry.centroid
    point_lon, point_lat = centroid.x, centroid.y
    
    polygon_results = {
        'polygon_id': idx,
        'centroid_latitude': point_lat,
        'centroid_longitude': point_lon,
        'locations': []
    }
    
    # 添加shapefile中的任何其他属性
    for col in polygons_gdf.columns:
        if col != 'geometry':
            # 转换任何不是JSON可序列化的值
            if isinstance(polygon[col], (pd.Timestamp, datetime)):
                polygon_results[col] = polygon[col].isoformat()
            else:
                polygon_results[col] = polygon[col]
    
    # 检查每个MGRS瓦片
    for mgrs_tile, tiff_file in mgrs_to_tiff.items():
        try:
            # 打开TIFF文件获取地理变换信息
            with rasterio.open(tiff_file) as src:
                # 创建一个与shapefile具有相同CRS的点的GeoDataFrame
                point_gdf = gpd.GeoDataFrame(
                    geometry=[Point(point_lon, point_lat)],
                    crs=polygons_gdf.crs
                )
                
                # 如果需要，将点重投影到栅格的CRS
                if point_gdf.crs != src.crs:
                    point_gdf = point_gdf.to_crs(src.crs)
                
                # 获取重投影点坐标
                reprojected_point = point_gdf.geometry.iloc[0]
                point_x, point_y = reprojected_point.x, reprojected_point.y
                
                # 将投影坐标转换为像素坐标
                # src.index返回(row, col) - 该点落在的像素索引
                row, col = src.index(point_x, point_y)
                
                # 检查点是否在栅格维度范围内
                if (0 <= col < src.width) and (0 <= row < src.height):
                    polygon_results['locations'].append({
                        'mgrs_tile': mgrs_tile,
                        'x': int(col),  # x坐标（numpy数组的列索引）
                        'y': int(row)   # y坐标（numpy数组的行索引）
                    })
        except Exception as e:
            # 可以注释掉这一行来减少输出信息量
            # print(f"处理多边形{idx}的瓦片{mgrs_tile}时出错: {e}")
            pass
    
    results.append(polygon_results)

# 转换结果为DataFrame以便于查看
print("\n生成结果摘要...")
results_summary = []
for result in results:
    entry = {
        'polygon_id': result['polygon_id'],
        'centroid_latitude': result['centroid_latitude'],
        'centroid_longitude': result['centroid_longitude']
    }
    
    # 添加shapefile中的其他列
    for key, value in result.items():
        if key not in ['polygon_id', 'centroid_latitude', 'centroid_longitude', 'locations']:
            entry[key] = value
            
    entry['found_locations'] = len(result['locations'])
    
    if result['locations']:
        locations_str = []
        for loc in result['locations']:
            locations_str.append(f"{loc['mgrs_tile']}:({loc['x']},{loc['y']})")
        entry['mgrs_locations'] = "; ".join(locations_str)
    else:
        entry['mgrs_locations'] = "未找到位置"
    
    results_summary.append(entry)

summary_df = pd.DataFrame(results_summary)
output_file = os.path.join(output_dir, "polygons_mgrs_locations.csv")
summary_df.to_csv(output_file, index=False)
print(f"结果已保存到 {output_file}")

# 解决JSON序列化问题 - 自定义JSON编码器处理特殊类型
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)

# 保存完整的结果
try:
    full_output_file = os.path.join(output_dir, "polygons_mgrs_locations_full.json")
    with open(full_output_file, 'w') as f:
        json.dump(results, f, cls=CustomJSONEncoder, indent=2)
    print(f"完整结果已保存到 {full_output_file}")
except Exception as e:
    print(f"保存JSON时出错: {e}")
    print("跳过JSON保存，继续生成统计报告")

# 输出统计信息
total_polygons = len(summary_df)
matched_df = summary_df[summary_df['found_locations'] > 0]
unmatched_df = summary_df[summary_df['found_locations'] == 0]
polygons_found = len(matched_df)

print("\n==== 匹配统计摘要 ====")
print(f"处理的多边形总数: {total_polygons}")
print(f"匹配到MGRS瓦片的多边形数: {polygons_found} ({polygons_found/total_polygons*100:.2f}%)")
print(f"未找到匹配的多边形数: {total_polygons - polygons_found} ({(total_polygons - polygons_found)/total_polygons*100:.2f}%)")

# 按c_class统计匹配情况
if 'c_class' in summary_df.columns:
    print("\n==== 按作物类型统计匹配情况 ====")
    
    # 获取所有唯一的c_class值
    all_classes = summary_df['c_class'].unique()
    
    print("\n匹配成功的分布:")
    class_counts = matched_df['c_class'].value_counts().sort_index()
    
    for crop_class in all_classes:
        count = class_counts.get(crop_class, 0)
        total_class_count = summary_df[summary_df['c_class'] == crop_class].shape[0]
        match_rate = count / total_class_count * 100 if total_class_count > 0 else 0
        print(f"  {crop_class}: {count}/{total_class_count} ({match_rate:.2f}%)")
    
    print("\n未匹配的分布:")
    unmatched_counts = unmatched_df['c_class'].value_counts().sort_index()
    
    for crop_class in all_classes:
        count = unmatched_counts.get(crop_class, 0)
        total_class_count = summary_df[summary_df['c_class'] == crop_class].shape[0]
        unmatch_rate = count / total_class_count * 100 if total_class_count > 0 else 0
        print(f"  {crop_class}: {count}/{total_class_count} ({unmatch_rate:.2f}%)")

# 打印多边形在每个MGRS瓦片中的分布
print("\n==== 匹配到各MGRS瓦片的多边形分布 ====")
tile_counts = {}

for result in results:
    if result['locations']:
        for loc in result['locations']:
            tile = loc['mgrs_tile']
            if tile not in tile_counts:
                tile_counts[tile] = 0
            tile_counts[tile] += 1

for tile, count in sorted(tile_counts.items()):
    print(f"  {tile}: {count} 个多边形")

# 输出前几个匹配和未匹配结果作为示例
print("\n匹配成功的示例 (前5个):")
for _, row in matched_df.head(5).iterrows():
    print(f"多边形ID: {row['polygon_id']}, 类型: {row.get('c_class', 'N/A')}")
    print(f"  质心坐标: ({row['centroid_longitude']:.6f}, {row['centroid_latitude']:.6f})")
    print(f"  位置: {row['mgrs_locations']}")

print("\n未匹配的示例 (前5个):")
for _, row in unmatched_df.head(5).iterrows():
    print(f"多边形ID: {row['polygon_id']}, 类型: {row.get('c_class', 'N/A')}")
    print(f"  质心坐标: ({row['centroid_longitude']:.6f}, {row['centroid_latitude']:.6f})")