import os
import glob
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon  # 重命名以避免冲突
from matplotlib.colors import to_rgba
from tqdm import tqdm
from shapely.geometry import box, Polygon, MultiPolygon

# 路径设置
# tiff_dir = '/scratch/zf281/global_s2_tiff_minimized/'
tiff_dir = '/scratch/zf281/global_map_1_degree_tiff/'
# world_map_path = '/maps/zf281/btfm4rs/data/global_map_shp/ne_110m_admin_0_countries.shp'
world_map_path = '/maps/zf281/btfm4rs/data/global_map_shp/detailed_world_map.shp'
output_path = 'mgrs_tile_coverage_non_antarctica.png'

# 获取所有TIFF文件
tiff_files = glob.glob(os.path.join(tiff_dir, '*.tiff'))

#DEBUG
# tiff_files = tiff_files[:2000]  # 仅处理前10个文件以进行调试

total_tiff_count = len(tiff_files)

# 读取世界地图
print("加载世界地图...")
world = gpd.read_file(world_map_path)

print(f"世界地图使用的坐标系统: {world.crs}")

# 排除南极地区
print("排除南极地区...")
# non_antarctica_world = world[world['SOVEREIGNT'] != 'Antarctica']
non_antarctica_world = world[world['NAME'] != 'Antarctica']
print(f"排除南极后，世界地图包含 {len(non_antarctica_world)} 个区域")

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(20, 10))

# 绘制非南极地区的世界地图作为底图
non_antarctica_world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)

# 定义颜色
border_color = 'red'
fill_color = to_rgba('gray', alpha=0.4)  # 40%透明度的灰色

# 存储所有图块几何形状和名称的列表
tile_geometries = []
tile_names = []
processed_tiffs = 0
intersecting_tiffs = 0
invalid_extents = 0  # 计数无效边界的TIFF

# 创建非南极大陆的联合几何形状以加速相交测试
non_antarctica_geometry = non_antarctica_world.unary_union

# 处理每个TIFF文件
print(f"开始处理{total_tiff_count}个TIFF文件...")
for tiff_file in tqdm(tiff_files, desc="处理TIFF文件"):
    try:
        with rasterio.open(tiff_file) as src:
            processed_tiffs += 1
            # 获取TIFF的边界
            bounds = src.bounds
            
            # 获取TIFF和世界地图的CRS
            src_crs = src.crs
            world_crs = world.crs
            
            # 如果CRS不同，则转换边界
            if src_crs != world_crs:
                try:
                    # 从源CRS转换到世界地图CRS
                    left, bottom, right, top = transform_bounds(src_crs, world_crs, 
                                                              bounds.left, bounds.bottom, 
                                                              bounds.right, bounds.top)
                except Exception as e:
                    print(f"警告: 无法转换{tiff_file}的CRS: {e}. 使用原始边界.")
                    left, bottom, right, top = bounds
            else:
                left, bottom, right, top = bounds
            
            # 从文件名中提取MGRS ID
            mgrs_id = os.path.splitext(os.path.basename(tiff_file))[0]
            
            # 检测并处理跨越国际日期变更线的情况
            # 在创建几何体或绘制前检查边界的有效性
            is_valid_extent = True
            
            # 检查经度跨度是否异常大（可能表示跨越日期变更线）
            if right - left > 180:
                print(f"警告: MGRS {mgrs_id} 的经度跨度异常大 ({left:.2f} 到 {right:.2f})，可能跨越日期变更线")
                # 做出决定：这种情况下我们认为这是无效的几何体
                is_valid_extent = False
                invalid_extents += 1
            
            # 检查经度是否在合理范围内 (-180到180)
            if left < -180 or left > 180 or right < -180 or right > 180:
                print(f"警告: MGRS {mgrs_id} 的经度超出正常范围 ({left:.2f} 到 {right:.2f})")
                is_valid_extent = False
                invalid_extents += 1
            
            # 如果边界有效，进行正常处理
            if is_valid_extent:
                # 处理跨越日期变更线的特殊情况
                if left > right:  # 这表明可能跨越了日期变更线
                    print(f"检测到 MGRS {mgrs_id} 跨越日期变更线 ({left:.2f} 到 {right:.2f})")
                    
                    # 创建两个几何体：一个从左边界到180度，另一个从-180度到右边界
                    left_box = box(left, bottom, 180, top)
                    right_box = box(-180, bottom, right, top)
                    
                    # 使用两个几何体的并集
                    tile_geom = left_box.union(right_box)
                    
                    # 检查是否与非南极大陆相交
                    if tile_geom.intersects(non_antarctica_geometry):
                        intersecting_tiffs += 1
                        tile_geometries.append(tile_geom)
                        tile_names.append(mgrs_id)
                        
                        # 绘制左半部分
                        left_polygon = MPLPolygon([
                            (left, bottom),
                            (left, top),
                            (180, top),
                            (180, bottom)
                        ], edgecolor=border_color, facecolor=fill_color, linewidth=1)
                        ax.add_patch(left_polygon)
                        
                        # 绘制右半部分
                        right_polygon = MPLPolygon([
                            (-180, bottom),
                            (-180, top),
                            (right, top),
                            (right, bottom)
                        ], edgecolor=border_color, facecolor=fill_color, linewidth=1)
                        ax.add_patch(right_polygon)
                else:
                    # 标准情况下创建单一几何体
                    tile_geom = box(left, bottom, right, top)
                    
                    # 检查是否与非南极大陆相交
                    if tile_geom.intersects(non_antarctica_geometry):
                        intersecting_tiffs += 1
                        tile_geometries.append(tile_geom)
                        tile_names.append(mgrs_id)
                        
                        # 创建用于绘图的多边形
                        polygon = MPLPolygon([
                            (left, bottom),
                            (left, top),
                            (right, top),
                            (right, bottom)
                        ], edgecolor=border_color, facecolor=fill_color, linewidth=1)
                        
                        # 将多边形添加到图中
                        ax.add_patch(polygon)
            
    except Exception as e:
        print(f"处理{tiff_file}时出错: {e}")

# 计算与非南极大陆相交的TIFF比例
intersection_ratio = (intersecting_tiffs / total_tiff_count) * 100 if total_tiff_count > 0 else 0
# 定义添加的变量
invalid_extents = 0  # 这将在处理过程中计数
dateline_crossing = sum(1 for geom in tile_geometries if isinstance(geom, MultiPolygon))

print(f"总共有 {total_tiff_count} 个TIFF文件")
print(f"成功处理了 {processed_tiffs} 个TIFF文件")
print(f"与非南极大陆相交的TIFF文件有 {intersecting_tiffs} 个")
print(f"相交比例: {intersection_ratio:.2f}%")
print(f"无效边界的TIFF文件: {invalid_extents} 个")
print(f"跨越日期变更线的TIFF文件: {dateline_crossing} 个")

# 从相交的图块几何形状创建GeoDataFrame
if tile_geometries:
    tiles_gdf = gpd.GeoDataFrame({'mgrs_id': tile_names, 'geometry': tile_geometries}, crs=world_crs)
    
    # 设置地图范围以显示所有图块
    bounds = tiles_gdf.total_bounds
    margin = max((bounds[2] - bounds[0]), (bounds[3] - bounds[1])) * 0.1
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
else:
    print("没有找到与非南极大陆相交的TIFF文件")

# 设置标题和坐标轴标签
mgrs_count = len(tile_geometries)
ax.set_title(f'MGRS Tile Coverage (Non-Antarctica) - {mgrs_count} tiles, {intersection_ratio:.2f}% of total', fontsize=16)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# 确保图形适当裁剪并具有良好的布局
plt.tight_layout()

# 保存图形为PNG
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"地图已保存到 {output_path}")

# 关闭图形以释放内存
plt.close(fig)