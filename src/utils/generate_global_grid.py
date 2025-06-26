
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from shapely.geometry import box
from tqdm import tqdm
import multiprocessing as mp
import glob
from functools import partial
import time
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union
from rasterio import features
from rasterio.warp import transform_bounds

def load_countries_shapefile(shapefile_path, exclude_antarctica=True, buffer_distance=3000):
    """
    加载国家边界shapefile，根据需要过滤掉南极洲，并添加缓冲区。
    """
    print(f"正在加载 Shapefile: {shapefile_path}")
    countries = gpd.read_file(shapefile_path)
    
    # 过滤出大陆区域(排除南极洲)
    if exclude_antarctica:
        countries = countries[countries['NAME'] != 'Antarctica']
    
    # 添加缓冲区 - 先转换到投影坐标系统进行缓冲
    # 使用 Web Mercator (EPSG:3857) 进行缓冲操作
    print(f"正在为国家边界添加 {buffer_distance} 米的缓冲区...")
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
    
    print(f"已添加 {buffer_distance}米 缓冲区至国家边界。")
    return countries_with_buffer

def generate_global_grid(grid_size=0.1, x_min=-180, x_max=180, y_min=-90, y_max=90, overlap_degree=0.01):
    """
    生成指定大小的全球网格(以度为单位)，带有重叠区域。
    此函数经过优化，可确保网格坐标的小数点后两位以'5'结尾 (当 grid_size=0.1 时)。
    """
    grid_cells = []
    
    # 计算起始坐标，以确保文件名中的坐标以 .x5 结尾
    # 例如，对于 grid_size=0.1, 坐标将是 -179.95, -179.85, ...
    start_x = x_min + grid_size / 2.0
    start_y = y_min + grid_size / 2.0
    
    # 使用 np.arange 生成坐标数组，以获得更好的浮点数稳定性
    x_coords = np.arange(start_x, x_max, grid_size)
    y_coords = np.arange(start_y, y_max, grid_size)
    
    # 四舍五入以避免浮点精度问题，例如 x.x499999999
    decimals = len(str(grid_size).split('.')[-1]) + 2
    x_coords = np.round(x_coords, decimals)
    y_coords = np.round(y_coords, decimals)

    # 循环生成网格单元
    for x in x_coords:
        for y in y_coords:
            original_lon_min = x
            original_lat_min = y
            original_lon_max = x + grid_size
            original_lat_max = y + grid_size

            center_lon = x + grid_size / 2.0
            center_lat = y + grid_size / 2.0

            # 为网格单元创建一个矩形框，添加重叠区域
            cell = box(
                x - overlap_degree,
                y - overlap_degree,
                original_lon_max + overlap_degree,
                original_lat_max + overlap_degree
            )

            grid_cells.append({
                'geometry': cell,
                'lon_min': x - overlap_degree,
                'lon_max': original_lon_max + overlap_degree,
                'lat_min': y - overlap_degree,
                'lat_max': original_lat_max + overlap_degree,
                'center_lon': center_lon,
                'center_lat': center_lat,
                'original_lon_min': original_lon_min,
                'original_lon_max': original_lon_max,
                'original_lat_min': original_lat_min,
                'original_lat_max': original_lat_max
            })

    if not grid_cells:
        print("警告：没有生成任何网格单元。请检查 grid_size 和范围参数。")
        return gpd.GeoDataFrame([], crs="EPSG:4326")

    grid = gpd.GeoDataFrame(grid_cells, crs="EPSG:4326")
    print(f"生成了 {len(grid)} 个网格单元，每个带有 {overlap_degree} 度的重叠。")

    # 验证生成的网格坐标
    print("\n验证生成的网格坐标是否符合要求 (例如以 .x5 结尾):")
    example_cell = grid_cells[100] # 从一个非边缘的单元格获取例子
    lon_str_check = f"{example_cell['original_lon_min']:.2f}"
    lat_str_check = f"{example_cell['original_lat_min']:.2f}"
    print(f"  - 示例文件名: grid_{lon_str_check}_{lat_str_check}.tiff")
    if lon_str_check.endswith('5') and lat_str_check.endswith('5'):
        print("✓ 初步验证通过: 示例文件名符合要求。")
    else:
        print(f"✗ 初步验证失败: 示例文件名 ({lon_str_check}, {lat_str_check}) 不符合要求。")

    return grid

def check_required_files(land_grid, required_files):
    """
    检查必需的文件是否会被生成。
    """
    print("\n检查必需的文件是否会被生成:")
    
    generated_filenames = set()
    for grid_cell in land_grid:
        orig_lon_min = grid_cell['original_lon_min']
        orig_lat_min = grid_cell['original_lat_min']
        filename = f"grid_{orig_lon_min:.2f}_{orig_lat_min:.2f}.tiff"
        generated_filenames.add(filename)
    
    all_found = True
    for req_file in required_files:
        if req_file in generated_filenames:
            print(f"  ✓ {req_file} - 将会生成 (此网格与陆地相交)")
        else:
            all_found = False
            print(f"  ✗ {req_file} - 不会生成 (原因可能是此网格不与陆地相交)")
    
    if all_found:
        print("所有必需文件都将在陆地区域内生成。")
    else:
        print("部分必需文件未在陆地区域找到，可能位于海洋中。")


def get_utm_zone(longitude, latitude):
    """
    获取给定经纬度的UTM区域。
    """
    if longitude >= 180:
        longitude = longitude - 360
    zone_number = int((longitude + 180) / 6) + 1
    if latitude >= 0:
        return f"EPSG:{32600 + zone_number}"
    else:
        return f"EPSG:{32700 + zone_number}"

def create_grid_raster(grid_cell, countries, output_path, resolution=10):
    """
    为网格单元创建栅格，其中与陆地相交的区域=1，其他区域=0。
    """
    lon_min, lon_max = grid_cell['lon_min'], grid_cell['lon_max']
    lat_min, lat_max = grid_cell['lat_min'], grid_cell['lat_max']
    center_lon, center_lat = grid_cell['center_lon'], grid_cell['center_lat']
    orig_lon_min, orig_lat_min = grid_cell['original_lon_min'], grid_cell['original_lat_min']
    
    # 根据原始网格坐标创建输出文件名
    filename = f"grid_{orig_lon_min:.2f}_{orig_lat_min:.2f}.tiff"
    output_file = os.path.join(output_path, filename)
    
    if os.path.exists(output_file):
        return output_file
    
    utm_epsg = get_utm_zone(center_lon, center_lat)
    grid_gdf = gpd.GeoDataFrame([{'geometry': box(lon_min, lat_min, lon_max, lat_max)}], crs="EPSG:4326")
    
    # 裁剪国家边界到此网格范围，提高效率
    countries_in_grid = gpd.clip(countries, grid_gdf)
    
    grid_utm = grid_gdf.to_crs(utm_epsg)
    bounds_utm = grid_utm.total_bounds
    xmin, ymin, xmax, ymax = bounds_utm
    
    width = int(round((xmax - xmin) / resolution))
    height = int(round((ymax - ymin) / resolution))

    # 确保宽度和高度至少为1
    if width <= 0 or height <= 0:
        # print(f"跳过无效尺寸的栅格: {filename}")
        return None

    transform = from_origin(xmin, ymax, resolution, resolution)
    
    raster = np.zeros((height, width), dtype=np.uint8)
    if not countries_in_grid.empty:
        countries_utm = countries_in_grid.to_crs(utm_epsg)
        shapes = [(geom, 1) for geom in countries_utm.geometry]
        raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

    with rasterio.open(
        output_file, 'w', driver='GTiff',
        height=height, width=width, count=1, dtype=np.uint8,
        crs=utm_epsg, transform=transform, compress='lzw',
        predictor=2, tiled=True, blockxsize=256, blockysize=256,
        zlevel=9, nodata=0
    ) as dst:
        dst.write(raster, 1)
    
    return output_file

def process_grid_cell(grid_cell, countries, output_path, resolution):
    """处理单个网格单元的函数，用于并行处理。"""
    try:
        return create_grid_raster(grid_cell, countries, output_path, resolution)
    except Exception as e:
        print(f"处理网格单元 {grid_cell['original_lon_min']:.2f}, {grid_cell['original_lat_min']:.2f} 时出错: {e}")
        return None

def process_tiff_for_coverage(tiff_file):
    """处理单个TIFF文件以提取覆盖范围多边形。"""
    wgs84_crs = CRS.from_epsg(4326)
    try:
        with rasterio.open(tiff_file) as src:
            if src.crs is None:
                print(f"警告: TIFF 文件 {tiff_file} 缺少 CRS。跳过。")
                return []
            mask = src.read_masks(1) > 0
            if np.any(mask):
                shapes = features.shapes(src.read(1, masked=True), mask=mask, transform=src.transform)
                polygons = [shapely_shape(shape) for shape, val in shapes if val == 1]
                
                # 批量转换坐标
                if src.crs != wgs84_crs:
                    projected_polygons = []
                    for poly in polygons:
                        bounds = poly.bounds
                        bounds_wgs84 = transform_bounds(src.crs, wgs84_crs, *bounds)
                        projected_polygons.append(box(*bounds_wgs84))
                    return projected_polygons
                return polygons
        return []
    except Exception as e:
        print(f"处理TIFF文件 {tiff_file} 时出错: {e}")
        return []

def check_tiff_exists(output_path, land_grid):
    """检查是否所有需要的TIFF文件都已经存在。"""
    missing_grids = []
    existing_files = 0
    for grid_cell in land_grid:
        filename = f"grid_{grid_cell['original_lon_min']:.2f}_{grid_cell['original_lat_min']:.2f}.tiff"
        if os.path.exists(os.path.join(output_path, filename)):
            existing_files += 1
        else:
            missing_grids.append(grid_cell)
    return existing_files, missing_grids

def get_equal_area_crs():
    """获取一个全球等面积投影CRS。"""
    return "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

def visualize_and_check_coverage(shapefile_path, tiff_folder, output_viz_path):
    """可视化检查TIFF覆盖情况并计算面积。"""
    print("\n开始可视化检查...")
    countries = gpd.read_file(shapefile_path)
    countries = countries[countries['NAME'] != 'Antarctica']
    
    tiff_files = glob.glob(os.path.join(tiff_folder, "*.tiff"))
    print(f"找到 {len(tiff_files)} 个TIFF文件用于检查。")
    if not tiff_files:
        print("未找到TIFF文件，无法进行覆盖分析。")
        return None

    with mp.Pool(processes=max(1, mp.cpu_count() // 2)) as pool:
        polygon_lists = list(tqdm(pool.imap(process_tiff_for_coverage, tiff_files), total=len(tiff_files), desc="处理TIFF获取覆盖范围"))
    
    tiff_coverage_polygons = [poly for sublist in polygon_lists for poly in sublist]
    
    if not tiff_coverage_polygons:
        print("警告：从TIFF文件中未能提取任何陆地覆盖多边形。")
        return None
        
    tiff_coverage = gpd.GeoDataFrame(geometry=tiff_coverage_polygons, crs="EPSG:4326")
    print("合并TIFF覆盖区域...")
    merged_coverage = unary_union(tiff_coverage.geometry)
    tiff_coverage_merged = gpd.GeoDataFrame(geometry=[merged_coverage], crs="EPSG:4326")

    print("\n计算面积...")
    area_crs = get_equal_area_crs()
    countries_proj = countries.to_crs(area_crs)
    tiff_coverage_proj = tiff_coverage_merged.to_crs(area_crs)
    
    countries_area = countries_proj.geometry.area.sum() / 1e6
    tiff_area = tiff_coverage_proj.geometry.area.sum() / 1e6
    
    intersection = gpd.overlay(countries_proj, tiff_coverage_proj, how='intersection')
    intersection_area = intersection.geometry.area.sum() / 1e6

    print(f"\n面积分析结果:")
    print(f"  - 原始Shapefile面积: {countries_area:,.2f} 平方公里")
    print(f"  - TIFF覆盖总面积:   {tiff_area:,.2f} 平方公里")
    print(f"  - 相交面积:         {intersection_area:,.2f} 平方公里")
    if countries_area > 0:
      print(f"  - 覆盖率:           {(intersection_area/countries_area)*100:.2f}%")

    print("\n创建可视化图表...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    countries.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)
    tiff_coverage_merged.plot(ax=ax, color='blue', alpha=0.5)
    ax.set_title(f'Shapefile (灰色) 与 TIFF 覆盖范围 (蓝色) 对比\n覆盖率: {(intersection_area/countries_area)*100:.2f}%' if countries_area > 0 else '覆盖率: N/A', fontsize=16)
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    
    plt.tight_layout()
    plt.savefig(output_viz_path, dpi=200, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_viz_path}")

    return {
        'shapefile_area': countries_area, 'tiff_area': tiff_area,
        'intersection_area': intersection_area,
        'coverage_percentage': (intersection_area/countries_area)*100 if countries_area > 0 else 0,
        'num_tiff_files': len(tiff_files)
    }

def main():
    # --- 用户配置 ---
    # 定义路径 (请根据您的环境修改)
    shapefile_path = "/shared/amdgpu/home/avsm2_f4q/code/btfm4rs/data/global_map_shp/detailed_world_map.shp"
    output_path = "/shared/amdgpu/home/avsm2_f4q/code/btfm4rs/data/global_map_0.1_degree_tiff"
    visualization_output = "/shared/amdgpu/home/avsm2_f4q/code/btfm4rs/data/coverage_check.png"

    # 自定义分辨率设置 (单位：米)
    resolution = 10  # 10米分辨率

    # 网格大小（度）
    grid_size = 0.1 # 0.1度的网格，将生成 x.x5 格式的坐标

    # 重叠大小（度）
    overlap_degree = 0.005  # 0.005度约等于0.5公里的重叠

    # 定义必需的文件列表（用户提供的例子）
    required_files = [
        "grid_-1.75_53.75.tiff", "grid_-1.75_54.05.tiff",
        "grid_-1.75_54.15.tiff", "grid_-1.75_54.45.tiff",
        "grid_-1.75_54.55.tiff", "grid_-1.75_54.65.tiff",
        "grid_-1.75_54.75.tiff", "grid_-1.75_55.05.tiff"
    ]
    # --- 配置结束 ---
    
    os.makedirs(output_path, exist_ok=True)
    
    if not os.path.exists(shapefile_path):
        print(f"错误: Shapefile 未找到于 '{shapefile_path}'。请检查路径。")
        return

    countries = load_countries_shapefile(shapefile_path, buffer_distance=3000)
    
    print("\n生成全球网格...")
    grid = generate_global_grid(grid_size=grid_size, overlap_degree=overlap_degree)
    
    print("\n筛选与陆地相交的网格单元...")
    land_grid_indices = countries.sindex.query(grid.geometry, predicate='intersects')
    land_grid_df = grid.iloc[sorted(np.unique(land_grid_indices))]
    land_grid = land_grid_df.to_dict('records')
    print(f"找到 {len(land_grid)} 个与陆地相交的网格单元。")

    # 检查必需的文件是否在陆地区域内
    check_required_files(land_grid, required_files)

    existing_files, missing_grids = check_tiff_exists(output_path, land_grid)
    print(f"\n已存在的TIFF文件: {existing_files}")
    print(f"需要生成的TIFF文件: {len(missing_grids)}")
    
    if missing_grids:
        print(f"\n开始生成 {len(missing_grids)} 个缺失的TIFF文件...")
        print(f"  - 使用分辨率: {resolution}米")
        print(f"  - 使用重叠: {overlap_degree}度")
        
        num_cores = max(1, mp.cpu_count() - 1)
        print(f"  - 使用 {num_cores} 个CPU核心进行并行处理")
        
        process_func = partial(process_grid_cell, countries=countries, output_path=output_path, resolution=resolution)
        
        start_time = time.time()
        with mp.Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap_unordered(process_func, missing_grids), total=len(missing_grids), desc="创建网格TIFF"))
        
        successful_files = [r for r in results if r is not None]
        print(f"\n成功创建 {len(successful_files)} 个TIFF文件。")
        print(f"处理时间: {time.time() - start_time:.2f} 秒。")
    else:
        print("\n所有需要的TIFF文件都已存在，跳过生成步骤。")
    
    stats = visualize_and_check_coverage(shapefile_path, output_path, visualization_output)
    
    if stats:
        print("\n=== 最终统计 ===")
        print(f"TIFF文件总数: {stats['num_tiff_files']}")
        print(f"Shapefile面积: {stats['shapefile_area']:,.2f} km²")
        print(f"TIFF覆盖面积: {stats['tiff_area']:,.2f} km²")
        print(f"交集面积: {stats['intersection_area']:,.2f} km²")
        print(f"覆盖率: {stats['coverage_percentage']:.2f}%")

    print("\n完成!")

if __name__ == "__main__":
    main()
