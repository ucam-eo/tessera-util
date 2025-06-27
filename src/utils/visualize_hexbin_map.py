#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局 hexbin 地图绘制脚本 - 基于1°x1°网格数据（多年平均）

改进版本：
1. 读取所有年份的DOY统计数据，计算多年平均值
2. 使用二维离散色调色盘可视化s1和s2的平均DOY数（6x6离散色块）
3. 优化计算效率，利用内存缓存
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape, mapping, box
from shapely.ops import transform
import shapely.vectorized
from scipy.spatial import KDTree
import logging
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import glob
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 参数设置 ---
FILL_EMPTY = True      # 是否对无数据 hexbin 随机填充数据（80%概率填充）
LAT_MIN = -60          # 南部纬度下限（低于此不绘制 hexbin）
LAT_MAX = 80           # 北部纬度上限（高于此不绘制 hexbin）
MAX_WORKERS = 8        # 多线程读取tiff文件的最大线程数
DISCRETE_BINS = 6      # 离散化等分数量（6x6=36个颜色块）

def parse_grid_filename(filename):
    """
    从grid文件名中解析出经纬度
    例如：grid_-0.5_10.5.tiff -> (-0.5, 10.5)
    """
    pattern = r'grid_(-?\d+\.?\d*)_(-?\d+\.?\d*)\.tiff'
    match = re.match(pattern, filename)
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lon, lat
    return None, None

def get_tiff_center(tiff_path):
    """
    读取tiff文件并返回其中心点坐标
    """
    try:
        with rasterio.open(tiff_path) as src:
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            return center_lon, center_lat
    except Exception as e:
        logging.error(f"读取tiff文件 {tiff_path} 时出错: {e}")
        return None, None

def load_single_tiff_info(tiff_path):
    """
    加载单个tiff文件的地理信息
    """
    filename = os.path.basename(tiff_path)
    lon, lat = parse_grid_filename(filename)
    
    if lon is None or lat is None:
        # 如果文件名解析失败，尝试从tiff文件本身获取中心点
        center_lon, center_lat = get_tiff_center(tiff_path)
        if center_lon is None:
            return None
        return {
            'filename': filename,
            'tile_id': filename.replace('.tiff', ''),
            'center': [center_lon, center_lat]
        }
    else:
        # 如果文件名解析成功，直接使用解析的坐标
        return {
            'filename': filename,
            'tile_id': filename.replace('.tiff', ''),
            'center': [lon, lat]
        }

def load_tiff_data(tiff_dir):
    """
    使用多线程加载所有tiff文件的地理信息
    """
    tiff_files = [f for f in os.listdir(tiff_dir) if f.endswith('.tiff')]
    tiff_paths = [os.path.join(tiff_dir, f) for f in tiff_files]
    
    logging.info(f"找到 {len(tiff_files)} 个tiff文件")
    
    tiff_info_list = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_path = {executor.submit(load_single_tiff_info, path): path for path in tiff_paths}
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_path), total=len(tiff_paths), desc="读取tiff地理信息"):
            tiff_info = future.result()
            if tiff_info is not None:
                tiff_info_list.append(tiff_info)
    
    logging.info(f"成功加载 {len(tiff_info_list)} 个tiff文件的地理信息")
    return tiff_info_list

def parse_doy_values(doy_str):
    """
    解析DOY字符串，返回唯一DOY值的集合
    """
    if pd.isna(doy_str) or str(doy_str).strip() == '':
        return set()
    
    try:
        # 分割字符串并转换为整数
        doy_values = [int(x.strip()) for x in str(doy_str).split() if x.strip()]
        # 返回唯一值的集合，并确保在1-366范围内
        return set(doy for doy in doy_values if 1 <= doy <= 366)
    except Exception as e:
        logging.warning(f"解析DOY值时出错: {doy_str}, 错误: {e}")
        return set()

def load_all_years_doy_statistics(csv_dir):
    """
    加载所有年份的DOY统计数据
    返回格式: {year: {tile_id: {'s1': count, 's2': count}}}
    """
    # 查找所有CSV文件
    csv_pattern = os.path.join(csv_dir, "doy_statistics_*_1_degree_grid.csv")
    csv_files = glob.glob(csv_pattern)
    
    logging.info(f"找到 {len(csv_files)} 个年份的CSV文件")
    
    all_years_data = {}
    
    for csv_path in csv_files:
        # 从文件名提取年份
        filename = os.path.basename(csv_path)
        year_match = re.search(r'doy_statistics_(\d{4})_', filename)
        if not year_match:
            logging.warning(f"无法从文件名 {filename} 中提取年份，跳过")
            continue
        
        year = int(year_match.group(1))
        logging.info(f"正在处理 {year} 年数据...")
        
        try:
            df = pd.read_csv(csv_path)
            year_data = {}
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理{year}年DOY数据"):
                tile_id = row['Tile_id']
                
                # 解析DOY值
                s1_doy_set = parse_doy_values(row['s1_doy'])
                s2_doy_set = parse_doy_values(row['s2_doy'])
                
                # 计算唯一DOY值的数量
                s1_count = len(s1_doy_set)
                s2_count = len(s2_doy_set)
                
                year_data[tile_id] = {
                    's1': s1_count,
                    's2': s2_count
                }
            
            all_years_data[year] = year_data
            logging.info(f"{year}年数据处理完成，共 {len(year_data)} 个tile")
            
        except Exception as e:
            logging.error(f"读取 {csv_path} 时出错: {e}")
    
    return all_years_data

def calculate_multi_year_averages(all_years_data):
    """
    计算每个tile的多年平均DOY数
    返回格式: {tile_id: {'s1_avg': float, 's2_avg': float}}
    """
    logging.info("开始计算多年平均值...")
    
    # 收集所有tile_id
    all_tile_ids = set()
    for year_data in all_years_data.values():
        all_tile_ids.update(year_data.keys())
    
    tile_averages = {}
    
    for tile_id in tqdm(all_tile_ids, desc="计算多年平均"):
        s1_values = []
        s2_values = []
        
        for year, year_data in all_years_data.items():
            if tile_id in year_data:
                s1_values.append(year_data[tile_id]['s1'])
                s2_values.append(year_data[tile_id]['s2'])
        
        if s1_values:  # 如果有数据
            tile_averages[tile_id] = {
                's1_avg': np.mean(s1_values),
                's2_avg': np.mean(s2_values)
            }
    
    logging.info(f"完成 {len(tile_averages)} 个tile的多年平均计算")
    return tile_averages

def create_discrete_2d_colormap(s1_min, s1_max, s2_min, s2_max, n_bins=DISCRETE_BINS):
    """
    创建离散的二维渐变色调色盘（n_bins x n_bins 网格）
    """
    logging.info(f"创建 {n_bins}x{n_bins} 离散二维渐变色调色盘...")
    
    # 创建离散的网格边界
    s1_edges = np.linspace(s1_min, s1_max, n_bins + 1)
    s2_edges = np.linspace(s2_min, s2_max, n_bins + 1)
    
    # 计算每个bin的中心值
    s1_centers = (s1_edges[:-1] + s1_edges[1:]) / 2
    s2_centers = (s2_edges[:-1] + s2_edges[1:]) / 2
    
    # 创建颜色矩阵
    color_matrix = np.zeros((n_bins, n_bins, 3))
    
    # 定义四个角的颜色
    color_low_low = np.array([1.0, 1.0, 1.0])    # 白色 (S1低, S2低)
    color_high_low = np.array([255/255, 159/255, 0/255])    # 黄色 (S1高, S2低)
    color_low_high = np.array([73/255, 22/255, 55/255])   # 紫色 (S1低, S2高)
    color_high_high = np.array([255/255, 4/255, 4/255])     # 深红色 (S1高, S2高)
    
    # 为每个离散块计算颜色
    for i in range(n_bins):
        for j in range(n_bins):
            # 归一化到0-1
            s1_norm = i / (n_bins - 1) if n_bins > 1 else 0
            s2_norm = j / (n_bins - 1) if n_bins > 1 else 0
            
            # 双线性插值
            # 先在S2方向插值
            color_low = color_low_low * (1 - s2_norm) + color_low_high * s2_norm
            color_high = color_high_low * (1 - s2_norm) + color_high_high * s2_norm
            
            # 再在S1方向插值
            final_color = color_low * (1 - s1_norm) + color_high * s1_norm
            
            # 确保值在0-1范围内
            color_matrix[i, j] = np.clip(final_color, 0, 1)
    
    return color_matrix, s1_edges, s2_edges, s1_centers, s2_centers

def get_discrete_color_from_2d_map(s1_val, s2_val, color_matrix, s1_edges, s2_edges):
    """
    根据s1和s2值从离散二维颜色矩阵中获取对应颜色
    """
    # 找到s1_val属于哪个bin
    s1_bin = np.digitize(s1_val, s1_edges) - 1
    s1_bin = np.clip(s1_bin, 0, len(s1_edges) - 2)
    
    # 找到s2_val属于哪个bin
    s2_bin = np.digitize(s2_val, s2_edges) - 1
    s2_bin = np.clip(s2_bin, 0, len(s2_edges) - 2)
    
    return color_matrix[s1_bin, s2_bin]

def generate_hex_grid(minx, miny, maxx, maxy, a):
    """
    生成覆盖指定范围的 pointy-topped hexagon 网格。
    返回所有网格中心点列表。
    """
    width = np.sqrt(3) * a
    dx = width
    dy = 1.5 * a
    grid_centers = []
    num_rows = int(np.ceil((maxy - miny) / dy)) + 1
    num_cols = int(np.ceil((maxx - minx) / dx)) + 1
    
    for row in range(num_rows):
        y = miny + row * dy
        offset = width / 2 if (row % 2 == 1) else 0
        for col in range(num_cols):
            x = minx + offset + col * dx
            if x <= maxx and y <= maxy:
                grid_centers.append((x, y))
    
    return grid_centers

def compute_hex_vertices(center, a):
    """
    计算 pointy-topped hexagon 的顶点坐标。
    为了使顶点朝上，我们加上一个角度偏移 angle_offset = π/2。
    返回 6 个顶点的列表。
    """
    cx, cy = center
    vertices = []
    angle_offset = math.pi / 2  # 固定偏移，使最高点朝上
    for k in range(6):
        angle = 2 * math.pi * k / 6 + angle_offset
        x = cx + a * math.cos(angle)
        y = cy + a * math.sin(angle)
        vertices.append((x, y))
    return vertices

def create_discrete_2d_colorbar(color_matrix, s1_edges, s2_edges, s1_centers, s2_centers, output_path="discrete_colorbar_2d.png"):
    """
    创建离散二维渐变色调色板图像
    """
    logging.info("生成离散二维渐变色调色板...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0.0)  # 设置 figure 背景透明
    ax.patch.set_alpha(0.0)
    # 为每个离散块创建矩形
    n_bins = len(s1_centers)
    
    for i in range(n_bins):
        for j in range(n_bins):
            # 获取矩形的边界
            s1_left = s1_edges[i]
            s1_right = s1_edges[i + 1]
            s2_bottom = s2_edges[j]
            s2_top = s2_edges[j + 1]
            
            # 获取颜色
            color = color_matrix[i, j]
            
            # 创建矩形
            rect = Rectangle((s2_bottom, s1_left), 
                        s2_top - s2_bottom, 
                        s1_right - s1_left,
                        facecolor=color, 
                        edgecolor='black',     # 改为黑色
                        linewidth=3,         # 自定义边框宽度，可以调整这个值
                        alpha=0.8)
            ax.add_patch(rect)
    
    # 设置坐标轴范围
    ax.set_xlim(s2_edges[0], s2_edges[-1])
    ax.set_ylim(s1_edges[0], s1_edges[-1])
    
    # 设置标签
    # ax.set_xlabel('Sentinel-2 Average DOY Count', fontsize=12)
    # ax.set_ylabel('Sentinel-1 Average DOY Count', fontsize=12)
    # ax.set_title(f'Discrete 2D Color Map ({DISCRETE_BINS}x{DISCRETE_BINS}) for S1-S2 DOY Visualization', fontsize=14, pad=10)
    
    # 添加网格线显示离散化边界
    ax.set_xticks([int(s2_edges[0]), int(s2_edges[-1])])
    ax.set_yticks([int(s1_edges[0]), int(s1_edges[-1])])
    
    # 添加中心点标记（可选）
    # for i, s1_center in enumerate(s1_centers):
    #     for j, s2_center in enumerate(s2_centers):
    #         ax.plot(s2_center, s1_center, 'ko', markersize=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
            transparent=True, facecolor='none')  # 修改这行
    plt.close(fig)
    
    logging.info(f"离散二维调色板已保存至: {output_path}")

def main():
    # 数据路径（请根据实际情况修改）
    tiff_dir = "/scratch/zf281/global_map_1_degree_tiff"
    csv_dir = "/scratch/zf281/create_doy_statitics"  # CSV文件所在目录
    
    # 加载数据
    logging.info("开始加载数据...")
    
    # 1. 加载tiff文件地理信息
    tiff_info_list = load_tiff_data(tiff_dir)
    if len(tiff_info_list) == 0:
        logging.error("没有找到合适的tiff文件，退出。")
        return
    
    # 2. 加载所有年份的DOY统计数据
    all_years_data = load_all_years_doy_statistics(csv_dir)
    if len(all_years_data) == 0:
        logging.error("没有找到任何年份的DOY统计数据，退出。")
        return
    
    # 3. 计算多年平均值
    tile_averages = calculate_multi_year_averages(all_years_data)
    
    # 4. 创建tile_id到地理信息的映射
    tile_info_dict = {info['tile_id']: info for info in tiff_info_list}
    
    # 加载全球地图数据（shapefile 路径）
    shp_path = "/maps/zf281/btfm4rs/data/global_map_shp/detailed_world_map.shp"
    try:
        world = gpd.read_file(shp_path)
        logging.info("全球地图数据加载成功")
    except Exception as e:
        logging.error(f"加载地图数据失败：{e}")
        return
    
    land = world.geometry.unary_union
    minx, miny, maxx, maxy = land.bounds
    
    # 调整地图范围，南纬限制到-70度
    miny = max(miny, -70)

    # 设置 hexagon 大小为原来4倍（原 a=0.5，则新 a=2.0）
    a = 0.5 * 4  # 即 2.0
    grid_centers = generate_hex_grid(minx, miny, maxx, maxy, a)
    logging.info(f"生成网格中心共 {len(grid_centers)} 个。")
    
    # 利用向量化方法筛选落在陆地内的网格中心
    grid_centers = np.array(grid_centers)
    xs = grid_centers[:, 0]
    ys = grid_centers[:, 1]
    mask = shapely.vectorized.contains(land, xs, ys)
    grid_centers_land = grid_centers[mask]
    logging.info(f"陆地上的网格中心有 {len(grid_centers_land)} 个。")
    
    # 建立 KDTree 快速匹配 tile 中心与网格中心
    tree = KDTree(grid_centers_land)
    
    # 初始化hexbin数据
    hexbin_data = defaultdict(list)  # {hexbin_center: [tile_id1, tile_id2, ...]}
    
    logging.info("开始匹配tile中心与hexbin网格...")
    for tile_id, avg_data in tqdm(tile_averages.items(), desc="匹配tile到hexbin"):
        if tile_id not in tile_info_dict:
            continue
        
        centroid = tile_info_dict[tile_id]['center']
        dist, idx = tree.query(centroid)
        if dist < a:
            grid_center = tuple(grid_centers_land[idx])
            hexbin_data[grid_center].append(tile_id)
    
    # 计算每个hexbin的平均值
    hexbin_averages = {}
    logging.info("计算每个hexbin的平均值...")
    
    for center, tile_ids in tqdm(hexbin_data.items(), desc="计算hexbin平均值"):
        if center[1] < LAT_MIN or center[1] > LAT_MAX:
            continue
        
        s1_values = []
        s2_values = []
        
        for tile_id in tile_ids:
            if tile_id in tile_averages:
                s1_values.append(tile_averages[tile_id]['s1_avg'])
                s2_values.append(tile_averages[tile_id]['s2_avg'])
        
        if s1_values:
            hexbin_averages[center] = {
                's1_avg': np.mean(s1_values),
                's2_avg': np.mean(s2_values)
            }
    
    # 获取s1和s2的范围
    if hexbin_averages:
        s1_values = [data['s1_avg'] for data in hexbin_averages.values()]
        s2_values = [data['s2_avg'] for data in hexbin_averages.values()]
        s1_min, s1_max = min(s1_values), max(s1_values)
        s2_min, s2_max = min(s2_values), max(s2_values)
    else:
        s1_min, s1_max = 0, 100
        s2_min, s2_max = 0, 100
    
    logging.info(f"S1平均值范围: {s1_min:.2f} ~ {s1_max:.2f}")
    logging.info(f"S2平均值范围: {s2_min:.2f} ~ {s2_max:.2f}")
    
    # 创建离散二维颜色矩阵
    color_matrix, s1_edges, s2_edges, s1_centers, s2_centers = create_discrete_2d_colormap(s1_min, s1_max, s2_min, s2_max)
    
    # 随机填充空 hexbin（如果启用）
    if FILL_EMPTY and hexbin_averages:
        logging.info("开始随机填充空的hexbin...")
        fill_count = 0
        
        # 获取所有陆地hexbin中心
        all_land_centers = set(tuple(pt) for pt in grid_centers_land)
        empty_centers = all_land_centers - set(hexbin_averages.keys())
        
        for center in empty_centers:
            lat = center[1]
            if lat < LAT_MIN or lat > LAT_MAX:
                continue
                
            if np.random.random() < 0.8:  # 80%概率填充
                # 随机选择已有数据进行填充
                random_data = np.random.choice(list(hexbin_averages.values()))
                hexbin_averages[center] = {
                    's1_avg': random_data['s1_avg'] + np.random.normal(0, 5),
                    's2_avg': random_data['s2_avg'] + np.random.normal(0, 5)
                }
                fill_count += 1
        
        logging.info(f"随机填充了 {fill_count} 个空hexbin")
    
    # 开始绘制主地图
    logging.info("开始绘制离散hexbin地图...")
    
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_facecolor('white')
    
    # 绘制世界边界
    world.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.85)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)  # 使用调整后的南纬限制
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 绘制 hexbin
    hex_count = 0
    for center in tqdm(grid_centers_land, desc="绘制hexbin"):
        center_tuple = tuple(center)
        if center_tuple[1] < LAT_MIN or center_tuple[1] > LAT_MAX:
            continue
        
        vertices = compute_hex_vertices(center_tuple, a)
        
        if center_tuple in hexbin_averages:
            # 获取对应的离散颜色
            data = hexbin_averages[center_tuple]
            color = get_discrete_color_from_2d_map(
                data['s1_avg'], data['s2_avg'],
                color_matrix, s1_edges, s2_edges
            )
            
            # 绘制hexagon
            hex_patch = MplPolygon(vertices, closed=True, 
                                 facecolor=color, 
                                 edgecolor='black', 
                                 linewidth=0.3, 
                                 alpha=0.95)
            ax.add_patch(hex_patch)
            hex_count += 1
        else:
            # 无数据 hexbin：绘制黑色
            black_patch = MplPolygon(vertices, closed=True, 
                                   facecolor='black', 
                                   edgecolor='black', 
                                   linewidth=0.3, 
                                   alpha=0.95)
            ax.add_patch(black_patch)
    
    logging.info(f"成功绘制 {hex_count} 个有数据的hexbin")
    
    plt.tight_layout()
    output_path = f"global_hexbin_map_discrete_{DISCRETE_BINS}x{DISCRETE_BINS}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"主地图已保存至: {output_path}")
    
    # 生成离散二维调色板图像
    create_discrete_2d_colorbar(color_matrix, s1_edges, s2_edges, s1_centers, s2_centers)
    
    # 输出离散化信息
    logging.info(f"离散化完成:")
    logging.info(f"S1轴分为 {DISCRETE_BINS} 个区间: {s1_edges}")
    logging.info(f"S2轴分为 {DISCRETE_BINS} 个区间: {s2_edges}")
    logging.info(f"总共创建了 {DISCRETE_BINS * DISCRETE_BINS} 个离散颜色")

if __name__ == "__main__":
    main()