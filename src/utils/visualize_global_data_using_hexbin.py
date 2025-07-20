#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局 hexbin 地图绘制脚本

要求：
1. Hexbin 大小为原来4倍。如果某个 hexbin 内包含多个 MGRS tile，则合并它们的 doy 信息（分别取哨兵1与哨兵2的集合后统计）。
2. 绘制的 hexbin 使用 pointy-topped（顶点朝上），且保证 hexbin 不重叠。
3. 增加布尔变量 FILL_EMPTY，为 True 时，对没有数据的 hexbin随机填充一些值（仅 80% 的空 hexbin填充，剩下的显示为黑色）；为 False 时，空 hexbin均显示为黑色。
4. 设置纬度上下限（LAT_MIN 与 LAT_MAX），对于低于或高于该范围的陆地，不绘制 hexbin（但显示 world 边界，便于观察南极边界）。
5. 在 colorbar 下方增加 legend：左侧 legend 显示 S1 与 S2 的叠加效果（颜色取 75% 分位数对应的颜色），右侧 legend 显示无数据 hexbin（黑色）。Legend中不显示任何文字。

数据说明：
- 每个 tile 文件夹名称为 MGRS tile ID，通过 from_mgrs_to_polygon() 得到 tile 的多边形，并取其质心作为中心。
- 每个 tile 仅使用 doys.npy（哨兵2）和 sar_ascending_doy.npy 与 sar_descending_doy.npy（哨兵1），分别以集合形式保存其 doy 值。

请根据实际情况修改 tile 数据与地图数据的路径。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape, mapping, box
from shapely.ops import transform
import shapely.vectorized
from scipy.spatial import KDTree
import mgrs
import pyproj
import logging
import math

logging.basicConfig(level=logging.INFO)

# --- 参数设置 ---
FILL_EMPTY = True      # 是否对无数据 hexbin 随机填充数据（80%概率填充）
LAT_MIN = -60          # 南部纬度下限（低于此不绘制 hexbin）
LAT_MAX = 80           # 北部纬度上限（高于此不绘制 hexbin）

def from_mgrs_to_polygon(tile_id: str, buffer_meters: float = 500) -> dict:
    """
    将 MGRS tile_id 转换为带缓冲区的 GeoJSON 多边形。
    """
    mgrs_converter = mgrs.MGRS()
    mgrs_center = tile_id + '50000' + '50000'
    try:
        lat, lon = mgrs_converter.toLatLon(mgrs_center)  # 返回 (lat, lon)
        point = Point(lon, lat)
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = 'north' if lat >= 0 else 'south'
        utm_crs = pyproj.CRS.from_dict({
            'proj': 'utm',
            'zone': utm_zone,
            'south': hemisphere == 'south'
        })
        project_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform
        project_to_wgs84 = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform
        point_utm = transform(project_to_utm, point)
        buffered_utm = point_utm.buffer(buffer_meters)
        buffered_polygon = transform(project_to_wgs84, buffered_utm)
        return mapping(buffered_polygon)
    except Exception as e:
        logging.error(f"转换 MGRS 到经纬度时出错 {mgrs_center}: {e}")
        return None

def load_tile_data(data_dir):
    """
    遍历 data_dir 下各个 MGRS tile 文件夹，读取 doys.npy、sar_ascending_doy.npy 和 sar_descending_doy.npy，
    分别将哨兵2与哨兵1的 doy 值以集合方式保存，
    并利用 from_mgrs_to_polygon() 得到 tile 的中心（取多边形质心）。
    返回 tile 数据列表。
    """
    tile_data_list = []
    for tile in os.listdir(data_dir):
        tile_path = os.path.join(data_dir, tile)
        if not os.path.isdir(tile_path):
            continue
        required_files = ['doys.npy', 'sar_ascending_doy.npy', 'sar_descending_doy.npy']
        if not all(os.path.exists(os.path.join(tile_path, f)) for f in required_files):
            continue
        try:
            s2_arr = np.load(os.path.join(tile_path, 'doys.npy'))
            s1_asc_arr = np.load(os.path.join(tile_path, 'sar_ascending_doy.npy'))
            s1_desc_arr = np.load(os.path.join(tile_path, 'sar_descending_doy.npy'))
        except Exception as e:
            logging.error(f"加载 tile {tile} 时出错: {e}")
            continue
        s2_doys = set(s2_arr.tolist())
        s1_doys = set(s1_asc_arr.tolist()).union(set(s1_desc_arr.tolist()))
        s1_count = len(s1_doys)
        s2_count = len(s2_doys)
        total_count = s1_count + s2_count
        poly_geojson = from_mgrs_to_polygon(tile, buffer_meters=500)
        if poly_geojson is None:
            logging.error(f"tile {tile} 转换 polygon 失败，跳过。")
            continue
        poly = shape(poly_geojson)
        centroid = poly.centroid
        center = [centroid.x, centroid.y]
        tile_data_list.append({
            'mgrs': tile,
            'center': center,
            's1_doys': s1_doys,
            's2_doys': s2_doys,
            's1': s1_count,
            's2': s2_count,
            'total': total_count
        })
    return tile_data_list

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

def main():
    # tile 数据目录（请根据实际情况修改）
    data_dir = "/mnt/e/Codes/btfm4rs/data/ssl_training/global"
    tile_data_list = load_tile_data(data_dir)
    logging.info(f"加载到 {len(tile_data_list)} 个 tile 数据。")
    if len(tile_data_list) == 0:
        logging.error("没有找到合适的 tile 数据，退出。")
        return

    # 加载全球地图数据（shapefile 路径）
    shp_path = "/mnt/e/Codes/btfm4rs/data/global_map/ne_110m_admin_0_countries.shp"
    try:
        world = gpd.read_file(shp_path)
    except Exception as e:
        logging.error(f"加载地图数据失败：{e}")
        return
    land = world.geometry.unary_union
    minx, miny, maxx, maxy = land.bounds

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
    grid_dict = {tuple(pt): None for pt in grid_centers_land}
    for tile in tile_data_list:
        centroid = tile['center']
        dist, idx = tree.query(centroid)
        if dist < a:
            grid_center = tuple(grid_centers_land[idx])
            if grid_dict[grid_center] is None:
                grid_dict[grid_center] = {
                    's1_doys': set(tile['s1_doys']),
                    's2_doys': set(tile['s2_doys'])
                }
            else:
                grid_dict[grid_center]['s1_doys'] |= tile['s1_doys']
                grid_dict[grid_center]['s2_doys'] |= tile['s2_doys']
        else:
            logging.info(f"Tile {tile['mgrs']} 中心 {centroid} 距离最近网格中心 {dist:.2f}°，未分配。")
    
    # 合并数据，构建 merged_data 字典
    merged_data = {}
    for center, data in grid_dict.items():
        if data is not None:
            s1_count = len(data['s1_doys'])
            s2_count = len(data['s2_doys'])
            total = s1_count + s2_count
            merged_data[center] = {'s1': s1_count, 's2': s2_count, 'total': total,
                                   's1_doys': data['s1_doys'], 's2_doys': data['s2_doys']}
        else:
            merged_data[center] = None

    # 随机填充空 hexbin：仅80%概率填充，其余保持为无数据
    existing_totals = [d['total'] for d in merged_data.values() if d is not None]
    existing_ratios = [d['s1']/d['total'] for d in merged_data.values() if d is not None and d['total']>0]
    if FILL_EMPTY and existing_totals and existing_ratios:
        for center, data in merged_data.items():
            lat = center[1]
            if lat < LAT_MIN or lat > LAT_MAX:
                continue
            if data is None and np.random.random() < 0.98:
                total_rand = int(np.random.choice(existing_totals))
                ratio_rand = float(np.random.choice(existing_ratios))
                s1_rand = int(round(total_rand * ratio_rand))
                s2_rand = total_rand - s1_rand
                merged_data[center] = {'s1': s1_rand, 's2': s2_rand, 'total': total_rand}
    # 计算 total doy 范围（仅统计在纬度范围内且有数据的 hexbin）
    total_counts = [d['total'] for center, d in merged_data.items()
                    if d is not None and LAT_MIN <= center[1] <= LAT_MAX]
    if total_counts:
        min_total_val = min(total_counts)
        max_total_val = max(total_counts)
    else:
        min_total_val, max_total_val = 0, 1
    logging.info(f"总 doy 数范围：{min_total_val} ~ {max_total_val}")
    
    cmap = plt.cm.BuPu
    norm = plt.Normalize(vmin=min_total_val, vmax=max_total_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_facecolor('white')
    
    # 绘制 hexbin：先绘制 S1 的完整 hexagon，再绘制 S2 的截断区域
    for center, data in merged_data.items():
        if center[1] < LAT_MIN or center[1] > LAT_MAX:
            continue
        vertices = compute_hex_vertices(center, a)
        hex_poly = Polygon(vertices)
        if data is not None:
            color = cmap(norm(data['total']))
            # 绘制 S1 部分：整个 hexagon，alpha=0.5，带边框
            s1_patch = MplPolygon(vertices, closed=True, facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.75)
            ax.add_patch(s1_patch)
            # S2 部分：根据 S2 占比截取顶部区域
            fraction = data['s2'] / data['total'] if data['total'] > 0 else 0
            if fraction > 0:
                total_height = a * math.sqrt(3)
                ys_vertices = [v[1] for v in vertices]
                Y_top = max(ys_vertices)
                Y_cut = Y_top - total_height * fraction
                clip_rect = box(-1e9, Y_cut, 1e9, 1e9)
                s2_poly = hex_poly.intersection(clip_rect)
                if not s2_poly.is_empty:
                    if s2_poly.geom_type == 'Polygon':
                        s2_coords = list(s2_poly.exterior.coords)
                        s2_patch = MplPolygon(s2_coords, closed=True, facecolor=color, edgecolor=(0, 0, 0, 0.3), linewidth=0.3, alpha=0.75)
                        ax.add_patch(s2_patch)
                    elif s2_poly.geom_type == 'MultiPolygon':
                        for poly in s2_poly:
                            s2_coords = list(poly.exterior.coords)
                            s2_patch = MplPolygon(s2_coords, closed=True, facecolor=color, edgecolor=(0, 0, 0, 0.3), linewidth=0.3, alpha=0.75)
                            ax.add_patch(s2_patch)
        else:
            # 无数据 hexbin：绘制黑色
            black_patch = MplPolygon(vertices, closed=True, facecolor='black', edgecolor='black', linewidth=0.5, alpha=1)
            ax.add_patch(black_patch)
    
    # 绘制世界边界
    world.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.8)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 在地图左下侧添加 colorbar
    cbar_ax = fig.add_axes([0.05, 0.3, 0.02, 0.3])
    fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    
    # -------------------------
    # 添加 legend：在 colorbar 下方增加两个 hexbin
    # 计算 75% 分位数对应的 total 值，取颜色
    if total_counts:
        total_75 = np.percentile(total_counts, 75)
    else:
        total_75 = 0
    legend_color = cmap(norm(total_75))
    # 在 figure 坐标中添加两个小轴作为 legend 区域
    # 左侧：有数据 legend（S1+S2效果），右侧：无数据 legend（黑色）
    legend_ax1 = fig.add_axes([0.05, 0.05, 0.04, 0.08])
    legend_ax2 = fig.add_axes([0.11, 0.05, 0.04, 0.08])
    for lax in [legend_ax1, legend_ax2]:
        lax.set_xlim(0,1)
        lax.set_ylim(0,1)
        lax.axis('off')
    # 在 legend_ax1 绘制有数据 hexbin：先绘制 S1，再绘制 S2叠加
    center_legend = (0.5, 0.5)
    r_legend = 0.4
    # 计算 legend hexagon顶点
    def compute_hex_vertices_legend(center, r):
        cx, cy = center
        verts = []
        angle_offset = math.pi / 2
        for k in range(6):
            angle = 2 * math.pi * k / 6 + angle_offset
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            verts.append((x, y))
        return verts
    legend_vertices = compute_hex_vertices_legend(center_legend, r_legend)
    legend_hex = Polygon(legend_vertices)
    # 绘制 S1部分（全 hexagon，alpha=0.5）
    leg_s1 = MplPolygon(legend_vertices, closed=True, facecolor=legend_color, edgecolor='black', linewidth=0.5, alpha=0.75)
    legend_ax1.add_patch(leg_s1)
    # S2部分：以固定比例（例如0.5）截取顶部区域
    fraction_legend = 0.5
    total_height_legend = r_legend * math.sqrt(3)
    ys_leg = [v[1] for v in legend_vertices]
    Y_top_leg = max(ys_leg)
    Y_cut_leg = Y_top_leg - total_height_legend * fraction_legend
    clip_rect_leg = box(-1e9, Y_cut_leg, 1e9, 1e9)
    s2_leg_poly = legend_hex.intersection(clip_rect_leg)
    if not s2_leg_poly.is_empty:
        if s2_leg_poly.geom_type == 'Polygon':
            s2_leg_coords = list(s2_leg_poly.exterior.coords)
            leg_s2 = MplPolygon(s2_leg_coords, closed=True, facecolor=legend_color, edgecolor=None, alpha=0.75)
            legend_ax1.add_patch(leg_s2)
        elif s2_leg_poly.geom_type == 'MultiPolygon':
            for poly in s2_leg_poly:
                s2_leg_coords = list(poly.exterior.coords)
                leg_s2 = MplPolygon(s2_leg_coords, closed=True, facecolor=legend_color, edgecolor=None, alpha=0.75)
                legend_ax1.add_patch(leg_s2)
    # 在 legend_ax2 绘制无数据 hexbin：黑色
    legend_vertices2 = compute_hex_vertices_legend(center_legend, r_legend)
    leg_empty = MplPolygon(legend_vertices2, closed=True, facecolor='black', edgecolor='black', linewidth=0.5, alpha=1)
    legend_ax2.add_patch(leg_empty)
    
    plt.tight_layout()
    plt.savefig("global_hexbin_map.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()
