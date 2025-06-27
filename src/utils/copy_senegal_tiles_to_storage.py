import os
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import box
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from pathlib import Path

def load_boundary(boundary_path):
    """
    加载边界文件（支持SHP, GPKG等格式）
    """
    print(f"加载边界文件: {boundary_path}")
    boundary = gpd.read_file(boundary_path)
    
    # 确保投影为WGS84
    if boundary.crs != "EPSG:4326":
        print(f"将坐标系从 {boundary.crs} 转换为 EPSG:4326")
        boundary = boundary.to_crs("EPSG:4326")
    
    # 获取边界范围
    bounds = boundary.total_bounds
    print(f"边界范围: ")
    print(f"  经度: {bounds[0]:.4f} 到 {bounds[2]:.4f}")
    print(f"  纬度: {bounds[1]:.4f} 到 {bounds[3]:.4f}")
    
    return boundary, bounds

def parse_tiff_filename(filename):
    """
    从TIFF文件名中提取中心坐标
    文件名格式: grid_{center_lon:.2f}_{center_lat:.2f}.tiff
    """
    try:
        parts = filename.replace('.tiff', '').split('_')
        if len(parts) >= 3 and parts[0] == 'grid':
            center_lon = float(parts[1])
            center_lat = float(parts[2])
            return center_lon, center_lat
        return None, None
    except:
        return None, None

def get_potential_tiff_files(tiff_dir, bounds, buffer_degrees=0.2):
    """
    根据文件名中的坐标信息，筛选可能在目标范围内的TIFF文件
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # 添加缓冲区以确保不遗漏边界附近的文件
    min_lon -= buffer_degrees
    max_lon += buffer_degrees
    min_lat -= buffer_degrees
    max_lat += buffer_degrees
    
    print(f"\n搜索范围（含缓冲）:")
    print(f"  经度: {min_lon:.4f} 到 {max_lon:.4f}")
    print(f"  纬度: {min_lat:.4f} 到 {max_lat:.4f}")
    
    potential_files = []
    
    # 构建可能的文件名模式
    print("\n扫描TIFF文件...")
    
    # 遍历可能的经纬度范围
    for lon in np.arange(int(min_lon) - 1.05, int(max_lon) + 2.05, 0.1):
        for lat in np.arange(int(min_lat) - 1.05, int(max_lat) + 2.05, 0.1):
            # 构建文件名
            filename = f"grid_{lon:.2f}_{lat:.2f}.tiff"
            filepath = os.path.join(tiff_dir, filename)
            
            # 检查文件是否存在
            if os.path.exists(filepath):
                center_lon, center_lat = parse_tiff_filename(filename)
                if center_lon is not None:
                    # 检查中心点是否在搜索范围内
                    if (min_lon <= center_lon <= max_lon and 
                        min_lat <= center_lat <= max_lat):
                        potential_files.append(filepath)
    
    print(f"找到 {len(potential_files)} 个潜在的TIFF文件")
    return potential_files

def check_tiff_intersection(tiff_path, boundary, grid_size=0.1):
    """
    检查TIFF文件对应的网格是否与目标边界相交
    """
    filename = os.path.basename(tiff_path)
    center_lon, center_lat = parse_tiff_filename(filename)
    
    if center_lon is None:
        return False
    
    # 创建网格边界框（0.1度 x 0.1度）
    half_size = grid_size / 2
    grid_box = box(
        center_lon - half_size,
        center_lat - half_size,
        center_lon + half_size,
        center_lat + half_size
    )
    
    # 创建网格的GeoDataFrame
    grid_gdf = gpd.GeoDataFrame([{'geometry': grid_box}], crs="EPSG:4326")
    
    # 检查是否与边界相交
    intersects = boundary.intersects(grid_box).any()
    
    return intersects

def process_and_save_results(tiff_dir, boundary_path, output_txt, output_plot, region_name="目标区域"):
    """
    主处理函数
    """
    # 加载边界
    boundary, bounds = load_boundary(boundary_path)
    
    # 获取潜在的TIFF文件
    potential_files = get_potential_tiff_files(tiff_dir, bounds)
    
    # 检查每个文件是否与边界相交
    print(f"\n检查TIFF文件与{region_name}边界的相交情况...")
    selected_tiff_files = []
    grid_boxes = []
    
    for tiff_path in tqdm(potential_files, desc="检查相交"):
        if check_tiff_intersection(tiff_path, boundary):
            selected_tiff_files.append(tiff_path)
            
            # 保存网格框用于可视化
            filename = os.path.basename(tiff_path)
            center_lon, center_lat = parse_tiff_filename(filename)
            if center_lon is not None:
                grid_box = box(
                    center_lon - 0.05,
                    center_lat - 0.05,
                    center_lon + 0.05,
                    center_lat + 0.05
                )
                grid_boxes.append(grid_box)
    
    print(f"\n找到 {len(selected_tiff_files)} 个在{region_name}内的TIFF文件")
    
    # 保存文件列表到TXT
    print(f"\n保存文件列表到: {output_txt}")
    with open(output_txt, 'w') as f:
        for tiff_path in selected_tiff_files:
            f.write(f"{tiff_path}\n")
    
    # 创建可视化
    print("\n创建可视化图...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制边界
    boundary.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=1, alpha=0.7)
    
    # 绘制选中的网格
    if grid_boxes:
        grid_gdf = gpd.GeoDataFrame({'geometry': grid_boxes}, crs="EPSG:4326")
        grid_gdf.plot(ax=ax, facecolor='red', edgecolor='darkred', alpha=0.5, linewidth=0.5)
    
    # 设置标题和标签
    ax.set_title(f'{region_name}内的TIFF文件覆盖情况 (共 {len(selected_tiff_files)} 个文件)', 
                 fontsize=14, fontproperties='DejaVu Sans')
    ax.set_xlabel('经度', fontsize=12)
    ax.set_ylabel('纬度', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 设置范围
    margin = 0.5
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"可视化图已保存到: {output_plot}")
    
    # 显示图像
    plt.show()
    
    return selected_tiff_files

def create_rsync_script(txt_file, source_base, target_host, target_user, target_path, script_file):
    """
    创建rsync脚本用于传输文件
    """
    print(f"\n创建rsync脚本: {script_file}")
    
    with open(txt_file, 'r') as f:
        files = [line.strip() for line in f.readlines()]
    
    with open(script_file, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# rsync script for selected TIFF files\n")
        f.write(f"# Total files: {len(files)}\n\n")
        
        f.write("# Create target directory if not exists\n")
        f.write(f"ssh {target_user}@{target_host} 'mkdir -p {target_path}'\n\n")
        
        f.write("# Transfer files\n")
        f.write(f"rsync -avzP --files-from={txt_file} / {target_user}@{target_host}:{target_path}/\n")
    
    # 使脚本可执行
    os.chmod(script_file, 0o755)
    print(f"rsync脚本已创建，可以运行: ./{script_file}")

def main():
    """
    主函数 - 可以根据需要修改参数
    """
    # ========== 配置参数 ==========
    # 输入路径
    tiff_dir = "/scratch/zf281/global_map_0.1_degree_tiff"  # TIFF文件目录
    boundary_path = "/maps/zf281/btfm4rs/centralsenegal_jcam_bbox/centralsenegal_jcam_bbox.shp"  # 修改为您的SHP文件路径
    
    # 输出文件
    region_name = "senegal"  # 可以修改为具体地区名称，如"中国"、"美国"等
    output_txt = f"{region_name.lower()}_tiff_files.txt"
    output_plot = f"{region_name.lower()}_tiff_coverage.png"
    rsync_script = f"transfer_{region_name.lower()}_tiffs.sh"
    
    # rsync参数
    target_host = "otrera.caelum.ci.dev"  # 目标服务器
    target_user = "zf281"  # 目标用户名
    target_path = f"/tank/{target_user}/global_0.1_degree_tiff"  # 目标路径
    # =============================
    
    # 处理并保存结果
    selected_tiff_files = process_and_save_results(
        tiff_dir, boundary_path, output_txt, output_plot, region_name
    )
    
    # 创建rsync脚本
    create_rsync_script(
        output_txt, tiff_dir, target_host, target_user, target_path, rsync_script
    )
    
    # 打印统计信息
    print("\n=== 处理完成 ===")
    print(f"找到的{region_name} TIFF文件数量: {len(selected_tiff_files)}")
    print(f"文件列表保存在: {output_txt}")
    print(f"覆盖图保存在: {output_plot}")
    print(f"rsync脚本: {rsync_script}")
    
    # 显示前10个文件作为示例
    if selected_tiff_files:
        print("\n前10个文件示例:")
        for i, f in enumerate(selected_tiff_files[:10]):
            print(f"  {i+1}. {os.path.basename(f)}")

if __name__ == "__main__":
    main()