import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from concurrent.futures import ProcessPoolExecutor, as_completed
import paramiko
import glob
from tqdm import tqdm
import time
from functools import lru_cache

class GridTiffFinder:
    def __init__(self, grid_size=0.1, tiff_dir="/scratch/zf281/global_map_0.1_degree_tiff"):
        self.grid_size = grid_size
        self.tiff_dir = tiff_dir
        self.remote_host = "antiope.cl.cam.ac.uk"
        self.remote_user = "zf281"
        self.remote_path = "/tank/zf281/global_0.1_degree_representation/2024"
        
    def calculate_grid_center(self, lon, lat):
        """根据经纬度计算对应的网格中心坐标"""
        # 计算网格的起始坐标
        grid_lon_start = np.floor(lon / self.grid_size) * self.grid_size
        grid_lat_start = np.floor(lat / self.grid_size) * self.grid_size
        
        # 计算网格中心坐标
        center_lon = grid_lon_start + self.grid_size / 2
        center_lat = grid_lat_start + self.grid_size / 2
        
        return center_lon, center_lat
    
    def get_grid_bounds_from_center(self, center_lon, center_lat):
        """根据网格中心坐标计算网格边界"""
        half_size = self.grid_size / 2
        return {
            'left': center_lon - half_size,
            'right': center_lon + half_size,
            'bottom': center_lat - half_size,
            'top': center_lat + half_size
        }
    
    def get_tiff_filename(self, center_lon, center_lat):
        """根据网格中心坐标生成TIFF文件名"""
        return f"grid_{center_lon:.2f}_{center_lat:.2f}.tiff"
    
    def parse_tiff_filename(self, filename):
        """从TIFF文件名解析出中心坐标"""
        # 格式: grid_-0.05_5.55.tiff
        parts = filename.replace('grid_', '').replace('.tiff', '').split('_')
        if len(parts) == 2:
            try:
                center_lon = float(parts[0])
                center_lat = float(parts[1])
                return center_lon, center_lat
            except ValueError:
                return None, None
        return None, None
    
    def get_processed_grids(self):
        """获取远程服务器上已处理的网格列表"""
        print("连接远程服务器获取已处理的网格...")
        processed_grids = set()
        
        try:
            # 创建SSH客户端
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接到远程服务器
            ssh.connect(self.remote_host, username=self.remote_user)
            
            # 执行命令获取所有grid文件夹
            stdin, stdout, stderr = ssh.exec_command(f"ls -d {self.remote_path}/grid_*")
            
            # 解析输出
            for line in stdout:
                folder_path = line.strip()
                if folder_path:
                    # 提取网格名称
                    grid_name = os.path.basename(folder_path)
                    # 从 grid_-0.05_5.55 格式转换为 grid_-0.05_5.55.tiff
                    processed_grids.add(f"{grid_name}.tiff")
            
            ssh.close()
            print(f"找到 {len(processed_grids)} 个已处理的网格")
            
        except Exception as e:
            print(f"警告：无法连接到远程服务器: {e}")
            print("将继续处理，但无法排除已处理的网格")
        
        return processed_grids
    
    def process_points_batch(self, points_batch):
        """批量处理点，找到对应的TIFF文件"""
        results = []
        
        for idx, row in points_batch.iterrows():
            lon, lat = row['Longitude'], row['Latitude']
            center_lon, center_lat = self.calculate_grid_center(lon, lat)
            tiff_filename = self.get_tiff_filename(center_lon, center_lat)
            tiff_path = os.path.join(self.tiff_dir, tiff_filename)
            
            result = {
                'index': idx,
                'longitude': lon,
                'latitude': lat,
                'center_lon': center_lon,
                'center_lat': center_lat,
                'tiff_filename': tiff_filename,
                'tiff_exists': os.path.exists(tiff_path)
            }
            results.append(result)
        
        return results
    
    def find_tiffs_for_points(self, csv_file):
        """找到所有点对应的TIFF文件"""
        # 读取CSV文件
        print(f"读取CSV文件: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"总共有 {len(df)} 个点")
        
        # 获取已处理的网格
        processed_grids = self.get_processed_grids()
        
        # 使用多进程并行处理点
        num_cores = min(96, os.cpu_count())  # 使用最多96个核心
        print(f"使用 {num_cores} 个CPU核心并行处理")
        
        # 将数据分批
        batch_size = max(1, len(df) // (num_cores * 10))
        batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
        
        # 并行处理
        all_results = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(self.process_points_batch, batch) for batch in batches]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理点批次"):
                all_results.extend(future.result())
        
        # 整理结果
        results_df = pd.DataFrame(all_results)
        
        # 分类点和网格
        points_in_tiff = results_df[results_df['tiff_exists']]
        points_not_in_tiff = results_df[~results_df['tiff_exists']]
        
        # 获取唯一的TIFF文件
        unique_tiffs = points_in_tiff['tiff_filename'].unique()
        
        # 分离已处理和未处理的TIFF
        tiffs_to_process = []
        tiffs_already_processed = []
        
        for tiff in unique_tiffs:
            if tiff in processed_grids:
                tiffs_already_processed.append(tiff)
            else:
                tiffs_to_process.append(tiff)
        
        print(f"\n统计结果:")
        print(f"- 落在TIFF中的点: {len(points_in_tiff)}")
        print(f"- 未找到TIFF的点: {len(points_not_in_tiff)}")
        print(f"- 找到的唯一TIFF文件: {len(unique_tiffs)}")
        print(f"- 已处理的TIFF: {len(tiffs_already_processed)}")
        print(f"- 需要处理的TIFF: {len(tiffs_to_process)}")
        
        return {
            'points_in_tiff': points_in_tiff,
            'points_not_in_tiff': points_not_in_tiff,
            'tiffs_to_process': tiffs_to_process,
            'tiffs_already_processed': tiffs_already_processed,
            'all_results': results_df
        }
    
    def save_results(self, results, output_dir="./output"):
        """保存结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存需要处理的TIFF列表
        tiff_list_file = os.path.join(output_dir, "tiffs_to_process.txt")
        with open(tiff_list_file, 'w') as f:
            for tiff in sorted(results['tiffs_to_process']):
                f.write(f"{tiff}\n")
        print(f"已保存需要处理的TIFF列表到: {tiff_list_file}")
        
        # 保存未找到TIFF的点
        if len(results['points_not_in_tiff']) > 0:
            missing_points_file = os.path.join(output_dir, "points_without_tiff.csv")
            results['points_not_in_tiff'][['longitude', 'latitude']].to_csv(
                missing_points_file, index=False
            )
            print(f"已保存未找到TIFF的点到: {missing_points_file}")
    
    def draw_grid_borders(self, ax, tiff_files, color, linewidth=2, linestyle='-', alpha=1.0):
        """绘制网格边框"""
        rectangles_drawn = 0
        
        for tiff_name in tiff_files:
            center_lon, center_lat = self.parse_tiff_filename(tiff_name)
            if center_lon is not None and center_lat is not None:
                bounds = self.get_grid_bounds_from_center(center_lon, center_lat)
                
                # 创建矩形边框
                rect = Rectangle(
                    (bounds['left'], bounds['bottom']),
                    self.grid_size, self.grid_size,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor='none',
                    linestyle=linestyle,
                    alpha=alpha,
                    zorder=3
                )
                ax.add_patch(rect)
                rectangles_drawn += 1
        
        return rectangles_drawn
    
    def visualize_results(self, results, output_dir="./output", shapefile_path=None, sample_size=2000):
        """可视化结果，显示网格边框"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        
        # 1. 加载地图底图
        if shapefile_path and os.path.exists(shapefile_path):
            print("加载地图底图...")
            world = gpd.read_file(shapefile_path)
            world = world.to_crs('EPSG:4326')  # 确保是WGS84
            
            # 绘制世界地图
            world.plot(ax=ax, color='lightgray', edgecolor='darkgray', linewidth=0.5, zorder=1)
        else:
            print("警告：未找到地图底图文件")
            world = None
        
        # 准备数据
        all_results = results['all_results']
        
        # 2. 绘制TIFF网格边框
        # 如果网格太多，进行采样
        tiffs_to_process = results['tiffs_to_process']
        tiffs_already_processed = results['tiffs_already_processed']
        
        if sample_size > 0:
            if len(tiffs_to_process) > sample_size:
                import random
                tiffs_to_process = random.sample(tiffs_to_process, min(sample_size, len(tiffs_to_process)))
            
            if len(tiffs_already_processed) > sample_size:
                import random
                tiffs_already_processed = random.sample(tiffs_already_processed, min(sample_size, len(tiffs_already_processed)))
        
        print(f"\n绘制网格边框...")
        print(f"- 待处理网格: {len(tiffs_to_process)} 个")
        print(f"- 已处理网格: {len(tiffs_already_processed)} 个")
        
        # 绘制已处理的网格（绿色）
        processed_count = self.draw_grid_borders(
            ax, tiffs_already_processed, 
            color='green', linewidth=1.5, linestyle='-', alpha=0.8
        )
        print(f"  成功绘制 {processed_count} 个已处理网格")
        
        # 绘制待处理的网格（黄色）
        to_process_count = self.draw_grid_borders(
            ax, tiffs_to_process,
            color='orange', linewidth=1.5, linestyle='-', alpha=0.8
        )
        print(f"  成功绘制 {to_process_count} 个待处理网格")
        
        # 3. 绘制点（使用更高的z-order确保点在顶层）
        # 落在TIFF中的点 - 蓝色
        points_in = all_results[all_results['tiff_exists']]
        if len(points_in) > 0:
            ax.scatter(points_in['longitude'], points_in['latitude'], 
                      c='blue', s=10, alpha=0.8, label='Points in TIFF', 
                      zorder=10, edgecolors='darkblue', linewidth=0.5)
        
        # 未找到TIFF的点 - 红色，更大更醒目
        points_out = all_results[~all_results['tiff_exists']]
        if len(points_out) > 0:
            ax.scatter(points_out['longitude'], points_out['latitude'], 
                      c='red', s=50, alpha=0.9, marker='*', 
                      label='Points without TIFF', zorder=11,
                      edgecolors='darkred', linewidth=1)
        
        # 设置图形属性
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Latitude', fontsize=14)
        ax.set_title(f'Points and Grid TIFF Status (Grid Borders)\n'
                     f'(Total points: {len(all_results)}, '
                     f'In TIFF: {len(points_in)}, '
                     f'Without TIFF: {len(points_out)})', 
                     fontsize=16)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        
        # 创建图例
        legend_elements = [
            mpatches.Rectangle((0, 0), 1, 1, 
                             facecolor='none', edgecolor='green', linewidth=2,
                             label=f'Processed grids ({len(results["tiffs_already_processed"])})'),
            mpatches.Rectangle((0, 0), 1, 1,
                             facecolor='none', edgecolor='orange', linewidth=2,
                             label=f'To be processed grids ({len(results["tiffs_to_process"])})'),
            plt.scatter([], [], c='blue', s=30, edgecolors='darkblue', label=f'Points in TIFF ({len(points_in)})'),
            plt.scatter([], [], c='red', s=100, marker='*', edgecolors='darkred', label=f'Points without TIFF ({len(points_out)})')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=12)
        
        # 设置坐标轴范围（基于点的分布和网格）
        if len(all_results) > 0:
            # 计算所有网格的边界
            all_lons = []
            all_lats = []
            
            # 添加点的坐标
            all_lons.extend(all_results['longitude'].tolist())
            all_lats.extend(all_results['latitude'].tolist())
            
            # 添加网格的边界坐标
            for tiff_name in tiffs_to_process + tiffs_already_processed:
                center_lon, center_lat = self.parse_tiff_filename(tiff_name)
                if center_lon is not None and center_lat is not None:
                    bounds = self.get_grid_bounds_from_center(center_lon, center_lat)
                    all_lons.extend([bounds['left'], bounds['right']])
                    all_lats.extend([bounds['bottom'], bounds['top']])
            
            if all_lons and all_lats:
                lon_min, lon_max = min(all_lons), max(all_lons)
                lat_min, lat_max = min(all_lats), max(all_lats)
                
                lon_range = lon_max - lon_min
                lat_range = lat_max - lat_min
                buffer = 0.1  # 10%的缓冲区
                
                ax.set_xlim(lon_min - lon_range*buffer, lon_max + lon_range*buffer)
                ax.set_ylim(lat_min - lat_range*buffer, lat_max + lat_range*buffer)
        
        # 保存图形
        output_file = os.path.join(output_dir, "grid_visualization_with_borders.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n已保存可视化图到: {output_file}")
        
        # 如果有未找到TIFF的点，创建一个专门的放大图
        if len(points_out) > 0:
            fig2, ax2 = plt.subplots(1, 1, figsize=(15, 10))
            
            # 加载底图
            if world is not None:
                world.plot(ax=ax2, color='lightgray', edgecolor='darkgray', linewidth=0.5)
            
            # 只显示未找到TIFF的点的区域
            ax2.scatter(points_out['longitude'], points_out['latitude'], 
                       c='red', s=100, alpha=0.9, marker='*', 
                       edgecolors='darkred', linewidth=2)
            
            # 为每个点绘制应该存在的网格边框（虚线）
            for idx, row in points_out.iterrows():
                center_lon, center_lat = self.calculate_grid_center(row['longitude'], row['latitude'])
                bounds = self.get_grid_bounds_from_center(center_lon, center_lat)
                
                rect = Rectangle(
                    (bounds['left'], bounds['bottom']),
                    self.grid_size, self.grid_size,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none',
                    linestyle='--',
                    alpha=0.5,
                    zorder=2
                )
                ax2.add_patch(rect)
                
                # 添加标签
                ax2.annotate(f"({row['longitude']:.3f}, {row['latitude']:.3f})",
                           (row['longitude'], row['latitude']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            # 设置范围
            lon_buffer = 1.0
            lat_buffer = 1.0
            ax2.set_xlim(points_out['longitude'].min() - lon_buffer, 
                        points_out['longitude'].max() + lon_buffer)
            ax2.set_ylim(points_out['latitude'].min() - lat_buffer, 
                        points_out['latitude'].max() + lat_buffer)
            
            ax2.set_xlabel('Longitude', fontsize=14)
            ax2.set_ylabel('Latitude', fontsize=14)
            ax2.set_title(f'Points without TIFF (Total: {len(points_out)})\nDashed boxes show missing grids', 
                         fontsize=16)
            ax2.grid(True, alpha=0.3)
            
            # 添加图例
            legend_elements2 = [
                plt.scatter([], [], c='red', s=100, marker='*', edgecolors='darkred', 
                           label='Points without TIFF'),
                mpatches.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='red', 
                                 linestyle='--', label='Missing grid location')
            ]
            ax2.legend(handles=legend_elements2, loc='best', fontsize=12)
            
            output_file2 = os.path.join(output_dir, "points_without_tiff_detail.png")
            plt.tight_layout()
            plt.savefig(output_file2, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存未找到TIFF的点详细图到: {output_file2}")
        
        # 创建一个网格概览图，显示所有网格的分布
        fig3, ax3 = plt.subplots(1, 1, figsize=(20, 12))
        
        if world is not None:
            world.plot(ax=ax3, color='lightgray', edgecolor='darkgray', linewidth=0.5)
        
        # 绘制所有网格（不限制数量）
        print("\n创建网格概览图...")
        processed_count_all = self.draw_grid_borders(
            ax3, results['tiffs_already_processed'], 
            color='green', linewidth=1, linestyle='-', alpha=0.6
        )
        
        to_process_count_all = self.draw_grid_borders(
            ax3, results['tiffs_to_process'],
            color='orange', linewidth=1, linestyle='-', alpha=0.6
        )
        
        ax3.set_xlabel('Longitude', fontsize=14)
        ax3.set_ylabel('Latitude', fontsize=14)
        ax3.set_title(f'All Grid Distribution Overview\n'
                     f'(Processed: {processed_count_all}, To Process: {to_process_count_all})',
                     fontsize=16)
        ax3.grid(True, alpha=0.3)
        
        # 设置全球范围
        ax3.set_xlim(-180, 180)
        ax3.set_ylim(-90, 90)
        
        output_file3 = os.path.join(output_dir, "grid_distribution_overview.png")
        plt.tight_layout()
        plt.savefig(output_file3, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存网格分布概览图到: {output_file3}")

def main():
    # 配置参数
    csv_file = "/maps/zf281/btfm4rs/occupancy_locations.csv"
    tiff_dir = "/scratch/zf281/global_map_0.1_degree_tiff"
    shapefile_path = "/maps/zf281/btfm4rs/data/global_map_shp/ne_110m_admin_0_countries.shp"
    output_dir = "./grid_analysis_output"
    
    # 可视化时的采样大小，设置为0或负数表示使用所有TIFF
    visualization_sample_size = 200
    
    # 创建处理器
    finder = GridTiffFinder(grid_size=0.1, tiff_dir=tiff_dir)
    
    # 处理点并找到对应的TIFF
    start_time = time.time()
    results = finder.find_tiffs_for_points(csv_file)
    
    # 保存结果
    finder.save_results(results, output_dir)
    
    # 可视化结果
    finder.visualize_results(results, output_dir, shapefile_path=shapefile_path, 
                           sample_size=visualization_sample_size)
    
    end_time = time.time()
    print(f"\n总处理时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
    
    # 创建处理器
    finder = GridTiffFinder(grid_size=0.1, tiff_dir=tiff_dir)
    
    # 处理点并找到对应的TIFF
    start_time = time.time()
    results = finder.find_tiffs_for_points(csv_file)
    
    # 保存结果
    finder.save_results(results, output_dir)
    
    # 可视化结果
    finder.visualize_results(results, output_dir, shapefile_path=shapefile_path, 
                           sample_size=visualization_sample_size)
    
    end_time = time.time()
    print(f"\n总处理时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()