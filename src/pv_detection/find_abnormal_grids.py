import os
import re
import shutil
from collections import defaultdict

def get_pv_pixels(log_path):
    """
    Parses a single log file to extract the 'PV pixels detected' value.
    """
    pv_pixel_pattern = re.compile(r'^\s*PV pixels detected:\s*(\d+)\s*$')
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = pv_pixel_pattern.match(line)
                if match:
                    return int(match.group(1))
    except (IOError, ValueError) as e:
        print(f"Warning: Could not read or parse file {log_path}. Error: {e}")
    return None

def find_unstable_grids(base_dir, start_year, end_year, dest_dir):
    """
    Finds the most unstable grids across a time window based on a custom
    instability score and copies their corresponding TIFF files.
    """
    print(f"Analyzing grid stability from {start_year} to {end_year}...")
    
    # 1. 数据收集
    # 使用defaultdict简化数据结构初始化
    # 结构: {'grid_id': {'year': pixels, 'year': pixels, ...}}
    grid_time_series = defaultdict(dict)
    
    years = [str(y) for y in range(start_year, end_year + 1)]
    
    for year in years:
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            print(f"Info: Directory for year {year} not found, skipping.")
            continue
            
        print(f"Processing year {year}...")
        for log_file in os.listdir(year_path):
            if log_file.endswith('_prediction.log'):
                grid_id = log_file.replace('_prediction.log', '')
                pixels = get_pv_pixels(os.path.join(year_path, log_file))
                if pixels is not None:
                    grid_time_series[grid_id][year] = pixels

    # 2. 计算不稳定性评分
    grid_scores = []
    for grid_id, yearly_data in grid_time_series.items():
        # 只处理拥有至少两年数据的网格，以便比较
        if len(yearly_data) < 2:
            continue
            
        # 按年份对像素数据排序，形成时间序列
        sorted_years = sorted(yearly_data.keys())
        pixel_series = [yearly_data[y] for y in sorted_years]
        
        instability_score = 0
        # 遍历时间序列，计算下降差值的总和
        for i in range(1, len(pixel_series)):
            # 如果当年的像素点少于前一年
            if pixel_series[i] < pixel_series[i-1]:
                decrease = pixel_series[i-1] - pixel_series[i]
                instability_score += decrease
                
        grid_scores.append({
            'grid': grid_id,
            'score': instability_score,
            'data': yearly_data # 保存原始数据以供报告
        })
        
    if not grid_scores:
        print("Could not find enough data to calculate instability scores.")
        return

    # 3. 排序和筛选
    sorted_grids = sorted(grid_scores, key=lambda x: x['score'], reverse=True)
    
    # 4. 报告和拷贝
    print("\n--- Top 5 Most Unstable Grids (2017-2024) ---")
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Ensured destination directory exists at: {dest_dir}")
    except OSError as e:
        print(f"Fatal: Could not create destination directory {dest_dir}. Error: {e}")
        return
        
    num_to_process = min(5, len(sorted_grids))
    
    if num_to_process == 0:
        print("No unstable grids found.")
        return

    for i in range(num_to_process):
        item = sorted_grids[i]
        grid_id = item['grid']
        
        print(f"\n{i+1}. Grid Name: {grid_id}")
        print(f"   - Instability Score: {item['score']}")
        
        # 打印详细的年度数据
        print("   - Yearly PV Pixels:")
        for year in sorted(item['data'].keys()):
            print(f"     {year}: {item['data'][year]}")

        # 拷贝这个不稳定网格的所有年份的TIFF文件
        print("   - Copying associated TIFF files...")
        tiff_filename = f"{grid_id}_prediction.tiff"
        for year in years:
            source_path = os.path.join(base_dir, year, tiff_filename)
            if os.path.exists(source_path):
                dest_filename = f"{year}_{tiff_filename}"
                dest_path = os.path.join(dest_dir, dest_filename)
                try:
                    shutil.copy(source_path, dest_path)
                    print(f"     -> Copied and renamed: {dest_filename}")
                except (shutil.Error, IOError) as e:
                    print(f"     -> ERROR: Failed to copy {source_path}. Error: {e}")
        print("-" * 40)


if __name__ == '__main__':
    # 定义根目录和目标目录
    prediction_root_dir = '/maps/zf281/btfm4rs/data/downstream/pv_detection/uk_prediction'
    dodgy_grids_dir = '/maps/zf281/btfm4rs/src/pv_detection/dodgy_grids'
    
    # 定义分析的时间窗口
    start_year = 2017
    end_year = 2024
    
    # 运行主函数
    find_unstable_grids(prediction_root_dir, start_year, end_year, dodgy_grids_dir)