import os
import re

def analyze_pv_logs(root_directory, outlier_threshold=20000):
    """
    Analyzes PV detection logs to calculate the total PV pixels, excluding
    any grids that exceed a specified outlier threshold. It then reports
    these outliers at the end.

    Args:
        root_directory (str): The path to the directory containing year folders.
        outlier_threshold (int): The pixel count above which a grid is considered an outlier.
    """
    print(f"Starting analysis in root directory: {root_directory}")
    print(f"Outlier threshold set to: > {outlier_threshold:,} pixels\n")

    if not os.path.isdir(root_directory):
        print(f"Error: Root directory not found at '{root_directory}'")
        return

    try:
        year_folders = sorted([d for d in os.listdir(root_directory)
                               if os.path.isdir(os.path.join(root_directory, d)) and d.isdigit()])
    except OSError as e:
        print(f"Error accessing the directory: {e}")
        return

    if not year_folders:
        print("No year folders found in the specified directory.")
        return

    # --- 新增功能：创建一个列表来存储所有被识别为异常值的网格 ---
    outlier_grids = []

    pv_pixel_pattern = re.compile(r'^\s*PV pixels detected:\s*([\d,]+)\s*$')

    # Process each year folder
    for year in year_folders:
        year_path = os.path.join(root_directory, year)
        total_pv_pixels = 0
        grids_with_pv = 0

        try:
            log_files = [f for f in os.listdir(year_path) if f.endswith('_prediction.log')]
        except OSError as e:
            print(f"Warning: Could not read directory {year_path}. Skipping. Error: {e}")
            continue

        if not log_files:
            print(f"Warning: No log files found for the year {year}. Skipping.")
            continue

        for log_file in log_files:
            log_path = os.path.join(year_path, log_file)
            try:
                with open(log_path, 'r') as f:
                    for line in f:
                        match = pv_pixel_pattern.match(line)
                        if match:
                            try:
                                pixel_str_with_commas = match.group(1)
                                pixel_str_no_commas = pixel_str_with_commas.replace(',', '')
                                pv_pixels = int(pixel_str_no_commas)

                                # --- 新增功能：检查是否为异常值 ---
                                if pv_pixels > outlier_threshold:
                                    # 如果是异常值，记录下来并跳过统计
                                    grid_name = log_file.replace('_prediction.log', '')
                                    outlier_grids.append({
                                        'year': year,
                                        'grid': grid_name,
                                        'pixels': pv_pixels
                                    })
                                else:
                                    # 如果不是异常值，则加入年度统计
                                    total_pv_pixels += pv_pixels
                                    if pv_pixels > 0:
                                        grids_with_pv += 1
                                
                                break # 找到数据后即可跳出内层循环
                            except (ValueError, IndexError):
                                print(f"Warning: Could not parse PV pixels from {log_path}. Line: '{line.strip()}'")
            except IOError as e:
                print(f"Warning: Could not process file {log_path}. Error: {e}")

        # 打印当年的统计结果（已排除异常值）
        print(f"--- Results for Year: {year} ---")
        print(f"  Total PV pixels detected (excluding outliers): {total_pv_pixels:,}")
        print(f"  Number of grids with PV pixels (count > 0, excluding outliers): {grids_with_pv}")
        print("-" * 30 + "\n")

    # --- 新增功能：在所有年份处理完毕后，打印异常值报告 ---
    print("=" * 50)
    if outlier_grids:
        print(f"\n--- Detected Outlier Grids (PV Pixels > {outlier_threshold:,}) ---")
        # 按年份和像素数排序，使报告更清晰
        sorted_outliers = sorted(outlier_grids, key=lambda x: (x['year'], -x['pixels']))
        for outlier in sorted_outliers:
            print(f"  - Year: {outlier['year']}, Grid: {outlier['grid']}, Pixels: {outlier['pixels']:,}")
    else:
        print("\nNo outlier grids were detected across all years.")
    print("=" * 50)


if __name__ == '__main__':
    # 定义根目录路径
    prediction_root_dir = '/maps/zf281/btfm4rs/data/downstream/pv_detection/uk_prediction'
    
    # 运行分析函数，可以自定义异常值阈值
    analyze_pv_logs(prediction_root_dir, outlier_threshold=10000)