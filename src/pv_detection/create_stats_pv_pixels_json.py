import os
import json # 引入json模块来处理json文件

def analyze_pv_logs_from_json(root_directory, outlier_threshold=10000):
    """
    Analyzes PV detection JSON logs to calculate the total PV pixels,
    excluding any grids that exceed a specified outlier threshold. It then
    reports these outliers at the end.

    This version is adapted for JSON files from the xgboost prediction pipeline.

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

    # 创建一个列表来存储所有被识别为异常值的网格
    outlier_grids = []

    # --- 重构：不再需要正则表达式 ---

    # Process each year folder
    for year in year_folders:
        year_path = os.path.join(root_directory, year)
        total_pv_pixels = 0
        grids_with_pv = 0

        try:
            # --- 重构：查找以 _xgboost_prediction_log.json 结尾的文件 ---
            json_files = [f for f in os.listdir(year_path) if f.endswith('_xgboost_prediction_log.json')]
        except OSError as e:
            print(f"Warning: Could not read directory {year_path}. Skipping. Error: {e}")
            continue

        if not json_files:
            print(f"Warning: No JSON log files found for the year {year}. Skipping.")
            continue

        for json_file in json_files:
            json_path = os.path.join(year_path, json_file)
            try:
                # --- 重构：打开并加载JSON文件 ---
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # --- 重构：从JSON结构中直接提取数据 ---
                # 这种方式比从文件名解析更可靠
                pv_pixels = data['statistics']['pv_pixels']
                grid_name = data['grid_info']['grid_name']
                
                # 检查是否为异常值 (这部分逻辑保持不变)
                if pv_pixels > outlier_threshold:
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

            # 捕获可能发生的错误：文件不是有效的JSON，或者缺少必要的键
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not process file {json_path}. Error: {e}")

        # 打印当年的统计结果（已排除异常值）
        print(f"--- Results for Year: {year} ---")
        print(f"  Total PV pixels detected (excluding outliers): {total_pv_pixels:,}")
        print(f"  Number of grids with PV pixels (count > 0, excluding outliers): {grids_with_pv}")
        print("-" * 30 + "\n")

    # 在所有年份处理完毕后，打印异常值报告 (这部分逻辑保持不变)
    print("=" * 50)
    if outlier_grids:
        print(f"\n--- Detected Outlier Grids (PV Pixels > {outlier_threshold:,}) ---")
        sorted_outliers = sorted(outlier_grids, key=lambda x: (x['year'], -x['pixels']))
        for outlier in sorted_outliers:
            print(f"  - Year: {outlier['year']}, Grid: {outlier['grid']}, Pixels: {outlier['pixels']:,}")
    else:
        print("\nNo outlier grids were detected across all years.")
    print("=" * 50)

if __name__ == '__main__':
    # --- 更新：定义新的根目录路径 ---
    prediction_root_dir = '/maps/zf281/btfm4rs/data/downstream/pv_detection/uk_prediction_xgboost'
    
    # 运行分析函数，可以自定义异常值阈值
    # 根据您上次的代码，阈值设置为10000
    analyze_pv_logs_from_json(prediction_root_dir, outlier_threshold=30000)