import os
import numpy as np
from pathlib import Path
import json

def find_gaps(doys, gap_threshold=45):
    """
    查找doys中的空窗期
    
    Args:
        doys: day of year数组
        gap_threshold: 认定为空窗期的天数阈值
    
    Returns:
        gaps: 空窗期列表，每个元素为(start_doy, end_doy, gap_days)
    """
    if len(doys) == 0:
        return []
    
    # 排序doys
    sorted_doys = np.sort(doys)
    gaps = []
    
    # 检查连续的doy之间的间隔
    for i in range(len(sorted_doys) - 1):
        current_doy = sorted_doys[i]
        next_doy = sorted_doys[i + 1]
        
        # 计算间隔天数
        gap = next_doy - current_doy
        
        if gap > gap_threshold:
            gaps.append({
                'start_doy': int(current_doy),
                'end_doy': int(next_doy),
                'gap_days': int(gap)
            })
    
    return gaps

def check_tile_integrity(base_path):
    """
    检查tiles文件夹中所有子文件夹的数据完整性
    
    Args:
        base_path: tiles文件夹的路径
    """
    base_path = Path(base_path)
    
    # 必需的9个文件
    required_files = [
        'band_mean.npy',
        'band_std.npy', 
        'bands.npy',
        'doys.npy',
        'masks.npy',
        'sar_ascending.npy',
        'sar_ascending_doy.npy',
        'sar_descending.npy',
        'sar_descending_doy.npy'
    ]
    
    # 需要检查形状的文件
    shape_check_files = ['bands.npy', 'masks.npy', 'sar_ascending.npy', 'sar_descending.npy']
    
    problematic_folders = []
    
    # 遍历所有子文件夹
    for folder in sorted(base_path.iterdir()):
        if not folder.is_dir():
            continue
            
        folder_name = folder.name
        problems = []
        
        # 1. 检查是否有9个必需的文件
        existing_files = set(f.name for f in folder.iterdir() if f.is_file())
        missing_files = set(required_files) - existing_files
        
        if missing_files:
            problems.append(f"缺少文件: {', '.join(missing_files)}")
        
        # 2. 检查特定文件的形状
        file_shapes = {}
        for file_name in shape_check_files:
            file_path = folder / file_name
            if file_path.exists():
                try:
                    # 使用memmap读取形状，不加载实际数据
                    arr = np.load(file_path, mmap_mode='r')
                    file_shapes[file_name] = arr.shape
                except Exception as e:
                    problems.append(f"无法读取 {file_name}: {str(e)}")
        
        # 3. 检查bands和masks的时间维度
        if 'bands.npy' in file_shapes:
            if file_shapes['bands.npy'][0] == 0:
                problems.append("bands.npy 时间维度为0")
        
        if 'masks.npy' in file_shapes:
            if file_shapes['masks.npy'][0] == 0:
                problems.append("masks.npy 时间维度为0")
        
        # 4. 检查sar_ascending和sar_descending的时间维度
        sar_asc_t = file_shapes.get('sar_ascending.npy', (0,))[0] if 'sar_ascending.npy' in file_shapes else None
        sar_desc_t = file_shapes.get('sar_descending.npy', (0,))[0] if 'sar_descending.npy' in file_shapes else None
        
        if sar_asc_t is not None and sar_desc_t is not None:
            if sar_asc_t == 0 and sar_desc_t == 0:
                problems.append("sar_ascending和sar_descending时间维度都为0")
        
        # 5. 检查S2的doys空窗期
        s2_doys_path = folder / 'doys.npy'
        if s2_doys_path.exists():
            try:
                s2_doys = np.load(s2_doys_path)
                s2_gaps = find_gaps(s2_doys, gap_threshold=45)
                if s2_gaps:
                    gap_strs = []
                    for gap in s2_gaps:
                        gap_strs.append(f"DOY {gap['start_doy']}到DOY {gap['end_doy']}({gap['gap_days']}天)")
                    problems.append(f"S2存在超过45天的空窗期: {'; '.join(gap_strs)}")
            except Exception as e:
                problems.append(f"无法读取S2 doys: {str(e)}")
        
        # 6. 检查S1的doys空窗期（合并ascending和descending）
        s1_asc_doys_path = folder / 'sar_ascending_doy.npy'
        s1_desc_doys_path = folder / 'sar_descending_doy.npy'
        
        s1_doys_combined = []
        if s1_asc_doys_path.exists():
            try:
                s1_asc_doys = np.load(s1_asc_doys_path)
                s1_doys_combined.extend(s1_asc_doys)
            except Exception as e:
                problems.append(f"无法读取S1 ascending doys: {str(e)}")
        
        if s1_desc_doys_path.exists():
            try:
                s1_desc_doys = np.load(s1_desc_doys_path)
                s1_doys_combined.extend(s1_desc_doys)
            except Exception as e:
                problems.append(f"无法读取S1 descending doys: {str(e)}")
        
        if s1_doys_combined:
            # 去重并排序
            s1_doys_unique = np.unique(s1_doys_combined)
            s1_gaps = find_gaps(s1_doys_unique, gap_threshold=45)
            if s1_gaps:
                gap_strs = []
                for gap in s1_gaps:
                    gap_strs.append(f"DOY {gap['start_doy']}到DOY {gap['end_doy']}({gap['gap_days']}天)")
                problems.append(f"S1存在超过45天的空窗期: {'; '.join(gap_strs)}")
        
        # 如果有问题，记录下来
        if problems:
            problematic_folders.append({
                'folder': folder_name,
                'problems': problems,
                'shapes': file_shapes
            })
    
    # 打印结果
    if problematic_folders:
        print(f"\n发现 {len(problematic_folders)} 个有问题的文件夹:\n")
        print("-" * 80)
        
        for item in problematic_folders:
            print(f"\n文件夹: {item['folder']}")
            print("问题:")
            for problem in item['problems']:
                print(f"  - {problem}")
            
            if item['shapes']:
                print("读取到的形状:")
                for file_name, shape in item['shapes'].items():
                    print(f"  - {file_name}: {shape}")
    else:
        print("\n所有文件夹都通过了完整性检查！")
    
    # 统计信息
    total_folders = sum(1 for f in base_path.iterdir() if f.is_dir())
    print(f"\n统计: 总共检查了 {total_folders} 个文件夹，其中 {len(problematic_folders)} 个有问题")
    
    # 打印空窗期统计
    s2_gap_count = 0
    s1_gap_count = 0
    for item in problematic_folders:
        for problem in item['problems']:
            if 'S2存在超过45天的空窗期' in problem:
                s2_gap_count += 1
            if 'S1存在超过45天的空窗期' in problem:
                s1_gap_count += 1
    
    if s2_gap_count > 0 or s1_gap_count > 0:
        print(f"\n空窗期统计:")
        if s2_gap_count > 0:
            print(f"  - {s2_gap_count} 个文件夹存在S2空窗期")
        if s1_gap_count > 0:
            print(f"  - {s1_gap_count} 个文件夹存在S1空窗期")
    
    return problematic_folders


if __name__ == "__main__":
    # 设置tiles文件夹路径
    tiles_path = "/shared/amdgpu/home/avsm2_f4q/code/btfm4rs/data/ssl_training/tiles"
    
    # 执行检查
    problematic = check_tile_integrity(tiles_path)
    
    # 如果需要，可以将结果保存到文件
    if problematic:
        output_file = "problematic_tiles.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(problematic, f, ensure_ascii=False, indent=2)
        print(f"\n问题详情已保存到: {output_file}")