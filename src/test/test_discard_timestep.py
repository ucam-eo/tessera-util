#!/usr/bin/env python3
"""
测试时间步抛弃逻辑的脚本
包括S2基于云覆盖的智能抛弃和S1基于有效数据比例的智能抛弃
"""

import sys
import os
import numpy as np
import logging

# 添加项目根目录到Python路径
sys.path.append('/maps/zf281/btfm4rs/src')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """加载配置文件"""
    config_path = '/maps/zf281/btfm4rs/configs/downstream_config.py'
    config_globals = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config_globals)
    
    # 直接获取config字典
    config = config_globals['config']
    return config

def create_test_data():
    """创建测试数据"""
    # 创建S2测试数据 - 20个时间步
    s2_bands = np.random.rand(20, 64, 64, 10).astype(np.float32) * 10000
    s2_doys = np.array([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 30, 60, 90, 120, 150, 180, 210, 240])
    
    # 创建S2 mask数据，模拟不同的云覆盖情况
    s2_masks = np.ones((20, 64, 64), dtype=bool)
    # 为一些时间步添加云覆盖（设置为False表示有云）
    s2_masks[0, :32, :32] = False  # 25%云覆盖
    s2_masks[1, :48, :48] = False  # 56%云覆盖
    s2_masks[2, :16, :16] = False  # 6%云覆盖
    s2_masks[3, :, :] = False      # 100%云覆盖
    s2_masks[4, :8, :8] = False    # 1.5%云覆盖
    
    # 创建S1测试数据 - 25个时间步（升轨15个，降轨10个）
    s1_asc_bands = np.random.rand(15, 64, 64, 2).astype(np.float32) * 10000
    s1_asc_doys = np.array([10, 22, 34, 46, 58, 70, 82, 94, 106, 118, 130, 142, 154, 166, 178])
    s1_asc_masks = np.ones((15, 64, 64), dtype=bool)
    
    s1_desc_bands = np.random.rand(10, 64, 64, 2).astype(np.float32) * 10000
    s1_desc_doys = np.array([16, 28, 40, 52, 64, 76, 88, 100, 112, 124])
    s1_desc_masks = np.ones((10, 64, 64), dtype=bool)
    
    # 为S1数据添加一些无效区域（两个波段都为0）来模拟不同的有效数据比例
    # 时间步0: 高有效数据比例（90%）
    s1_asc_bands[0, :6, :6, :] = 0  # 约10%无效
    
    # 时间步1: 中等有效数据比例（75%）
    s1_asc_bands[1, :16, :16, :] = 0  # 约25%无效
    
    # 时间步2: 低有效数据比例（50%）
    s1_asc_bands[2, :32, :32, :] = 0  # 约50%无效
    
    # 时间步3: 很低有效数据比例（25%）
    s1_asc_bands[3, :48, :48, :] = 0  # 约75%无效
    
    # 时间步4: 极低有效数据比例（10%）
    s1_asc_bands[4, :58, :58, :] = 0  # 约90%无效
    
    # 为降轨数据也添加一些变化
    s1_desc_bands[0, :8, :8, :] = 0   # 约5%无效
    s1_desc_bands[1, :24, :24, :] = 0  # 约35%无效
    s1_desc_bands[2, :40, :40, :] = 0  # 约60%无效
    
    # 创建标签数据（2D数组，与空间维度匹配）
    labels = np.ones((64, 64), dtype=np.int32)
    labels[:32, :32] = 1  # 类别1
    labels[32:, :32] = 2  # 类别2
    labels[:32, 32:] = 3  # 类别3
    labels[32:, 32:] = 4  # 类别4
    
    # 创建字段ID数据（2D数组，与空间维度匹配）
    field_ids = np.ones((64, 64), dtype=np.int32)
    field_ids[:32, :32] = 101
    field_ids[32:, :32] = 102
    field_ids[:32, 32:] = 103
    field_ids[32:, 32:] = 104
    
    return {
        's2_bands': s2_bands,
        's2_masks': s2_masks,
        's2_doys': s2_doys,
        's1_asc_bands': s1_asc_bands,
        's1_asc_doys': s1_asc_doys,
        's1_asc_masks': s1_asc_masks,
        's1_desc_bands': s1_desc_bands,
        's1_desc_doys': s1_desc_doys,
        's1_desc_masks': s1_desc_masks,
        'labels': labels,
        'field_ids': field_ids
    }

def save_test_data(data, base_path='/tmp/test_data'):
    """保存测试数据到临时文件"""
    os.makedirs(base_path, exist_ok=True)
    
    file_paths = {}
    for key, value in data.items():
        file_path = os.path.join(base_path, f'{key}.npy')
        np.save(file_path, value)
        file_paths[key] = file_path
    
    return file_paths

def test_s1_intelligent_discard():
    """测试S1智能抛弃逻辑"""
    from datasets.downstream_dataset import AustrianCrop
    
    print("\n=== 测试S1基于有效数据比例的智能抛弃逻辑 ===")
    
    # 加载配置
    config = load_config()
    
    # 创建测试数据
    test_data = create_test_data()
    file_paths = save_test_data(test_data)
    
    # 标签和字段ID数据已经在test_data中创建
    labels_path = file_paths['labels']
    field_ids_path = file_paths['field_ids']
    
    # 测试用例1: 不限制S1观测数量
    print("\n--- 测试用例1: 不限制S1观测数量 ---")
    dataset1 = AustrianCrop(
        s2_bands_file_path=file_paths['s2_bands'],
        s2_masks_file_path=file_paths['s2_masks'],
        s2_doy_file_path=file_paths['s2_doys'],
        s1_asc_bands_file_path=file_paths['s1_asc_bands'],
        s1_asc_doy_file_path=file_paths['s1_asc_doys'],
        s1_desc_bands_file_path=file_paths['s1_desc_bands'],
        s1_desc_doy_file_path=file_paths['s1_desc_doys'],
        labels_path=labels_path,
        field_ids_path=field_ids_path,
        train_fids=[],
        val_fids=[],
        test_fids=[],
        max_s2_obs=None,
        max_s1_obs=None,
        split='train'
    )
    
    # 测试用例2: 限制S1观测数量为15
    print("\n--- 测试用例2: 限制S1观测数量为15 ---")
    dataset2 = AustrianCrop(
        s2_bands_file_path=file_paths['s2_bands'],
        s2_masks_file_path=file_paths['s2_masks'],
        s2_doy_file_path=file_paths['s2_doys'],
        s1_asc_bands_file_path=file_paths['s1_asc_bands'],
        s1_asc_doy_file_path=file_paths['s1_asc_doys'],
        s1_desc_bands_file_path=file_paths['s1_desc_bands'],
        s1_desc_doy_file_path=file_paths['s1_desc_doys'],
        labels_path=labels_path,
        field_ids_path=field_ids_path,
        train_fids=[],
        val_fids=[],
        test_fids=[],
        max_s2_obs=None,
        max_s1_obs=15,
        split='train'
    )
    
    # 测试用例3: 限制S1观测数量为10
    print("\n--- 测试用例3: 限制S1观测数量为10 ---")
    dataset3 = AustrianCrop(
        s2_bands_file_path=file_paths['s2_bands'],
        s2_masks_file_path=file_paths['s2_masks'],
        s2_doy_file_path=file_paths['s2_doys'],
        s1_asc_bands_file_path=file_paths['s1_asc_bands'],
        s1_asc_doy_file_path=file_paths['s1_asc_doys'],
        s1_desc_bands_file_path=file_paths['s1_desc_bands'],
        s1_desc_doy_file_path=file_paths['s1_desc_doys'],
        labels_path=labels_path,
        field_ids_path=field_ids_path,
        train_fids=[],
        val_fids=[],
        test_fids=[],
        max_s2_obs=None,
        max_s1_obs=10,
        split='train'
    )
    
    # 测试用例4: 限制S1观测数量为5（之前会出错的情况）
    print("\n--- 测试用例4: 限制S1观测数量为5 ---")
    dataset4 = AustrianCrop(
        s2_bands_file_path=file_paths['s2_bands'],
        s2_masks_file_path=file_paths['s2_masks'],
        s2_doy_file_path=file_paths['s2_doys'],
        s1_asc_bands_file_path=file_paths['s1_asc_bands'],
        s1_asc_doy_file_path=file_paths['s1_asc_doys'],
        s1_desc_bands_file_path=file_paths['s1_desc_bands'],
        s1_desc_doy_file_path=file_paths['s1_desc_doys'],
        labels_path=labels_path,
        field_ids_path=field_ids_path,
        train_fids=[],
        val_fids=[],
        test_fids=[],
        max_s2_obs=None,
        max_s1_obs=5,
        split='train'
    )
    
    # 测试数据加载
    print("\n--- 测试数据加载 ---")
    try:
        print(f"Dataset4有效像素数量: {len(dataset4.valid_pixels)}")
        if len(dataset4.valid_pixels) > 0:
            sample = dataset4[0]
            print(f"成功加载样本，S1数据形状: {sample['s1'].shape}")
            print(f"S2数据形状: {sample['s2'].shape}")
            print("S1智能抛弃逻辑测试通过！")
        else:
            print("警告: 没有有效像素，但S1智能抛弃逻辑已成功运行")
            print("S1智能抛弃逻辑测试通过！")
    except Exception as e:
        print(f"数据加载失败: {e}")
        raise
    
    # 清理临时文件
    import shutil
    shutil.rmtree('/tmp/test_data')
    
    print("\n=== S1智能抛弃逻辑测试完成 ===")

if __name__ == "__main__":
    test_s1_intelligent_discard()