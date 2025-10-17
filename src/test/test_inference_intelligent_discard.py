#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试推理阶段的智能时间步选择功能
"""

import os
import sys
import numpy as np
import tempfile
import shutil
import logging

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.ssl_dataset import SingleTileInferenceDataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_inference_data(temp_dir, num_s2_obs=50, num_s1_asc_obs=30, num_s1_desc_obs=25, H=100, W=100):
    """
    创建用于推理测试的合成数据
    """
    # 创建S2数据 - 不同的云覆盖率
    s2_bands = np.random.rand(num_s2_obs, H, W, 10).astype(np.float32) * 1000
    s2_masks = np.random.rand(num_s2_obs, H, W).astype(np.int32)
    
    # 为前20个时间步设置高云覆盖率（低质量）
    s2_masks[:20] = np.random.choice([0, 1], size=(20, H, W), p=[0.8, 0.2])  # 80%云覆盖
    # 为后30个时间步设置低云覆盖率（高质量）
    s2_masks[20:] = np.random.choice([0, 1], size=(30, H, W), p=[0.2, 0.8])  # 20%云覆盖
    
    s2_doys = np.arange(1, num_s2_obs + 1).astype(np.int32)
    
    # 创建S1数据 - 不同的有效数据比例
    s1_asc_bands = np.random.rand(num_s1_asc_obs, H, W, 2).astype(np.float32) * 100
    s1_desc_bands = np.random.rand(num_s1_desc_obs, H, W, 2).astype(np.float32) * 100
    
    # 为前15个S1 asc时间步设置低有效数据比例
    for i in range(15):
        mask = np.random.choice([True, False], size=(H, W), p=[0.7, 0.3])  # 70%无效数据
        s1_asc_bands[i][mask] = 0
    
    # 为后15个S1 asc时间步设置高有效数据比例
    for i in range(15, num_s1_asc_obs):
        mask = np.random.choice([True, False], size=(H, W), p=[0.2, 0.8])  # 20%无效数据
        s1_asc_bands[i][mask] = 0
    
    # 为前12个S1 desc时间步设置低有效数据比例
    for i in range(12):
        mask = np.random.choice([True, False], size=(H, W), p=[0.8, 0.2])  # 80%无效数据
        s1_desc_bands[i][mask] = 0
    
    # 为后13个S1 desc时间步设置高有效数据比例
    for i in range(12, num_s1_desc_obs):
        mask = np.random.choice([True, False], size=(H, W), p=[0.1, 0.9])  # 10%无效数据
        s1_desc_bands[i][mask] = 0
    
    s1_asc_doys = np.arange(1, num_s1_asc_obs + 1).astype(np.int32)
    s1_desc_doys = np.arange(1, num_s1_desc_obs + 1).astype(np.int32)
    
    # 保存数据
    np.save(os.path.join(temp_dir, "bands.npy"), s2_bands)
    np.save(os.path.join(temp_dir, "masks.npy"), s2_masks)
    np.save(os.path.join(temp_dir, "doys.npy"), s2_doys)
    np.save(os.path.join(temp_dir, "sar_ascending.npy"), s1_asc_bands)
    np.save(os.path.join(temp_dir, "sar_ascending_doy.npy"), s1_asc_doys)
    np.save(os.path.join(temp_dir, "sar_descending.npy"), s1_desc_bands)
    np.save(os.path.join(temp_dir, "sar_descending_doy.npy"), s1_desc_doys)
    
    return {
        'num_s2_obs': num_s2_obs,
        'num_s1_asc_obs': num_s1_asc_obs,
        'num_s1_desc_obs': num_s1_desc_obs,
        'H': H,
        'W': W
    }

def test_inference_intelligent_discard():
    """
    测试推理阶段的智能时间步选择功能
    """
    print("开始测试推理阶段的智能时间步选择功能...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建测试数据
        data_info = create_test_inference_data(temp_dir)
        print(f"创建测试数据: S2={data_info['num_s2_obs']}, S1_asc={data_info['num_s1_asc_obs']}, S1_desc={data_info['num_s1_desc_obs']}")
        
        # 测试1: 无限制（应该保留所有时间步）
        print("\n测试1: 无限制的时间步选择")
        dataset1 = SingleTileInferenceDataset(
            tile_path=temp_dir,
            min_valid_timesteps=5,
            max_s2_obs=None,
            max_s1_obs=None
        )
        print(f"无限制: S2={dataset1.s2_bands.shape[0]}, S1_asc={dataset1.s1_asc_bands.shape[0]}, S1_desc={dataset1.s1_desc_bands.shape[0]}")
        
        # 测试2: 限制S2观测数量
        print("\n测试2: 限制S2观测数量为30")
        dataset2 = SingleTileInferenceDataset(
            tile_path=temp_dir,
            min_valid_timesteps=5,
            max_s2_obs=30,
            max_s1_obs=None
        )
        print(f"限制S2=30: S2={dataset2.s2_bands.shape[0]}, S1_asc={dataset2.s1_asc_bands.shape[0]}, S1_desc={dataset2.s1_desc_bands.shape[0]}")
        assert dataset2.s2_bands.shape[0] <= 30, f"S2观测数量应该<=30，实际为{dataset2.s2_bands.shape[0]}"
        
        # 测试3: 限制S1观测数量
        print("\n测试3: 限制S1观测数量为40")
        dataset3 = SingleTileInferenceDataset(
            tile_path=temp_dir,
            min_valid_timesteps=5,
            max_s2_obs=None,
            max_s1_obs=40
        )
        total_s1_obs = dataset3.s1_asc_bands.shape[0] + dataset3.s1_desc_bands.shape[0]
        print(f"限制S1=40: S2={dataset3.s2_bands.shape[0]}, S1_asc={dataset3.s1_asc_bands.shape[0]}, S1_desc={dataset3.s1_desc_bands.shape[0]}, 总S1={total_s1_obs}")
        assert total_s1_obs <= 40, f"S1总观测数量应该<=40，实际为{total_s1_obs}"
        
        # 测试4: 同时限制S2和S1
        print("\n测试4: 同时限制S2=25和S1=30")
        dataset4 = SingleTileInferenceDataset(
            tile_path=temp_dir,
            min_valid_timesteps=5,
            max_s2_obs=25,
            max_s1_obs=30
        )
        total_s1_obs_4 = dataset4.s1_asc_bands.shape[0] + dataset4.s1_desc_bands.shape[0]
        print(f"同时限制: S2={dataset4.s2_bands.shape[0]}, S1_asc={dataset4.s1_asc_bands.shape[0]}, S1_desc={dataset4.s1_desc_bands.shape[0]}, 总S1={total_s1_obs_4}")
        assert dataset4.s2_bands.shape[0] <= 25, f"S2观测数量应该<=25，实际为{dataset4.s2_bands.shape[0]}"
        assert total_s1_obs_4 <= 30, f"S1总观测数量应该<=30，实际为{total_s1_obs_4}"
        
        # 测试5: 极小限制（测试边界情况）
        print("\n测试5: 极小限制S2=5和S1=10")
        dataset5 = SingleTileInferenceDataset(
            tile_path=temp_dir,
            min_valid_timesteps=3,
            max_s2_obs=5,
            max_s1_obs=10
        )
        total_s1_obs_5 = dataset5.s1_asc_bands.shape[0] + dataset5.s1_desc_bands.shape[0]
        print(f"极小限制: S2={dataset5.s2_bands.shape[0]}, S1_asc={dataset5.s1_asc_bands.shape[0]}, S1_desc={dataset5.s1_desc_bands.shape[0]}, 总S1={total_s1_obs_5}")
        assert dataset5.s2_bands.shape[0] <= 5, f"S2观测数量应该<=5，实际为{dataset5.s2_bands.shape[0]}"
        assert total_s1_obs_5 <= 10, f"S1总观测数量应该<=10，实际为{total_s1_obs_5}"
        
        # 验证有效像素数量
        print(f"\n有效像素数量: dataset1={len(dataset1.valid_pixels)}, dataset2={len(dataset2.valid_pixels)}, dataset3={len(dataset3.valid_pixels)}, dataset4={len(dataset4.valid_pixels)}, dataset5={len(dataset5.valid_pixels)}")
        
        # 测试数据加载
        if len(dataset4.valid_pixels) > 0:
            sample = dataset4[0]
            print(f"样本数据形状: s2_bands={sample['s2_bands'].shape}, s1_asc_bands={sample['s1_asc_bands'].shape}, s1_desc_bands={sample['s1_desc_bands'].shape}")
        else:
            print("警告: dataset4没有有效像素，跳过样本测试")
        
        print("\n✅ 推理阶段智能时间步选择功能测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"清理临时目录: {temp_dir}")

if __name__ == "__main__":
    test_inference_intelligent_discard()