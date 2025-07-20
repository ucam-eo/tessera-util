# configs/downstream_regression_config.py

config = {
    # SSL 模型 checkpoint 路径（请确保该 checkpoint 与当前模型结构匹配）
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250407_195912.pt",
    
    # 数据文件路径（50NNL_subset 文件夹下的各个 numpy 文件）
    "s2_bands_file_path": "data/downstream/borneo/50NNL_subset/bands.npy",
    "s2_masks_file_path": "data/downstream/borneo/50NNL_subset/masks.npy",
    "s2_doy_file_path": "data/downstream/borneo/50NNL_subset/doys.npy",
    "s1_asc_bands_file_path": "data/downstream/borneo/50NNL_subset/sar_ascending.npy",
    "s1_asc_doy_file_path": "data/downstream/borneo/50NNL_subset/sar_ascending_doy.npy",
    "s1_desc_bands_file_path": "data/downstream/borneo/50NNL_subset/sar_descending.npy",
    "s1_desc_doy_file_path": "data/downstream/borneo/50NNL_subset/sar_descending_doy.npy",
    # CHM 文件路径（形状 (1, H, W)），部分区域值为 NaN
    "chm_path": "data/downstream/borneo/Danum_2020_chm_lspikefree_subsampled.npy",
    
    # 数据拆分比例：训练集占总样本 50%，剩余 50%中按比例分配验证和测试（例如 1/7 为验证，其余为测试）
    "train_ratio": 0.3,
    "val_ratio": 1/7,  # 剩余部分中验证所占比例
    
    # 训练超参数
    "batch_size": 128,
    "epochs": 30,
    "lr": 0.001,
    "weight_decay": 0.01,
    
    # 模型相关参数
    "fusion_method": "concat",   # 可选 "sum" 或 "concat"
    "sample_size_s2": 40,     # 固定时间步数（光学）
    "sample_size_s1": 40,     # 固定时间步数（SAR，采自拼接后的数据）
    "latent_dim": 128,
    "s2_num_channels": 10,    # 输入给 S2 Transformer 的通道数
    "s1_num_channels": 2,     # 输入给 S1 Transformer 的通道数
    "projector_hidden_dim": 8192*4,
    "projector_out_dim": 8192*4,
    
    # 其他设置
    "standardize": True,
    "min_valid_timesteps": 0,
    "num_workers": 12
}
