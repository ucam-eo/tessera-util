from copy import deepcopy
from configs.ssl_config import config

# 复制基础配置，避免直接修改基础配置对象
config = deepcopy(config)

# 更新或新增下游配置的专有内容
config.update({
    # "tile_path": "/scratch/zf281/jovana/data_processed/11SLA",
    # "tile_path": "/scratch/zf281/jovana/data_processed/11SLA_downsampled",
    # "tile_path": "/scratch/zf281/global/30UXB",
    "tile_path": "/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled/2500_3000_3000_3500",
    # "tile_path": "/scratch/zf281/downstream_dataset/london/2024/data_processed",
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250408_101211.pt",
    "repeat_times": 1,
    "min_valid_timesteps": 0,
    "batch_size": 128,
    "output_npy": "/scratch/zf281/downstream_dataset/discard_timesteps/5_100_fsdp_20250408_101211.npy",
    "num_workers": 0,
    "fusion_method": "concat",  # 可选 'sum' 或 'concat'
    "max_seq_len_s2": 40,
    "max_seq_len_s1": 40,
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "max_s2_obs": 5,  # 最大哨兵2观测数量，用于智能时间步选择
    "max_s1_obs": 100,  # 最大哨兵1观测数量，用于智能时间步选择
    "latent_dim": 128,
    "projector_hidden_dim": 8192*4,
    "projector_out_dim": 8192*4
})