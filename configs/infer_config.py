from copy import deepcopy
from configs.ssl_config import config

# 复制基础配置，避免直接修改基础配置对象
config = deepcopy(config)

# 更新或新增下游配置的专有内容
config.update({
    "tile_path": "/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/s2_s1_global_project/robin_fungal/2021/34VEJ",
    # "tile_path": "/home/zf281/rds/hpc-work/Files/btfm4rs/data/ssl_training/austrian_crop",
    "checkpoint_path": "checkpoints/ssl/best_model_dual_transformer_32layer_120tiles_5epoch.pt",
    "repeat_times": 10,
    "min_valid_timesteps": 0,
    "batch_size": 8192,
    "output_npy": "/home/zf281/rds/hpc-work/Files/btfm4rs/data/representation/robin_fungal/34VEJ/representation.npy",
    "num_workers": 0,
    "max_band_file_size": 10,   # 单位：GB，每个 chunk 最大允许大小
})