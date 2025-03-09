from copy import deepcopy
from configs.ssl_config import config

# 复制基础配置，避免直接修改基础配置对象
config = deepcopy(config)

# 更新或新增下游配置的专有内容
config.update({
    "tile_path": "/scratch/zf281/jovana/data_processed/11SLA",
    # "tile_path": "/scratch/zf281/jovana/data_processed/11SLA_downsampled",
    # "tile_path": "/scratch/zf281/global/30UXB",
    "checkpoint_path": "checkpoints/ssl/best_model_dual_transformer_32layer_120tiles_5epoch.pt",
    "repeat_times": 10,
    "min_valid_timesteps": 5,
    "batch_size": 32768,
    "output_npy": "/scratch/zf281/jovana/representation/11SLA_representations.npy",
    "num_workers": 0,
})