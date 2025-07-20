from copy import deepcopy
from configs.ssl_config import config

# 复制基础配置，避免直接修改基础配置对象
config = deepcopy(config)

# 更新或新增下游配置的专有内容
config.update({
    "tile_path": "data/downstream/london/2024",
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250407_195912.pt",
    "repeat_times": 1,
    "min_valid_timesteps": 0,
    "batch_size": 1024,
    "output_npy": "data/downstream/london/2024_fsdp_20250407_195912.npy",
    "num_workers": 0,
    "projector_hidden_dim": 32768,
    "projector_out_dim": 32768,
    "sample_size_s2": 40,
    "sample_size_s1": 40,
})