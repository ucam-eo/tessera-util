# configs/pastis_patch_infer_config.py

from copy import deepcopy
from configs.ssl_config import config

config = deepcopy(config)

# 以下为示例默认配置，请根据实际情况进行修改
config.update({
    "patch_root": "/scratch/zf281/pastis/pastis_patch_d-pixel",
    "checkpoint_path": "checkpoints/ssl/best_model.pt",
    "output_dir": "/scratch/zf281/pastis/representation/dual_transformer_120tiles_20epochs",
    "repeat_times": 10,
    "min_valid_timesteps": 0,
    "batch_size": 8192,
    "num_workers": 0,
    "sample_size_s2": 20,
    "sample_size_s1": 20,
})
