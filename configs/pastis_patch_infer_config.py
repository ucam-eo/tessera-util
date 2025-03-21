# configs/pastis_patch_infer_config.py

from copy import deepcopy
from configs.ssl_config import config

config = deepcopy(config)

# 以下为示例默认配置，请根据实际情况进行修改
config.update({
    "patch_root": "/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/s2_s1_global_project/pastis/pastis_patch_d-pixel",
    "checkpoint_path": "checkpoints/ssl/best_model_20250315_222553.pt",
    "output_dir": "/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/s2_s1_global_project/pastis/representation/dawn_val_acc_75946",
    "repeat_times": 10,
    "min_valid_timesteps": 0,
    "batch_size": 65536,
    "num_workers": 0,
    "sample_size_s2": 20,
    "sample_size_s1": 20,
})
