# configs/pastis_patch_infer_config.py

from copy import deepcopy
from configs.ssl_config import config

config = deepcopy(config)

# 以下为示例默认配置，请根据实际情况进行修改
config.update({
    "patch_root": "/scratch/zf281/pastis/pastis_patch_d-pixel",
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250408_101211.pt",
    "output_dir": "/scratch/zf281/pastis/representation/amd_fsdp_20250408_101211",
    "repeat_times": 10,
    "min_valid_timesteps": 0,
    "batch_size": 4096,
    "num_workers": 0,
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "max_seq_len_s2": 40,
    "max_seq_len_s1": 40,
    "num_workers": 0,
    "latent_dim": 128,
    "projector_hidden_dim": 8192*4,
    "projector_out_dim": 8192*4
})
