from copy import deepcopy
from configs.ssl_config import config

# Copy base config to avoid modifying it directly
config = deepcopy(config)

# Update or add downstream-specific configuration
config.update({
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250408_101211.pt",
    "repeat_times": 5,
    "min_valid_timesteps": 0,
    "batch_size": 2048,
    "output_dir": "/scratch/zf281/downstream_dataset/jovana/tile_1/representation_retiled",
    "num_workers": 8,
    "fusion_method": "concat",  # Options: 'sum' or 'concat'
    "latent_dim": 128,
    "projector_hidden_dim": 8192*4,
    "projector_out_dim": 8192*4,
    "sample_size_s2": 40,
    "sample_size_s1": 40
})