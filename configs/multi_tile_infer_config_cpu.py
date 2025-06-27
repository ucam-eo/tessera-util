from copy import deepcopy
from configs.ssl_config import config

# Copy base config to avoid modifying it directly
config = deepcopy(config)

# Update or add CPU-specific configuration
config.update({
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250408_101211.pt",
    "repeat_times": 5,
    "min_valid_timesteps": 0,
    "batch_size": 256,  # Smaller batch size for CPU
    "output_dir": "/scratch/zf281/jovana/representation_retiled",
    "num_workers": 0,  # Fewer workers for CPU
    "fusion_method": "concat",
    "latent_dim": 128,
    "projector_hidden_dim": 8192*4,
    "projector_out_dim": 8192*4,
    "sample_size_s2": 40,
    "sample_size_s1": 40
})