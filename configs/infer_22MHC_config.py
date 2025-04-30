from copy import deepcopy
from configs.ssl_config import config

# Copy base configuration to avoid modifying it directly
config = deepcopy(config)

# Update or add configuration for distributed inference
config.update({
    # Data path
    "tile_path": "data/downstream/clement_agb/22MHC",
    
    # Model checkpoint
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250417_101636.pt",
    
    # Inference parameters
    "repeat_times": 5,
    "min_valid_timesteps": 0,
    "batch_size": 16384,  # Increased for better GPU utilization
    "output_npy": "data/representation/clement_agb/22MHC/representations_fsdp_20250417_101636.npy",
    "num_workers": 0,  # Increased for better data loading
    
    # Model parameters
    "fusion_method": "concat",  # 'sum' or 'concat'
    "max_seq_len_s2": 40,
    "max_seq_len_s1": 40,
    "sample_size_s2": 40,
    "sample_size_s1": 40,

    "latent_dim": 128,
    "projector_hidden_dim": 8192*2,
    "projector_out_dim": 8192*2,
    
    # Distributed inference parameters
    "distributed": True,
    "world_size": 8,  # Number of GPUs
    
    # New optimized inference parameters
    "cleanup_shards": True,  # Whether to delete intermediate shard files
    "persistent_workers": True,  # Keep workers alive between batches
})
