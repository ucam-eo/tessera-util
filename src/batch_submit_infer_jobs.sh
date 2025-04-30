#!/bin/bash

# Define array of tile names
TILES=("22MHC")

# Base paths
BASE_INPUT_PATH="data/downstream/clement_agb"
BASE_OUTPUT_PATH="data/representation/clement_agb"

# Config directory (from root project directory)
CONFIG_DIR="configs"

# Store created config files for cleanup
CONFIG_FILES=()

# Clean up function
cleanup() {
    echo "Cleaning up temporary config files..."
    for config_file in "${CONFIG_FILES[@]}"; do
        if [ -f "$config_file" ]; then
            rm "$config_file"
            echo "Removed: $config_file"
        fi
    done
    echo "Cleanup completed."
}

# Register cleanup function for normal exit and interruptions
trap cleanup EXIT INT TERM

# Function to create config file and return its path
create_config() {
    local tile=$1
    local config_file="${CONFIG_DIR}/infer_${tile}_config.py"
    
    # Create output directory
    mkdir -p "${BASE_OUTPUT_PATH}/${tile}"
    
    # Create config file
    cat > "$config_file" << EOF
from copy import deepcopy
from configs.ssl_config import config

# Copy base configuration to avoid modifying it directly
config = deepcopy(config)

# Update or add configuration for distributed inference
config.update({
    # Data path
    "tile_path": "${BASE_INPUT_PATH}/${tile}",
    
    # Model checkpoint
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250417_101636.pt",
    
    # Inference parameters
    "repeat_times": 5,
    "min_valid_timesteps": 0,
    "batch_size": 16384,  # Increased for better GPU utilization
    "output_npy": "${BASE_OUTPUT_PATH}/${tile}/representations_fsdp_20250417_101636.npy",
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
EOF
    
    # Add to list for cleanup
    CONFIG_FILES+=("$config_file")
    
    # Return only the config file path, without any echo statements
    echo "$config_file"
}

# Main loop
total_tiles=${#TILES[@]}
current=1

echo "Starting tile processing pipeline from $(pwd)"
echo "Config files will be created in ${CONFIG_DIR}"

for tile in "${TILES[@]}"; do
    echo "=========================================================="
    echo "Processing tile $current/$total_tiles: $tile"
    echo "Starting at: $(date)"
    echo "=========================================================="
    
    # Create config for this tile (without capturing any outputs)
    config_path=$(create_config "$tile")
    echo "Created config file: $config_path"
    
    # Run inference
    echo "Starting inference for $tile..."
    torchrun --nproc_per_node=5 --master_addr=127.0.0.1 --master_port=29500 src/infer_multi_gpu.py --config "$config_path"
    
    # Check if inference completed successfully
    if [ $? -eq 0 ]; then
        echo "Inference completed successfully for $tile"
    else
        echo "ERROR: Inference failed for $tile with exit code $?"
        exit 1
    fi
    
    echo "Completed processing for $tile at $(date)"
    echo "Progress: $current/$total_tiles tiles completed"
    echo ""
    
    ((current++))
done

echo "All $total_tiles tiles processed successfully!"
exit 0