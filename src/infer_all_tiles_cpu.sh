#!/bin/bash
#################请在src上级目录提交#####################
# bash src/infer_all_tiles_cpu.sh

# Configuration
TILES_DIR="/scratch/zf281/jovana/retiled_d_pixel"
OUTPUT_DIR="/scratch/zf281/jovana/representation_retiled"
CONFIG_FILE="configs/multi_tile_infer_config_cpu.py"
PYTHON_SCRIPT="src/multi_tile_infer_cpu.py"
export PYTHON_ENV="/maps/zf281/btfm-training-frank/venv/bin/python"

# Max concurrent processes (change this value to control parallelism)
MAX_CONCURRENT_PROCESSES=10

# Calculate available CPU cores
TOTAL_CPU_CORES=$(nproc)  # Get total number of CPU cores
AVAILABLE_CORES=$((TOTAL_CPU_CORES - 8))  # Reserve 1 core for system
echo "$(date): Total CPU cores: $TOTAL_CPU_CORES, Available for processing: $AVAILABLE_CORES"

# Array to store process PIDs
declare -a PIDS

# Function to kill process and all its children
kill_process_tree() {
    local pid=$1
    echo "$(date): Killing process tree for PID: $pid"
    
    # Get all child processes
    local children=$(pgrep -P $pid)
    
    # Kill children first (recursively)
    for child in $children; do
        kill_process_tree $child
    done
    
    # Kill the process
    if ps -p $pid > /dev/null; then
        echo "$(date): Terminating process $pid"
        kill -9 $pid 2>/dev/null
    fi
}

# Function to clean up processes on exit
cleanup() {
    echo "$(date): Received interrupt signal. Cleaning up processes..."
    
    # First try to kill each process tree
    for pid in "${PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            kill_process_tree $pid
        fi
    done
    
    echo "$(date): All processes terminated. Exiting."
    exit 1
}

# Set up trap for Ctrl+C (SIGINT)
trap cleanup SIGINT SIGTERM

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"  # Make sure logs directory exists

echo "$(date): Starting inference with up to $MAX_PROCESSES CPU processes"
echo "$(date): Scanning for tiles in $TILES_DIR"

# Find all tile directories
TILES=($(find "$TILES_DIR" -maxdepth 1 -mindepth 1 -type d | sort))
NUM_TILES=${#TILES[@]}

echo "$(date): Found $NUM_TILES tiles to process"
echo "$(date): Tile list (first 5):"
for ((i=0; i<5 && i<NUM_TILES; i++)); do
    echo "  - ${TILES[$i]}"
done

# Determine how many processes to use (min of tiles, max concurrent processes, and available cores)
NUM_PROCESSES=$(( NUM_TILES < MAX_CONCURRENT_PROCESSES ? NUM_TILES : MAX_CONCURRENT_PROCESSES ))
echo "$(date): Will use $NUM_PROCESSES processes at a time"

# Calculate CPU threads per process
THREADS_PER_PROCESS=$(( AVAILABLE_CORES / NUM_PROCESSES ))
if [ $THREADS_PER_PROCESS -lt 1 ]; then
    THREADS_PER_PROCESS=1
fi
echo "$(date): Each process will use $THREADS_PER_PROCESS CPU threads"

# Function to process tiles in batches
process_tiles_batch() {
    local start_idx=$1
    local end_idx=$2
    local num_processes=$3
    local threads_per_process=$4
    
    echo "$(date): Starting batch processing from tile $start_idx to $end_idx"
    
    # Clear PIDs array
    PIDS=()
    
    # Launch processes for this batch
    for ((i=start_idx; i<end_idx && i<NUM_TILES; i++)); do
        tile_path="${TILES[$i]}"
        tile_name=$(basename "$tile_path")
        process_id=$i
        
        echo "$(date): Starting process for tile $i: $tile_name (PID will use $threads_per_process threads)"
        
        # Make sure python script exists
        if [ ! -f "$PYTHON_SCRIPT" ]; then
            echo "$(date): ERROR: Python script not found: $PYTHON_SCRIPT"
            exit 1
        fi
        
        # Make sure config file exists
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "$(date): ERROR: Config file not found: $CONFIG_FILE"
            exit 1
        fi
        
        # Launch the process in the background
        $PYTHON_ENV "$PYTHON_SCRIPT" \
            --config "$CONFIG_FILE" \
            --tile_path "$tile_path" \
            --process_id $process_id \
            --num_threads $threads_per_process 2>&1 | tee "logs/infer_process_${process_id}.log" &
        
        PID=$!
        echo "$(date): Process for tile $i started with PID $PID"
        PIDS+=($PID)
    done
    
    # Wait for all processes in this batch to complete
    echo "$(date): Waiting for batch processes to complete..."
    
    for pid in "${PIDS[@]}"; do
        wait $pid
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "$(date): Process with PID $pid completed successfully"
        else
            echo "$(date): Process with PID $pid failed with exit code $EXIT_CODE"
        fi
    done
    
    echo "$(date): Batch processing completed"
}

# Process tiles in batches of NUM_PROCESSES
for ((batch_start=0; batch_start<NUM_TILES; batch_start+=NUM_PROCESSES)); do
    batch_end=$((batch_start + NUM_PROCESSES))
    process_tiles_batch $batch_start $batch_end $NUM_PROCESSES $THREADS_PER_PROCESS
done

echo "$(date): All inference processes completed"

# Check if all output files were created
EXPECTED_FILES=$NUM_TILES
ACTUAL_FILES=$(find "$OUTPUT_DIR" -name "*.npy" | wc -l)

echo "$(date): Expected $EXPECTED_FILES output files, found $ACTUAL_FILES"

if [ $ACTUAL_FILES -eq $EXPECTED_FILES ]; then
    echo "$(date): All tiles were processed successfully!"
else
    echo "$(date): WARNING: Some tiles may not have been processed correctly"
    echo "$(date): Missing $(($EXPECTED_FILES - $ACTUAL_FILES)) output files"
fi

echo "$(date): Inference completed"