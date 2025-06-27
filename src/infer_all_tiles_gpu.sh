#!/bin/bash
#################请在src上级目录提交#####################

# Configuration
TILES_DIR="/scratch/zf281/downstream_dataset/jovana/tile_1/retiled_d_pixel"
OUTPUT_DIR="/scratch/zf281/downstream_dataset/jovana/tile_1/representation_retiled"
CONFIG_FILE="configs/multi_tile_infer_config_gpu.py"
PYTHON_SCRIPT="src/multi_tile_infer_gpu.py"
export PYTHON_ENV="/maps/zf281/btfm-training-frank/venv/bin/python"
NUM_GPUS=1 # how many GPUs/processes to use

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

# Function to kill all GPU processes
kill_gpu_processes() {
    echo "$(date): Checking for remaining GPU processes..."
    
    # Get all Python processes using GPU
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do
        # Check if this is one of our processes or a child of our processes
        for our_pid in "${PIDS[@]}"; do
            if [ "$pid" = "$our_pid" ] || pgrep -P $our_pid | grep -q $pid; then
                echo "$(date): Killing GPU process with PID: $pid"
                kill -9 $pid 2>/dev/null
            fi
        done
    done
    
    # Final check to make sure GPU is cleared
    echo "$(date): Current GPU processes after cleanup:"
    nvidia-smi
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
    
    # Then make sure no GPU processes are left
    kill_gpu_processes
    
    echo "$(date): All processes terminated. Exiting."
    exit 1
}

# Set up trap for Ctrl+C (SIGINT)
trap cleanup SIGINT SIGTERM

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "src/tile_lists"
mkdir -p "logs"  # Make sure logs directory exists

echo "$(date): Starting inference with $NUM_GPUS GPU(s)"
echo "$(date): Scanning for tiles in $TILES_DIR"

# Find all tile directories
TILES=($(find "$TILES_DIR" -maxdepth 1 -mindepth 1 -type d | sort))
NUM_TILES=${#TILES[@]}

echo "$(date): Found $NUM_TILES tiles to process"
echo "$(date): Tile list (first 5):"
for ((i=0; i<5 && i<NUM_TILES; i++)); do
    echo "  - ${TILES[$i]}"
done

# Divide tiles among GPUs
TILES_PER_GPU=$((NUM_TILES / NUM_GPUS))
REMAINDER=$((NUM_TILES % NUM_GPUS))

echo "$(date): Each GPU will process approximately $TILES_PER_GPU tiles"

# Create tile lists for each GPU
for ((i=0; i<NUM_GPUS; i++)); do
    # Calculate start and end indices for this GPU
    START=$((i * TILES_PER_GPU))
    END=$(((i + 1) * TILES_PER_GPU))
    
    # Add one more tile to earlier GPUs if there's a remainder
    if [ $i -lt $REMAINDER ]; then
        END=$((END + 1))
    fi
    
    # Calculate start index accounting for remainder distribution
    if [ $i -ge $REMAINDER ]; then
        START=$((START + REMAINDER))
    else
        START=$((START + i))
    fi
    
    # Create tile list for this GPU
    TILE_LIST="src/tile_lists/tiles_gpu_${i}.json"
    
    # Build JSON array of tiles
    echo -n "[" > "$TILE_LIST"
    for ((j=START; j<END; j++)); do
        # Add comma if not the first item
        if [ $j -gt $START ]; then
            echo -n "," >> "$TILE_LIST"
        fi
        echo -n "\"${TILES[$j]}\"" >> "$TILE_LIST"
    done
    echo "]" >> "$TILE_LIST"
    
    # Verify the JSON file
    echo "$(date): Created tile list for GPU $i: $TILE_LIST"
    echo "$(date): JSON content (truncated):"
    head -c 300 "$TILE_LIST"
    echo
    
    echo "$(date): GPU $i will process tiles $START to $((END-1)) ($(($END-$START)) tiles)"
done

# Launch python processes for each GPU
echo "$(date): Launching inference processes"

for ((i=0; i<NUM_GPUS; i++)); do
    TILE_LIST="src/tile_lists/tiles_gpu_${i}.json"
    LOG_FILE="logs/infer_process_${i}.log"
    
    echo "$(date): Starting process for GPU $i with tile list $TILE_LIST"
    
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
        --gpu_id $i \
        --tile_list "$TILE_LIST" \
        --process_id $i 2>&1 | tee "$LOG_FILE" &
    
    PID=$!
    echo "$(date): Process for GPU $i started with PID $PID"
    PIDS[$i]=$PID
    
    # Sleep briefly to allow process to initialize GPU connection
    sleep 1
    
    # Record GPU processes associated with this PID for future cleanup
    echo "$(date): GPU processes for PID $PID:" 
    nvidia-smi | grep -i python
done

# Wait for all processes to complete
echo "$(date): All processes launched, waiting for completion"

for ((i=0; i<NUM_GPUS; i++)); do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date): Process on GPU $i completed successfully"
    else
        echo "$(date): Process on GPU $i failed with exit code $EXIT_CODE"
        echo "$(date): Last 50 lines of log file:"
        tail -50 "logs/infer_process_${i}.log"
    fi
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