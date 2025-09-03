#!/bin/bash

# 批量运行Austria土地分类实验
# 参数组合: [sample_per_pixel, num_experiment]

echo "=========================================="
echo "Starting batch Austria classification experiments"
echo "=========================================="

# 定义参数数组 - 添加新的组合
declare -a sample_per_pixel_values=(1 2 4 6 8 10)
declare -a num_experiment_values=(2000 1000 500 333 250 200)

# 获取数组长度
array_length=${#sample_per_pixel_values[@]}

# Python脚本路径
PYTHON_SCRIPT="/mnt/e/Codes/btfm4rs/src/train_downstream_via_representation_austria_select_by_class.py"

# 记录开始时间
start_time=$(date +%s)
echo "Start time: $(date)"
echo ""

# 循环执行每个参数组合
for (( i=0; i<${array_length}; i++ )); do
    sample_per_pixel=${sample_per_pixel_values[$i]}
    num_experiment=${num_experiment_values[$i]}
    
    # 根据sample_per_pixel值动态设置epochs
    if [ ${sample_per_pixel} -lt 8 ]; then
        num_epochs=100
    else
        num_epochs=200
    fi
    
    echo "=========================================="
    echo "Running experiment $(($i+1))/${array_length}"
    echo "Parameters: sample_per_pixel=${sample_per_pixel}, num_experiment=${num_experiment}, num_epochs=${num_epochs}"
    echo "=========================================="
    
    # 记录任务开始时间
    task_start=$(date +%s)
    
    # 执行Python脚本
    /mnt/d/wsl_anaconda3/bin/python ${PYTHON_SCRIPT} \
        --sample_per_pixel ${sample_per_pixel} \
        --num_experiment ${num_experiment} \
        --method mlp \
        --num_epochs ${num_epochs} \
        --batch_size 8192
    
    # 检查执行状态
    if [ $? -eq 0 ]; then
        echo "✓ Experiment $(($i+1)) completed successfully"
    else
        echo "✗ Experiment $(($i+1)) failed"
        echo "Continuing with next experiment..."
    fi
    
    # 计算任务执行时间
    task_end=$(date +%s)
    task_duration=$((task_end - task_start))
    echo "Task duration: $(($task_duration / 60)) minutes $(($task_duration % 60)) seconds"
    echo ""
    
    # 如果不是最后一个任务，等待5秒
    if [ $i -lt $((array_length - 1)) ]; then
        echo "Waiting 5 seconds before next experiment..."
        sleep 5
    fi
done

# 记录结束时间
end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo "=========================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "Total duration: $(($total_duration / 3600)) hours $(($total_duration % 3600 / 60)) minutes $(($total_duration % 60)) seconds"
echo "=========================================="