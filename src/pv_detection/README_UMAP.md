# UMAP Visualization for Solar Panel Detection

## 概述

这个脚本用于对2024年的太阳能板检测训练数据进行UMAP可视化。它会加载embeddings数据，进行反量化，然后使用UMAP降维到2D空间，最终生成符合Nature期刊标准的可视化图像。

## 功能特点

- **数据加载**: 自动加载和反量化2024年的embeddings数据
- **智能采样**: 使用所有solar panels样本 + 10倍随机others样本
- **优化的UMAP**: 配置了更好聚类效果的参数，利用64核CPU并行处理
- **高质量可视化**: 符合Nature期刊标准，包含透明度和密度反映
- **详细日志**: 完整的进度跟踪和统计信息
- **多格式输出**: 支持PNG、PDF、SVG格式

## 使用方法

### 基本使用

```bash
# 使用默认参数
/maps/zf281/miniconda3/envs/detectree-env/bin/python umap_visualization.py

# 指定输出目录
/maps/zf281/miniconda3/envs/detectree-env/bin/python umap_visualization.py --output_dir ./umap_results
```

### 高级参数

```bash
/maps/zf281/miniconda3/envs/detectree-env/bin/python umap_visualization.py \
    --data_dir /maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1 \
    --output_dir /maps/zf281/btfm4rs/src/pv_detection/umap_results \
    --year 2024 \
    --others_multiplier 10 \
    --n_neighbors 50 \
    --min_dist 0.1 \
    --n_epochs 1000 \
    --random_seed 42
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1` | 数据文件目录 |
| `--output_dir` | `/maps/zf281/btfm4rs/src/pv_detection` | 输出目录 |
| `--year` | `2024` | 要可视化的数据年份 |
| `--others_multiplier` | `10` | others样本相对于solar panels的倍数 |
| `--n_neighbors` | `50` | UMAP邻居数量参数 |
| `--min_dist` | `0.1` | UMAP最小距离参数 |
| `--n_epochs` | `1000` | UMAP训练轮数 |
| `--random_seed` | `42` | 随机种子 |

## 输出文件

脚本会在输出目录生成以下文件：

1. **可视化图像**:
   - `umap_visualization_2024.png` - 主要的散点图可视化
   - `umap_visualization_2024.pdf` - PDF格式（适合论文）
   - `umap_visualization_2024.svg` - 矢量格式（可编辑）

2. **密度可视化**:
   - `umap_density_2024.png` - 密度分布图
   - `umap_density_2024.pdf` - PDF格式密度图

3. **数据文件**:
   - `umap_model_2024.pkl` - 训练好的UMAP模型
   - `umap_embeddings_2024.npy` - 2D降维后的embeddings
   - `visualization_stats_2024.json` - 统计信息和元数据

4. **日志文件**:
   - `umap_visualization.log` - 详细的运行日志

## 数据要求

脚本需要以下数据文件存在：

1. `2024_roi_1_map_10m_utm30n_128bands.npy` - embeddings数据（int8量化）
2. `2024_roi_1_map_10m_utm30n_scales.npy` - 反量化的scale因子
3. `roi_1_clipped_gt_10m.npy` - 标签数据

## 性能优化

- **并行处理**: 利用64核CPU进行UMAP计算
- **内存优化**: 智能采样减少内存使用
- **缓存支持**: 可以保存中间结果避免重复计算

## 可视化特点

### 主要散点图
- **无坐标轴**: 专注于数据分布模式
- **透明度**: Solar panels (α=0.8), Others (α=0.3)
- **颜色**: Solar panels (橙色), Others (蓝色)
- **大小**: 适合密度显示的点大小
- **图例**: 包含样本数量信息

### 密度图
- **六边形密度图**: 更好地显示聚类模式
- **分离显示**: Solar panels和Others分别显示
- **颜色映射**: 使用专业的颜色方案

## 故障排除

### 常见问题

1. **内存不足**: 减少`others_multiplier`参数
2. **文件不存在**: 检查`data_dir`路径是否正确
3. **UMAP安装**: 确保在detectree-env环境中安装了umap-learn

### 环境检查

```bash
# 检查Python环境
/maps/zf281/miniconda3/envs/detectree-env/bin/python --version

# 检查必要的包
/maps/zf281/miniconda3/envs/detectree-env/bin/python -c "import umap, matplotlib, seaborn, numpy; print('All packages available')"
```

## 技术细节

### UMAP参数选择
- `n_neighbors=50`: 平衡局部和全局结构
- `min_dist=0.1`: 允许紧密聚类
- `n_epochs=1000`: 充分训练以获得稳定结果
- `metric='euclidean'`: 适合embeddings数据的距离度量

### 采样策略
- **Solar panels**: 使用所有可用样本
- **Others**: 随机采样10倍数量，确保类别平衡的同时保持足够的代表性

### 可视化标准
- **DPI**: 300 (高质量打印)
- **格式**: 多种格式支持不同用途
- **颜色**: 色盲友好的颜色选择
- **透明度**: 反映数据密度

## 示例输出

运行成功后，你会看到类似以下的日志输出：

```
2024-XX-XX XX:XX:XX - INFO - Initialized UMAP Visualizer
2024-XX-XX XX:XX:XX - INFO - Loading data for year 2024...
2024-XX-XX XX:XX:XX - INFO - Total valid samples: XXX,XXX
2024-XX-XX XX:XX:XX - INFO - Solar panel samples: X,XXX
2024-XX-XX XX:XX:XX - INFO - UMAP fitting completed in XX.XXs
2024-XX-XX XX:XX:XX - INFO - Full pipeline completed in XX.XXs
```

## 后续分析

生成的可视化可以用于：
- 评估embeddings的质量
- 分析solar panels和others的可分离性
- 识别潜在的数据问题或异常值
- 为模型改进提供洞察

## 联系信息

如有问题或需要修改，请联系项目维护者。