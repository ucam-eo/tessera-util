# 推荐的简化训练命令

## 方案1：仅使用class weights（推荐）
```bash
cd /maps/zf281/btfm4rs/src/pv_detection
python train_cnn.py --use_class_weights --batch_size 128
```

## 方案2：如果要使用balanced sampling（已修复）
```bash
python train_cnn.py --use_balanced_sampling --batch_size 128
```

## 方案3：最简单配置（无特殊采样）
```bash
python train_cnn.py --batch_size 128 --num_epochs 50
```

## 问题说明
原始错误是因为数据集太大（超过1000万个patches），PyTorch的WeightedRandomSampler有数量限制。

现在已经修复：
1. 限制每个epoch的样本数为1M（仍然足够大）
2. 推荐直接使用class weights而不是balanced sampling
3. 减小batch size避免内存问题

## 建议
从方案1开始，它最稳定且有效处理类别不平衡问题。