# CNN Training Configuration

## Quick Start

To train the CNN model with default settings:

```bash
cd /maps/zf281/btfm4rs/src/pv_detection
python train_cnn.py --use_balanced_sampling --use_class_weights
```

## Training Options

### Basic Usage
```bash
# Train simple CNN with balanced sampling
python train_cnn.py --model_type simple --use_balanced_sampling

# Train deeper CNN with class weights
python train_cnn.py --model_type deep --use_class_weights

# Custom batch size and learning rate
python train_cnn.py --batch_size 512 --learning_rate 0.0005
```

### Advanced Options
```bash
# Long training with early stopping
python train_cnn.py \
    --num_epochs 200 \
    --early_stopping \
    --early_stopping_patience 20 \
    --scheduler plateau

# High regularization setup
python train_cnn.py \
    --dropout_rate 0.3 \
    --weight_decay 1e-3 \
    --use_class_weights
```

## Key Parameters

### Model Architecture
- `--model_type`: Choose 'simple' (lightweight) or 'deep' (more parameters)
- `--patch_size`: Size of input patches (default: 3 for 3x3)
- `--dropout_rate`: Dropout probability for regularization

### Training Settings
- `--batch_size`: Batch size (default: 256)
- `--num_epochs`: Maximum training epochs (default: 100)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--optimizer`: 'adam' or 'sgd'

### Data Handling
- `--use_balanced_sampling`: Use weighted sampling for balanced batches
- `--use_class_weights`: Apply class weights to loss function
- `--val_ratio`: Validation split ratio (default: 0.1)

### Learning Rate Scheduling
- `--scheduler`: 'cosine', 'step', 'plateau', or 'none'
- `--early_stopping`: Enable early stopping based on validation F1

## Expected Output

The training will create:
- `cnn_models/cnn_model.pth`: Trained model weights
- `cnn_models/training_history.json`: Training metrics over time
- `cnn_models/config.json`: Training configuration

## Performance Tips

1. **For quick experimentation**: Use simple model with small batch size
2. **For best performance**: Use deep model with balanced sampling and class weights
3. **For large datasets**: Increase batch size and use more workers
4. **If overfitting**: Increase dropout rate and weight decay

## Hardware Requirements

- **GPU**: Recommended for faster training (will use CUDA if available)
- **CPU**: Works but slower, recommend increasing num_workers
- **Memory**: ~4GB GPU memory for batch_size=256

## Model Architecture Details

### Simple CNN (~50K parameters)
```
Input (3x3x128) → Conv2d(64,3x3) → Conv2d(32,3x3) → Conv2d(16,1x1) → GlobalAvgPool → FC(2)
```

### Deep CNN (~100K parameters)
```
Input (3x3x128) → Conv2d(64,3x3) → Conv2d(64,3x3) → Conv2d(32,3x3) → Conv2d(16,1x1) → GlobalAvgPool → FC(2)
```

Both models use:
- Batch normalization after each conv layer
- ReLU activation functions
- Global average pooling instead of flatten
- Dropout before final classification layer