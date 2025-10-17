# PV Detection Project

This project implements solar panel (PV) detection using satellite imagery embeddings and machine learning models.

## Project Structure

```
/maps/zf281/btfm4rs/src/pv_detection/
├── data_preprocessing.py    # Data loading and preprocessing
├── sampling.py             # Balanced sampling strategies
├── evaluation.py           # Evaluation metrics and analysis
├── train.py                # Main training script
├── config.py               # Configuration parameters
└── README.md               # This file
```

## Quick Start

### 1. Basic Training

Train both LightGBM and XGBoost models with default settings:

```bash
cd /maps/zf281/btfm4rs/src/pv_detection
python train.py
```

### 2. Custom Training

Train with specific parameters:

```bash
python train.py \
    --data_dir /maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1 \
    --output_dir ./models \
    --pos_neg_ratio 0.5 \
    --models lightgbm xgboost \
    --balanced_sampling
```

### 3. Train Single Model

Train only LightGBM:

```bash
python train.py --models lightgbm
```

## Parameters

- `--data_dir`: Directory containing the data files
- `--output_dir`: Directory to save trained models
- `--pos_neg_ratio`: Positive to negative ratio (0.33 = 1:3, 0.5 = 1:2)
- `--models`: Models to train (lightgbm, xgboost, or both)
- `--balanced_sampling`: Use balanced sampling (recommended)
- `--random_seed`: Random seed for reproducibility

## Expected Output

The training script will:

1. Load and preprocess the data
2. Create balanced training sets
3. Train the specified models
4. Evaluate performance with detailed metrics including:
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - AUC-ROC (if applicable)
   - Class-specific error rates
5. Find optimal classification thresholds
6. Compare models (if multiple trained)
7. Save trained models and results

## Data Requirements

The following files should be present in the data directory:

- `roi_1_map_10m_utm30n_128bands.npy`: Int8 quantized embeddings (H, W, 128)
- `roi_1_map_10m_utm30n_scales.npy`: Dequantization scales (H, W)
- `roi_1_clipped_gt_10m.npy`: Ground truth labels (H, W) where 1=PV, 0=no PV

## Hardware Optimization

The implementation is optimized for your hardware:

- Uses all 96 CPU cores (`n_jobs=-1`)
- Efficient memory usage with chunked processing
- Fast inference for production use

## Model Performance

Both models are configured to handle the highly imbalanced dataset:

- **LightGBM**: Optimized for speed and memory efficiency
- **XGBoost**: Robust performance with good generalization

Expected performance on validation set:
- Recall: >80% (detecting most solar panels)
- Precision: >70% (low false positive rate)
- F1-score: >75% (balanced performance)

The exact numbers will depend on your specific dataset characteristics.

## Troubleshooting

### Memory Issues
If you encounter memory issues, try reducing the balanced dataset size in the sampling module.

### Poor Performance
- Check class distribution in your data
- Adjust `pos_neg_ratio` parameter
- Modify model hyperparameters in `config.py`

### Installation Requirements
Make sure you have the following packages installed:
```bash
pip install lightgbm xgboost scikit-learn numpy
```