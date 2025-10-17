"""
Configuration file for PV detection project.
"""

# Data paths
DATA_DIR = "/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1"
OUTPUT_DIR = "/maps/zf281/btfm4rs/src/pv_detection/models"

# Data files
REPRESENTATION_FILE = "roi_1_map_10m_utm30n_128bands.npy"
SCALES_FILE = "roi_1_map_10m_utm30n_scales.npy"
LABELS_FILE = "roi_1_clipped_gt_10m.npy"

# Training parameters
TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% validation
POS_NEG_RATIO = 0.33  # 1:3 positive to negative ratio
RANDOM_SEED = 42

# Model parameters
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'n_jobs': -1,
    'verbosity': 1
}

# Training settings
NUM_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50
EVAL_PERIOD = 100

# Evaluation settings
THRESHOLD_SEARCH_RANGE = (0.01, 1.0)
THRESHOLD_SEARCH_STEP = 0.01
OPTIMIZATION_METRIC = 'f1'  # Can be 'f1', 'precision', 'recall', 'accuracy'