# configs/ssl_config_64_fixed.py
# Configuration for the 64 timesteps SSL training

config = {
    # Data paths
    "data_root": "data/ready_to_use_64",  # Updated path for 64 timesteps data
    
    # Training parameters
    "batch_size": 512,
    "epochs": 1,
    "learning_rate": 0.002,
    # "learning_rate": 0.02,
    "barlow_lambda": 5e-3,
    "fusion_method": "concat",  # Options: 'sum', 'concat', 'transformer'
    
    # Model dimensions
    "latent_dim": 128,
    "projector_hidden_dim": 4096,
    "projector_out_dim": 4096,
    
    # Data processing
    "min_valid_timesteps": 0,  # Minimum valid timesteps required
    "sample_size_s2": 64,       # Number of S2 timesteps (fixed at 64)
    "sample_size_s1": 64,       # Number of S1 timesteps (fixed at 64)
    "num_workers": 8,
    "shuffle_tiles": True,
    
    # Logging and validation
    "log_interval_steps": 10,
    "val_interval_steps": 1200,
    "eval_method": "linear_probe",
    
    # Validation data paths
    "val_s2_bands_file_path": "data/ssl_training/austrian_crop/bands_downsample_100.npy",
    "val_s2_masks_file_path": "data/ssl_training/austrian_crop/masks_downsample_100.npy",
    "val_s2_doy_file_path": "data/ssl_training/austrian_crop/doys.npy",
    "val_s1_asc_bands_file_path": "data/ssl_training/austrian_crop/sar_ascending_downsample_100.npy",
    "val_s1_asc_doy_file_path": "data/ssl_training/austrian_crop/sar_ascending_doy.npy",
    "val_s1_desc_bands_file_path": "data/ssl_training/austrian_crop/sar_descending_downsample_100.npy",
    "val_s1_desc_doy_file_path": "data/ssl_training/austrian_crop/sar_descending_doy.npy",
    "val_labels_path": "data/ssl_training/austrian_crop/fieldtype_17classes_downsample_100.npy",
    
    # Field-based validation parameters
    "field_id_path": "data/ssl_training/austrian_crop/fieldid_downsample_100.npy",
    "fielddata_csv_path": "data/ssl_training/austrian_crop/updated_fielddata.csv",
    "training_ratio": 0.1,  # Proportion of fields used for training
    "val_test_split_ratio": 1/7,  # Validation/test split ratio
    
    # Multiple inference settings
    "num_inference": 1,  # Number of inference passes to average
    
    # Linear probe classifier type
    "classifier_type": "lr",  # Options: 'lr' (LogisticRegression) or 'rf' (RandomForest)
    
    # Learning rate schedule
    "warmup_ratio": 0.4,
    "plateau_ratio": 0,
    
    # Mixup settings
    "apply_mixup": True,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,
    
    # Total number of samples for training
    "total_samples": 50*500000,
    
    # AMP (Automatic Mixed Precision)
    "apply_amp": True,
}