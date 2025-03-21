# configs/ssl_config.py

config = {
    "data_root": "data/ssl_training/ready_to_use_temp",
    "batch_size": 1024,
    "epochs": 4,
    "learning_rate": 0.005,
    "barlow_lambda": 5e-3,
    "fusion_method": "sum",  # 可选 'sum', 'concat', 'transformer'
    "latent_dim": 128,
    "projector_hidden_dim": 2048,
    "projector_out_dim": 2048,
    "min_valid_timesteps": 20,
    "sample_size_s2": 20,
    "sample_size_s1": 20,
    "num_workers": 0,
    "shuffle_tiles": True,
    "log_interval_steps":10,
    "val_interval_steps": 600,
    "eval_method": "linear_probe",
    "val_s2_bands_file_path": "data/ssl_training/austrian_crop/bands_downsample_100.npy",
    "val_s2_masks_file_path": "data/ssl_training/austrian_crop/masks_downsample_100.npy",
    "val_s2_doy_file_path": "data/ssl_training/austrian_crop/doys.npy",
    "val_s1_asc_bands_file_path": "data/ssl_training/austrian_crop/sar_ascending_downsample_100.npy",
    "val_s1_asc_doy_file_path": "data/ssl_training/austrian_crop/sar_ascending_doy.npy",
    "val_s1_desc_bands_file_path": "data/ssl_training/austrian_crop/sar_descending_downsample_100.npy",
    "val_s1_desc_doy_file_path": "data/ssl_training/austrian_crop/sar_descending_doy.npy",
    "val_labels_path": "data/ssl_training/austrian_crop/fieldtype_17classes_downsample_100.npy",
    "warmup_ratio": 0.3,
    "plateau_ratio": 0,
    "apply_mixup": True,
    # "apply_mixup": False,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,
    "total_samples": 67000000,
    "rust_cmd": (
        "RUST_LOG=debug singularity run training-data-preprocessing/training-data-preprocessing.sif "
        "--data-root /home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/s2_s1_global_project/data_processed "
        "--output-dir data/ssl_training/ready_to_use_temp "
        "--tile-batch 90 "
        "--time-steps 20 "
        "--chunk-size 1000000 "
        "--min-valid-timesteps 20"
    ),
    "chunk_batch": 23, # chunk batch when loading data
    # "apply_amp": True,
    "apply_amp": False,
}
