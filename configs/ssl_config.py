# configs/ssl_config.py

config = {
    "data_root": "data/ssl_training/sweep/s1_s2_ratio",
    "batch_size": 1024,
    "epochs": 1,
    "learning_rate": 0.01,
    "barlow_lambda": 5e-3,
    "fusion_method": "concat",  # 可选 'sum', 'concat', 'transformer'
    "latent_dim": 128,
    "projector_hidden_dim": 2048,
    "projector_out_dim": 2048,
    "min_valid_timesteps": 20,
    "sample_size_s2": 20,
    "sample_size_s1": 20,
    "num_workers": 16,
    "shuffle_tiles": True,
    "log_interval_steps":10,
    "val_interval_steps": 300,
    "eval_method": "linear_probe",
    "val_s2_bands_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/bands_downsample_100.npy",
    "val_s2_masks_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/masks_downsample_100.npy",
    "val_s2_doy_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/doys.npy",
    "val_s1_asc_bands_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_ascending_downsample_100.npy",
    "val_s1_asc_doy_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_ascending_doy.npy",
    "val_s1_desc_bands_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_descending_downsample_100.npy",
    "val_s1_desc_doy_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_descending_doy.npy",
    "val_labels_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes_downsample_100.npy",
    "warmup_ratio": 0.2,
    "plateau_ratio": 0.2,
    "apply_mixup": False,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,
    "total_samples": 8500000,
    "rust_cmd": (
        "RUST_LOG=debug src/utils/training-data-preprocessing "
        "--data-root /scratch/zf281/global "
        "--output-dir /scratch/zf281/ready_to_use "
        "--tile-batch 60 "
        "--time-steps 20 "
        "--chunk-size 1000000 "
        "--min-valid-timesteps 10"
    )
}
