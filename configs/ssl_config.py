# configs/ssl_config.py

config = {
    # "data_root": "data/ssl_training/ready_to_use",
    "data_root": "data/ssl_training/ready_to_use_temp_small",
    "batch_size": 1024,
    "epochs": 1,
    "learning_rate": 0.002,
    "barlow_lambda": 5e-3,
    "fusion_method": "concat",  # 可选 'sum', 'concat', 'transformer'
    "latent_dim": 128,

    "s2_num_heads": 16,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 4096,
    "s1_num_heads": 16,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 4096,

    # 投影头维度
    "projector_out_dim": 24576,
    "projector_hidden_dim": 24576,

    "sample_size_s2": 20,
    "sample_size_s1": 20,
    "num_workers": 2,
    "shuffle_tiles": True,
    "log_interval_steps": 10,
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
    "warmup_ratio": 0.4,
    "plateau_ratio": 0,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,

    "total_samples": 132000000,

    "rust_cmd": (
        "RUST_LOG=debug src/utils/training-data-preprocessing "
        "--data-root data/ssl_training/tiles "
        "--output-dir data/ssl_training/ready_to_use "
        "--tile-batch 200 "
        "--time-steps 20 "
        "--chunk-size 1000000 "
        "--s2-min-valid-timesteps 10 "
        "--s1-min-valid-timesteps 10 "
    ),

    "chunk_batch": 45,   # 每次加载多少个 .npy 文件一起拼成一个 ChunkDataset
    
    "apply_mixup": True,

    # 是否使用AMP
    # "apply_amp": True,
    "apply_amp": False,

    # 是否使用torch.compile
    "use_torch_compile": True,

    # 是否在 W&B 中禁用对 Git 仓库信息的检查（可选）
    "disable_wandb_git": True,
}
