# configs/ssl_config.py

config = {
    # "data_root": "data/ready_to_use",
    # "data_root": "data/ready_to_use_3_aug",
    # "data_root": "data/ready_to_use_min_step_30",
    "data_root": "data/ready_to_use_64_steps",
    "batch_size": 256,
    "epochs": 1,
    "learning_rate": 0.002,
    "barlow_lambda": 5e-3,
    "fusion_method": "concat",  # 可选 'sum', 'concat', 'transformer'
    "latent_dim": 128,
    "projector_hidden_dim": 4096,
    "projector_out_dim": 4096,
    "min_valid_timesteps": 20,
    "sample_size_s2": 64,
    "sample_size_s1": 64,
    "num_workers": 8,
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
    
    # 基于field的验证参数
    "field_id_path": "data/ssl_training/austrian_crop/fieldid_downsample_100.npy",
    "fielddata_csv_path": "data/ssl_training/austrian_crop/updated_fielddata.csv",
    "training_ratio": 0.1,  # 用于训练的field面积比例
    "val_test_split_ratio": 1/7,  # 验证集和测试集之间的分割比例
    # 多次推理
    "num_inference": 10,  # 推理次数，取平均
    # 线性探针的分类器类型
    "classifier_type": "lr",  # 选项: 'lr' (LogisticRegression) 或 'rf' (RandomForest)
    
    "warmup_ratio": 0.1,
    "plateau_ratio": 0,
    "apply_mixup": True,
    # "apply_mixup": False,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,
    "total_samples": 229*100000,
    "rust_cmd": (
        "RUST_LOG=debug src/utils/training-data-preprocessing "
        "--data-root /mnt/d/global_tiles_267 "
        "--output-dir data/ready_to_use "
        "--tile-batch 30 "
        "--time-steps 20 "
        "--chunk-size 500000 "
        "--min-valid-timesteps 20"
    ),
    "apply_amp": True,
}
