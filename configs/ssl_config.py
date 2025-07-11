# configs/ssl_config.py

config = {
    # "data_root": "data/ssl_training/ready_to_use_temp",
    "data_root": "data/ssl_training/ready_to_use_40_steps",
    # "data_root": "data/ssl_training/ready_to_use_64_steps",
    # "data_root": "data/ssl_training/ready_to_use_temp_small",
    "batch_size": 2048,
    "epochs": 1,
    "learning_rate": 0.002,
    "barlow_lambda": 5e-3,
    "fusion_method": "concat",  # 可选 'sum', 'concat', 'transformer'
    "latent_dim": 128,

    "s2_num_heads": 4,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 4096,
    "s1_num_heads": 4,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 4096,
    
    # ========== 从Checkpoint恢复训练的参数 ==========
    # 取消注释下面的行以从checkpoint恢复训练
    # "resume_from_checkpoint": "btfm4rs/checkpoints/ssl/checkpoint_20250605_121934.pt",
    "resume_from_checkpoint": None,  # 设置为None表示不从checkpoint恢复
    # 恢复训练时的学习率，如果不设置则使用 learning_rate
    "resume_learning_rate": 0.002,  # 可以设置比初始学习率更小的值
    # 恢复训练时的warmup步数，用于平滑过渡（可选，设为0表示不进行warmup）
    "resume_warmup_steps": 0,  # 例如设置为100会在恢复后进行100步的warmup
    
    # Representation Quantization Aware Training (QAT)
    "apply_qat_representation": True,  # 是否对表征应用QAT
    "qat_representation_bits": 8,      # 量化位数 (目前固定为8)
    "qat_representation_symmetric": True, # 是否使用对称量化 (目前固定为True)
    "qat_representation_start_step": 5000, # 从哪个全局step开始应用QAT (例如，如果total_steps是10000，设为5000则从一半开始)
                                        # 设置为 0 则从一开始就应用

    # 投影头维度
    # "projector_out_dim": 8192*2,
    # "projector_hidden_dim": 8192*2,
    "projector_out_dim": 128,
    "projector_hidden_dim": 128,

    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "num_workers": 2,
    "shuffle_tiles": True,
    "log_interval_steps": 10,
    "val_interval_steps": 600,
    "eval_method": "linear_probe",
    "val_s2_bands_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/bands_downsample_100.npy",
    "val_s2_masks_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/masks_downsample_100.npy",
    "val_s2_doy_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/doys.npy",
    "val_s1_asc_bands_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_ascending_downsample_100.npy",
    "val_s1_asc_doy_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_ascending_doy.npy",
    "val_s1_desc_bands_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_descending_downsample_100.npy",
    "val_s1_desc_doy_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_descending_doy.npy",
    "val_labels_path": "data/ssl_training/austrian_crop_v1.0_pipeline/fieldtype_17classes_downsample_100.npy",
    
    
    # 基于field的验证参数
    "field_id_path": "data/ssl_training/austrian_crop_v1.0_pipeline/fieldid_downsample_100.npy",
    "fielddata_csv_path": "data/ssl_training/austrian_crop_v1.0_pipeline/updated_fielddata.csv",
    "training_ratio": 0.1,  # 用于训练的field面积比例
    "val_test_split_ratio": 1/7,  # 验证集和测试集之间的分割比例
    # 多次推理
    "num_inference": 1,  # 推理次数，取平均
    # 线性探针的分类器类型
    "classifier_type": "lr",  # 选项: 'lr' (LogisticRegression) 或 'rf' (RandomForest)
    
    "warmup_ratio": 0.1,
    "plateau_ratio": 0,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,

    # "total_samples": 132000000,
    "total_samples": 465 * 1000000,
    # "total_samples": 15000000,

    "rust_cmd": (
        "RUST_LOG=debug src/utils/training-data-preprocessing-single-doy "
        "--data-root data/ssl_training/tiles "
        "--output-dir data/ssl_training/ready_to_use_40_steps "
        "--tile-batch 200 "
        "--time-steps 40 "
        "--chunk-size 1000000 "
        "--s2-min-valid-timesteps 10 "
        "--s1-min-valid-timesteps 10 "
    ),

    "chunk_batch": 20,   # 每次加载多少个 .npy 文件一起拼成一个 ChunkDataset
    
    "apply_mixup": True,

    # 是否使用AMP
    # "apply_amp": True,
    "apply_amp": False,

    # 是否使用torch.compile
    "use_torch_compile": True,

    # 是否在 W&B 中禁用对 Git 仓库信息的检查（可选）
    "disable_wandb_git": True,
}