config = {
    # 数据所在目录：包含 s1/ 与 s2/ 两个子文件夹
    "data_root": "data/ssl_training/ready_to_use_365",

    # 训练相关
    "epochs": 1,
    "batch_size": 512,
    "learning_rate": 0.001,
    "warmup_ratio": 0.1,
    "plateau_ratio": 0.0,
    "barlow_lambda": 5e-3,
    "fusion_method": "concat",   # 可选 'sum' 或 'concat'
    "apply_mixup": False,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,
    "d_model": 512,              # PixelTransformer输出的维度
    "fuse_out_dim": 512,         # 若 fusion_method='concat', fusion后再线性映射到512
    "projector_hidden_dim": 2048,
    "projector_out_dim": 2048,

    # S2 PixelTransformer
    "s2_num_heads": 8,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 1024,

    # S1 PixelTransformer
    "s1_num_heads": 8,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 1024,

    # 分布式/数据加载
    "num_workers": 4,
    "chunk_batch": 20,          # 一次读取多少对 .npy
    "total_samples": 35400000, # 用于估计 steps(请根据真实数据量修改)
    
    # 日志/验证
    "log_interval_steps": 10,
    "val_interval_steps": 1000,

    # 如果你有验证集(线性探针)的文件路径，可以填入，否则保持 None
    "val_s2_bands_file_path": None,
    "val_s2_masks_file_path": None,
    "val_s1_bands_file_path": None,
    "val_s1_masks_file_path": None,
    "val_labels_path": None,

    # 是否使用AMP / torch.compile
    "apply_amp": False,
    "use_torch_compile": True,

    # 是否禁用W&B检查Git
    "disable_wandb_git": True,

    # 如果需要在每个epoch末执行rust_cmd生成新数据，可以填写
    "rust_cmd": None,
}
