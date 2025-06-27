# configs/ethiopia_config.py

config = {
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250408_101211.pt",
    
    # 数据路径
    "data_raw_dir": "/scratch/zf281/ethiopia/data_raw",
    "data_processed_dir": "/scratch/zf281/ethiopia/data_processed",
    "shapefile_path": "/maps/zf281/btfm4rs/data/downstream/ethiopia/shp/EthCT2020_top4_classes.shp",
    "polygons_locations_csv": "/maps/zf281/btfm4rs/data/downstream/ethiopia/polygons_mgrs_locations.csv",
    "output_dir": "/maps/zf281/btfm4rs/data/downstream/ethiopia",
    
    # 训练参数
    "samples_per_class": 10,  # 每个类别用于训练的样本数
    "batch_size": 128,
    "epochs": 100,
    "lr": 0.001,
    "weight_decay": 0.01,
    
    # 模型参数
    "fusion_method": "concat",  # 'sum'或'concat'
    "max_seq_len_s2": 40,
    "max_seq_len_s1": 40,
    "num_workers": 0,
    "latent_dim": 128,
    "projector_hidden_dim": 8192*4,
    "projector_out_dim": 8192*4,
    
    # 类别映射
    "class_names": ["Teff", "maize", "barley", "wheat"]
}