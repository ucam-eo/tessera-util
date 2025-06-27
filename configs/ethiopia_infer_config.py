# configs/ethiopia_infer_config.py

config = {
    # Paths
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250408_101211.pt",
    "data_raw_dir": "/scratch/zf281/ethiopia/data_raw",
    "data_processed_dir": "/scratch/zf281/ethiopia/data_processed",
    "polygons_locations_csv": "/maps/zf281/btfm4rs/data/downstream/ethiopia/polygons_mgrs_locations.csv",
    "output_npz": "/maps/zf281/btfm4rs/data/representation/ethiopia/representations.npz",
    
    # Model parameters
    "fusion_method": "concat",  # 'sum' or 'concat'
    "max_seq_len_s2": 40,
    "max_seq_len_s1": 40,
    "latent_dim": 128,
    "projector_hidden_dim": 8192*4,
    "projector_out_dim": 8192*4,
    
    # Inference parameters
    "batch_size": 64,
    "num_workers": 4,
    "repeat_times": 10,  # Number of forward passes to average per sample
    
    # Class names
    "class_names": ["Teff", "maize", "barley", "wheat"]
}