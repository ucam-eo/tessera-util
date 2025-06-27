# configs/downstream_config.py

config = {
    # "checkpoint_path": "checkpoints/ssl/best_model_dual_transformer_32layer_120tiles_5epoch.pt",
    "s2_bands_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/bands_downsample_100.npy",
    "s2_masks_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/masks_downsample_100.npy",
    "s2_doy_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/doys.npy",
    "s1_asc_bands_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_ascending_downsample_100.npy",
    "s1_asc_doy_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_ascending_doy.npy",
    "s1_desc_bands_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_descending_downsample_100.npy",
    "s1_desc_doy_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_descending_doy.npy",
    "labels_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes_downsample_100.npy",
    "field_ids_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldid_downsample_100.npy",
    
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250408_101211.pt",
    # "s2_bands_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/bands.npy",
    # "s2_masks_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/masks.npy",
    # "s2_doy_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/doys.npy",
    # "s1_asc_bands_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_ascending.npy", 
    # "s1_asc_doy_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_ascending_doy.npy",
    # "s1_desc_bands_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_descending.npy",
    # "s1_desc_doy_file_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/sar_descending_doy.npy",
    # "labels_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes.npy",
    # "field_ids_path": "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldid.npy",
    
    "updated_fielddata_csv": "/maps/zf281/btfm-training-10.4/maddy_code/data/updated_fielddata.csv",
    "lon_lat_path": None,
    "t2m_path": None,
    "lai_hv_path": None,
    "lai_lv_path": None,
    "training_ratio": 0.1,
    "val_test_split_ratio": 1/7,
    "batch_size": 1024,
    "epochs": 50,
    "lr": 0.001,
    "weight_decay": 0.01,
    "fusion_method": "concat",  # 可选 'sum' 或 'concat'
    "max_seq_len_s2": 40,
    "max_seq_len_s1": 40,
    "num_workers": 0,
    "latent_dim": 128,
    "projector_hidden_dim": 8192*4,
    "projector_out_dim": 8192*4
}
