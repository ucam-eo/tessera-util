# configs/downstream_config.py

config = {
    # "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250407_195912.pt",
    "checkpoint_path": "checkpoints/ssl/best_model_fsdp_20250602_221903.pt",
    # "checkpoint_path": "checkpoints/ssl/fsdp_checkpoint_20250602_200048.pt",
    
    "s2_bands_file_path": "data/downstream/austrian_crop_v1.0_pipeline/bands_downsample_100.npy",
    "s2_masks_file_path": "data/downstream/austrian_crop_v1.0_pipeline/masks_downsample_100.npy",
    "s2_doy_file_path": "data/downstream/austrian_crop_v1.0_pipeline/doys.npy",
    "s1_asc_bands_file_path": "data/downstream/austrian_crop_v1.0_pipeline/sar_ascending_downsample_100.npy",
    "s1_asc_doy_file_path": "data/downstream/austrian_crop_v1.0_pipeline/sar_ascending_doy.npy",
    "s1_desc_bands_file_path": "data/downstream/austrian_crop_v1.0_pipeline/sar_descending_downsample_100.npy",
    "s1_desc_doy_file_path": "data/downstream/austrian_crop_v1.0_pipeline/sar_descending_doy.npy",
    "labels_path": "data/downstream/austrian_crop_v1.0_pipeline/fieldtype_17classes_downsample_100.npy",
    "field_ids_path": "data/downstream/austrian_crop_v1.0_pipeline/fieldid_downsample_100.npy",
    
    # "s2_bands_file_path": "data/downstream/austrian_crop/bands_downsample_100.npy",
    # "s2_masks_file_path": "data/downstream/austrian_crop/masks_downsample_100.npy",
    # "s2_doy_file_path": "data/downstream/austrian_crop/doys.npy",
    # "s1_asc_bands_file_path": "data/downstream/austrian_crop/sar_ascending_downsample_100.npy",
    # "s1_asc_doy_file_path": "data/downstream/austrian_crop/sar_ascending_doy.npy",
    # "s1_desc_bands_file_path": "data/downstream/austrian_crop/sar_descending_downsample_100.npy",
    # "s1_desc_doy_file_path": "data/downstream/austrian_crop/sar_descending_doy.npy",
    # "labels_path": "data/downstream/austrian_crop/fieldtype_17classes_downsample_100.npy",
    # "field_ids_path": "data/downstream/austrian_crop/fieldid_downsample_100.npy",
    
    "updated_fielddata_csv": "data/downstream/austrian_crop/updated_fielddata.csv",
    "lon_lat_path": None,
    "t2m_path": None,
    "lai_hv_path": None,
    "lai_lv_path": None,
    "training_ratio": 0.1,
    "val_test_split_ratio": 1/7,
    "batch_size": 512,
    "epochs": 50,
    "lr": 0.001,
    "weight_decay": 0.01,
    "fusion_method": "concat",  # 可选 'sum' 或 'concat'
    "max_seq_len_s2": 40,
    "max_seq_len_s1": 40,
    "num_workers": 8,
    "latent_dim": 128,
    "projector_hidden_dim": 8192*4,
    "projector_out_dim": 8192*4
}
