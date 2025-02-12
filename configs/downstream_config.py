# configs/downstream_config.py

config = {
    "checkpoint_path": "checkpoints/ssl/best_model.pt",
    "s2_bands_file_path": "data/downstream/austrian_crop/bands_downsample_100.npy",
    "s2_masks_file_path": "data/downstream/austrian_crop/masks_downsample_100.npy",
    "s2_doy_file_path": "data/downstream/austrian_crop/doys.npy",
    "s1_asc_bands_file_path": "data/downstream/austrian_crop/sar_ascending_downsample_100.npy",
    "s1_asc_doy_file_path": "data/downstream/austrian_crop/sar_ascending_doy.npy",
    "s1_desc_bands_file_path": "data/downstream/austrian_crop/sar_descending_downsample_100.npy",
    "s1_desc_doy_file_path": "data/downstream/austrian_crop/sar_descending_doy.npy",
    "labels_path": "data/downstream/austrian_crop/fieldtype_17classes_downsample_100.npy",
    "field_ids_path": "data/downstream/austrian_crop/fieldid_downsample_100.npy",
    "updated_fielddata_csv": "data/downstream/austrian_crop/updated_fielddata.csv",
    "lon_lat_path": None,
    "t2m_path": None,
    "lai_hv_path": None,
    "lai_lv_path": None,
    "training_ratio": 0.3,
    "val_test_split_ratio": 1/7,
    "batch_size": 8192,
    "epochs": 30,
    "lr": 0.001,
    "weight_decay": 0.01,
    "fusion_method": "sum",  # 可选 'sum' 或 'concat'
    "max_seq_len_s2": 20,
    "max_seq_len_s1": 20,
    "num_workers": 8,
    "latent_dim": 128,
    "projector_hidden_dim": 512,
    "projector_out_dim": 512
}
