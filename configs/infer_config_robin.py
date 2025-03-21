from copy import deepcopy
from configs.ssl_config import config

# copy base config
config = deepcopy(config)

# update or add new configs
config.update({
    # "tile_path": "/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/s2_s1_global_project/processed/33UXP_whole_year",
    # "tile_path": "data/ssl_training/austrian_crop/MGRS_33UXP",
    "tile_path": "/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/s2_s1_global_project/robin_fungal/2021/34VFJ",
    # "tile_path": "/home/zf281/rds/hpc-work/Files/btfm4rs/data/ssl_training/austrian_crop",
    "checkpoint_path": "checkpoints/ssl/best_model_20250315_222553.pt",
    "repeat_times": 10,
    "min_valid_timesteps": 0,
    "batch_size": 8192,
    "output_npy": "data/representation/robin_fungal/34VFJ/representation.npy",
    "num_workers": 0,
    "max_band_file_size": 10, # Unit: GB
})