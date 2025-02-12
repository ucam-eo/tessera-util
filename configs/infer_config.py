from copy import deepcopy
from configs.ssl_config import config

# 复制基础配置，避免直接修改基础配置对象
config = deepcopy(config)

# 更新或新增下游配置的专有内容
config.update({
    "tile_path": "data/downstream/borneo/50NNL_subset",
    "checkpoint_path": "checkpoints/ssl/best_model.pt",
    "repeat_times": 10,
    "min_valid_timesteps": 0,
    "batch_size": 1024,
    "output_npy": "data/representation/borneo_representations.npy",
    "num_workers": 0,
})