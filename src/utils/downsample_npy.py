import numpy as np

decimation_factor = 20 # shrink the data decimation_factor*decimation_factor times

data_path = '/scratch/zf281/jovana/data_processed/11SLA/sar_descending.npy'

data = np.load(data_path, mmap_mode='r') # (n, h, w, c)

# shrink the data decimation_factor*decimation_factor times
data = data[:, ::decimation_factor, ::decimation_factor, ...] # (h, w)

# save the downsampled data
save_path = data_path.replace('11SLA', '11SLA_downsampled')
np.save(save_path, data)
print(f'Saved downsampled data to {save_path}')