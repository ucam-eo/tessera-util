import numpy as np
import matplotlib.pyplot as plt

path = "/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/biomassters_train/train_agbm_representation/2021_ffc563f4_agbm.npy"
data = np.load(path, mmap_mode='r')
print(data.dtype)  # Output the data type
print(data.shape)  # Output the shape of the array

# print(data)

# first_three_band = data[::10,::10,:3].copy()
first_three_band = data[:,:,:3].copy()
# first_three_band = data[:,:,3:6,0].copy()
# first_three_band = data[:,:].copy()
# Convert to float
first_three_band = first_three_band.astype(np.float32)
# Normalize
for i in range(first_three_band.shape[2]):
    first_three_band[:, :, i] = (first_three_band[:, :, i] - np.min(first_three_band[:, :, i])) / (np.max(first_three_band[:, :, i]) - np.min(first_three_band[:, :, i]))

# Visualization
plt.imshow(first_three_band)
plt.savefig('first_three_band.png')
plt.close()