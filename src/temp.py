import numpy as np

filepath = "/mnt/e/Codes/btfm4rs/data/representation/borneo_representations.npy"
data = np.load(filepath)
print(data.shape)
print(data[0].shape)