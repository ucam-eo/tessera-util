import numpy as np

np_file_path = "/shared/amdgpu/home/avsm2_f4q/code/btfm4rs/data/ssl_training/ready_to_use_40_steps/aug1/s2/data_B1_F1.npy"
data = np.load(np_file_path, mmap_mode='r')
print(data.shape)  # 输出数组的形状
print(data.dtype)  # 输出数组的数据类型
print(data[0,...])