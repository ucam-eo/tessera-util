# import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端
# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# training_ratios = [0.01, 0.05, 0.1, 0.3]
# data = {
#     "Representation Map": [(11.7468, 0.3611), (10.3378, 0.4357), (8.7605, 0.5529), (5.2077, 0.7822)],
#     "S2 RGB": [(15.1084, 0.0038), (13.0802, 0.2113), (10.0813, 0.4440), (9.97, 0.4479)],
#     "S2 MSI": [(15.3084, 0.0071), (13.1827, 0.2215), (10.0245, 0.4592), (8.78, 0.4457)],
#     "S2 MSI + S1 SAR": [(14.628, 0.0169), (12.862, 0.2154), (10.146, 0.4676), (8.11, 0.4982)]
# }

# # 颜色字典
# colors = {
#     "Representation Map": "tab:blue",
#     "S2 RGB": "tab:orange",
#     "S2 MSI": "tab:green",
#     "S2 MSI + S1 SAR": "tab:red"
# }

# # Figure 1: MAE
# plt.figure(figsize=(8, 6))
# for label, values in data.items():
#     mae_values = [v[0] for v in values]
#     plt.plot(training_ratios, mae_values, marker='o', markersize=6, alpha=0.6,
#              linestyle='-', color=colors[label], label=label)
# plt.xlabel("Training Ratio")
# plt.ylabel("MAE")
# plt.title("Mean Absolute Error (MAE)")
# plt.xticks(training_ratios)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.savefig("figure1.png", dpi=300)
# plt.close()

# # Figure 2: R²
# plt.figure(figsize=(8, 6))
# for label, values in data.items():
#     r2_values = [v[1] for v in values]
#     plt.plot(training_ratios, r2_values, marker='o', markersize=6, alpha=0.6,
#              linestyle='-', color=colors[label], label=label)
# plt.xlabel("Training Ratio")
# plt.ylabel("R²")
# plt.title("R² Score")
# plt.xticks(training_ratios)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.savefig("figure2.png", dpi=300)
# plt.close()

import numpy as np

file_path = "data/ssl_training/ready_to_use_temp/aug1/s2/data_B1_F1.npy"
data = np.load(file_path, mmap_mode='r')
print(data.shape)
# print(data)
# 随机生成0-1000000之间的20个整数
# indices = np.random.randint(0, 1000000, 20)
# print(data[indices, :, :])  # 0.0



# import numpy as np
# import matplotlib.pyplot as plt

# file_path = "/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/austrian_crop/sar_descending_downsample_100.npy"
# data = np.load(file_path, mmap_mode='r')
# print(f"Number of non-zero elements: {np.count_nonzero(data)}")
# # 遍历时间维度，找出不为0的元素最多的时间步
# nonzero_count = np.count_nonzero(data, axis=(1,2,3))
# max_idx = np.argmax(nonzero_count)
# print(f"valid time step:", max_idx)
# # 用memmap
# # data = np.memmap(file_path, dtype='int16', mode='r', shape=(142, 10980, 10980, 10))
# sar_time_step = max_idx

# print(data.shape)  # 输出数组的形状
# print(data.dtype)  # 输出数组的数据类型
# print(data[sar_time_step,0:10,0:10,:])

# single_image = data[sar_time_step, :, :, 0]  # 获取第一个时间步的第一个波段
# plt.imshow(single_image, cmap='gray')
# plt.savefig('sar_0.png')
# plt.close()

# single_image = data[sar_time_step, :, :, 1]  # 获取第一个时间步的第一个波段
# plt.imshow(single_image, cmap='gray')
# plt.savefig('sar_1.png')
# plt.close()

# band_file_path = "/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/austrian_crop/bands_downsample_100.npy"
# band_data = np.load(band_file_path, mmap_mode='r')
# print(band_data.shape)  # 输出数组的形状

# mask_file_path = "/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/austrian_crop/masks_downsample_100.npy"
# mask_data = np.load(mask_file_path) # (T,H,W)
# # 找出含有最多1的时间步
# mask_sum = np.sum(mask_data, axis=(1,2))
# max_idx = np.argmax(mask_sum)
# print("valid time step:", max_idx)

# rgb_time_step = max_idx

# single_rbg_image = band_data[rgb_time_step, :, :, :3]  # 获取第一个时间步的RGB波段
# # 转为float
# single_rbg_image = single_rbg_image.astype(np.float32)
# # 转为rgb
# single_rbg_image = single_rbg_image[:, :, [2, 1, 0]]
# # 归一化
# for i in range(3):
#     single_rbg_image[:, :, i] = (single_rbg_image[:, :, i] - np.min(single_rbg_image[:, :, i])) / (np.max(single_rbg_image[:, :, i]) - np.min(single_rbg_image[:, :, i]))

# import cv2

# def histogram_equalization(image):
#     for i in range(3):
#         image[:,:,i] = cv2.equalizeHist((image[:,:,i] * 255).astype(np.uint8)) / 255.0
#     return image

# single_rbg_image = histogram_equalization(single_rbg_image)    

# plt.imshow(single_rbg_image)
# plt.savefig('rgb.png')
# plt.close()

# doy_file_path = "/scratch/zf281/data_processed/37NFG/doys.npy"
# doy_data = np.load(doy_file_path, mmap_mode='r')
# print(doy_data.shape)  # 输出数组的形状

# sar_doy_file_path = "/scratch/zf281/data_processed/37NFG/sar_ascending_doy.npy"
# sar_doy_data = np.load(sar_doy_file_path, mmap_mode='r')
# print(sar_doy_data.shape)  # 输出数组的形状

# mask_file_path = "/scratch/zf281/data_processed/37NFG/masks.npy"
# mask_data = np.load(mask_file_path, mmap_mode='r')
# print(mask_data.shape)  # 输出数组的形状
# print(mask_data[0, 100:120, 100:120])