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


# import numpy as np
# import matplotlib.pyplot as plt
# band_file_path = "/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/ready_to_use_64_steps/aug1/s2/data_B1_F1.npy"
# band_file_path1 = "/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/ready_to_use_64_steps/aug2/s2/data_B1_F1.npy"
# num_sample = 1

# band_data = np.load(band_file_path, mmap_mode='r')
# band_data1 = np.load(band_file_path1, mmap_mode='r')
# print(band_data.shape)  # 输出数组的形状
# doy = band_data[num_sample,:,-1]
# #排序
# sorted_indices = np.argsort(doy)
# doy = doy[sorted_indices]

# doy1 = band_data1[num_sample,:,-1]
# #排序
# sorted_indices1 = np.argsort(doy1)
# doy1 = doy1[sorted_indices1]

# print(doy)
# print(doy1)


# import torch
# from collections import OrderedDict

# # 安全加载检查点（处理FSDP分片和非张量值）
# checkpoint = torch.load(
#     'checkpoints/ssl/best_model_fsdp_20250407_195912.pt',
#     weights_only=True,
#     map_location=torch.device('cpu')  # 避免显存问题
# )

# # 1. 检查模型状态字典是否存在
# if 'model_state_dict' in checkpoint:
#     print("Model Structure & Shapes:")
#     # 提取并遍历 OrderedDict 中的参数
#     model_state_dict = checkpoint['model_state_dict']
#     if isinstance(model_state_dict, OrderedDict):
#         for param_name, param_tensor in model_state_dict.items():
#             # 仅处理张量类型的参数
#             if isinstance(param_tensor, torch.Tensor):
#                 print(f"  - Layer: {param_name}\n    Shape: {param_tensor.shape}")
#             else:
#                 print(f"  - Layer: {param_name} (non-tensor type: {type(param_tensor)})")
#     else:
#         print(f"model_state_dict 不是 OrderedDict，而是 {type(model_state_dict)}")
# else:
#     print("检查点中没有 model_state_dict 键！")


# patch_rbg = band_data[num_sample, :, :, 4, 0]
# # 归一化
# patch_rbg = (patch_rbg - np.min(patch_rbg)) / (np.max(patch_rbg) - np.min(patch_rbg))

# # 可视化
# plt.imshow(patch_rbg)
# plt.savefig('rgb.png')
# plt.close()

# sar_file_path = "data/ssl_training/ready_to_use_patch/aug1/s1/data_B1_F1.npy"
# sar_band_data = np.load(sar_file_path, mmap_mode='r')
# print(sar_band_data.shape)  # 输出数组的形状
# print(sar_band_data[num_sample,:,:,0,0])

# patch_sar = sar_band_data[num_sample, :, :, 6, :1]  # 获取第一个时间步的SAR波段
# # 归一化
# patch_sar = (patch_sar - np.min(patch_sar)) / (np.max(patch_sar) - np.min(patch_sar))
# # 可视化
# plt.imshow(patch_sar)
# plt.savefig('sar.png')
# plt.close()

# import numpy as np

# mask_path = "data/raw_tile_data/robin_fungal/35VME/masks.npy"
# mask = np.load(mask_path, mmap_mode='r')
# print(mask.shape)  # (T,H,W)
# # 找出那些第二和第三个维度基本全为0的时间步
# mask_sum = np.sum(mask, axis=(1,2))
# zero_threshold = 0.05 * mask.shape[1] * mask.shape[2]  # 95% 的像素都是0
# invalid_idx = np.where(mask_sum <= zero_threshold)[0]
# print("invalid time step:", invalid_idx)
# print(f"invalid time step count: {len(invalid_idx)}")

import numpy as np
import matplotlib.pyplot as plt
# file_path = "data/representation/austrian_crop/representations_fsdp_20250427_084307.npy"
# file_path = "data/representation/austrian_crop/representations_fsdp_20250427_084307_repreat_5.npy"
# file_path = "data/representation/austrian_crop/whole_year_representations_fsdp_20250417_101636.npy"
file_path = "data/representation/austrian_crop/representations_fsdp_20250427_084307_repreat_1.npy"
# file_path = "data/representation/austrian_crop/representations_fsdp_20250427_084307_repreat_10.npy"
data = np.load(file_path, mmap_mode='r')
# data = np.load(file_path)
print(data.shape)  # 输出数组的形状 (T,H,W)
rgb = data[::1, ::1, 30:33].copy()  # 获取第一个时间步的第一个波段
# 归一化
for i in range(3):
    rgb[:, :, i] = (rgb[:, :, i] - np.min(rgb[:, :, i])) / (np.max(rgb[:, :, i]) - np.min(rgb[:, :, i]))
# 可视化
plt.imshow(rgb)
plt.savefig('rgb1.png')
plt.close()

bands_file_path = "data/downstream/austrian_crop/whole_year_data/33UXP/bands.npy"
# bands_file_path = "data/downstream/austrian_crop/bands.npy"

bands_data = np.load(bands_file_path, mmap_mode='r')
# bands_data = np.load(bands_file_path)
print(bands_data.shape)  # 输出数组的形状 (T,H,W)
bands_rgb = bands_data[20, :, :, 3:6].copy()  # 获取第一个时间步的第一个波段
# 转为float
bands_rgb = bands_rgb.astype(np.float32)
# 归一化
for i in range(3):
    bands_rgb[:, :, i] = (bands_rgb[:, :, i] - np.min(bands_rgb[:, :, i])) / (np.max(bands_rgb[:, :, i]) - np.min(bands_rgb[:, :, i]))
# 可视化
plt.imshow(bands_rgb)
plt.savefig('rgb1_bands.png')
plt.close()

# # 遍历时间维度，找出不为0的元素最多的时间步
# nonzero_count = np.count_nonzero(data, axis=(1, 2))
# max_idx = np.argmax(nonzero_count)
# print(f"valid time step:", max_idx)

# sar_time_step = 0

# # 用memmap
# sar_file_path = "/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/tiles/33UXU/sar_ascending.npy"
# data = np.load(sar_file_path, mmap_mode='r')
# print(data.shape)  # 输出数组的形状
# print(data.dtype)  # 输出数组的数据类型
# print(data[sar_time_step,0:10,0:10,:])

# single_image = data[sar_time_step, :, :, 0]  # 获取第一个时间步的第一个波段  /media/12TBNVME/frankfeng/btfm4rs/configs
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