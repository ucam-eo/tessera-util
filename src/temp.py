# import numpy as np

# left_down_x = 1500
# left_down_y = 1500
# right_up_x = 2000
# right_up_y = 2000

# x = 0
# y = 0

# rgb_time_index = 8

# band_file_path = "/scratch/zf281/downstream_dataset/austrian_whole_year/data_processed/33UXP/bands.npy"
# band = np.load(band_file_path, mmap_mode='r')
# print(band.dtype)
# print(band.shape)

# # # 遍历所有时间步
# # for i in range(band.shape[0]):
# #     # 取出当前时间步的数据
# #     current_time_step = band[i,::20, ::20, 6:9].copy()
# #     # 转为float
# #     current_time_step = current_time_step.astype(np.float32)
# #     for j in range(3):
# #         # 归一化到[0, 1]
# #         current_time_step[:, :, j] = (current_time_step[:, :, j] - current_time_step[:, :, j].min()) / (current_time_step[:, :, j].max() - current_time_step[:, :, j].min()+1e-8)
# #     # 单通道归一化
# #     # current_time_step = (current_time_step - current_time_step.min()) / (current_time_step.max() - current_time_step.min())
# #     # 可视化
# #     import matplotlib.pyplot as plt
# #     plt.imshow(current_time_step)
# #     save_name = f"temp_log/{i}_timestep.png"
# #     plt.imsave(save_name, current_time_step)
# #     print(f"保存为: {save_name}")

# mask_path = "/scratch/zf281/downstream_dataset/austrian_whole_year/data_processed/33UXP/masks.npy"
# mask = np.load(mask_path, mmap_mode='r') # （T, H, W）
# print(mask.shape)

# # 可视化rgb
# tile_rgb = band[rgb_time_index, left_down_y:left_down_y + 500, left_down_x:left_down_x + 500, 3:6].copy()
# # 转为float
# tile_rgb = tile_rgb.astype(np.float32)
# # 归一化到[0, 1]
# for i in range(3):
#     tile_rgb[:, :, i] = (tile_rgb[:, :, i] - tile_rgb[:, :, i].min()) / (tile_rgb[:, :, i].max() - tile_rgb[:, :, i].min())
# import matplotlib.pyplot as plt
# # plt.imshow(tile_rgb.transpose(1, 2, 0))
# save_name = f"rgb_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}.png"
# plt.imsave(save_name, tile_rgb)
# print(f"保存为: {save_name}")

# tile_file_path = f"/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled/{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}/bands.npy"
# tile_band = np.load(tile_file_path, mmap_mode='r')


# # 可视化rgb
# tile_rgb = tile_band[rgb_time_index, :, :, 3:6].copy()
# # 转为float
# tile_rgb = tile_rgb.astype(np.float32)
# # 归一化到[0, 1]
# for i in range(3):
#     tile_rgb[:, :, i] = (tile_rgb[:, :, i] - tile_rgb[:, :, i].min()) / (tile_rgb[:, :, i].max() - tile_rgb[:, :, i].min())
# # plt.imshow(tile_rgb.transpose(1, 2, 0))
# save_name = f"rgb_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}_1.png"
# plt.imsave(save_name, tile_rgb)
# print(f"保存为: {save_name}")



# tile_mask_path = f"/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled/{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}/masks.npy"
# tile_mask = np.load(tile_mask_path, mmap_mode='r')
# print(tile_mask.shape)
# tile_mask_roi = tile_mask[:, y, x]
# print(tile_mask_roi)
# print(np.sum(tile_mask_roi))

# print(f"mask for time step: {rgb_time_index}")
# tile_mask_timestep = tile_mask[rgb_time_index, :, :].copy()
# # 变为0-255
# tile_mask_timestep = tile_mask_timestep* 255
# # 转为uint8
# tile_mask_timestep = tile_mask_timestep.astype(np.uint8)
# plt.imshow(tile_mask_timestep, cmap='gray')
# save_name = f"mask_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}.png"
# plt.imsave(save_name, tile_mask_timestep)
# print(f"保存为: {save_name}")
# plt.close()

# # Sar
# time_sar_index = 100

# sar_band_file_path = "/scratch/zf281/downstream_dataset/austrian_whole_year/data_processed/33UXP/sar_ascending.npy"
# sar_band = np.load(sar_band_file_path, mmap_mode='r')
# print(sar_band.shape)
# # 可视化
# sar_rgb = sar_band[time_sar_index, left_down_y:left_down_y + 500, left_down_x:left_down_x + 500, 0].copy()
# # 转为float
# sar_rgb = sar_rgb.astype(np.float32)
# # 归一化到[0, 1]
# sar_rgb = (sar_rgb - sar_rgb.min()) / (sar_rgb.max() - sar_rgb.min())
# plt.imshow(sar_rgb)
# save_name = f"rgb_sar_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}.png"
# plt.imsave(save_name, sar_rgb)
# print(f"保存为: {save_name}")
# plt.close()

# # tile sar
# tile_sar_band_file_path = f"/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled/{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}/sar_ascending.npy"
# tile_sar_band = np.load(tile_sar_band_file_path, mmap_mode='r')
# print(tile_sar_band.shape)
# # 可视化
# tile_sar_rgb = tile_sar_band[time_sar_index, :, :, 0].copy()
# # 转为float
# tile_sar_rgb = tile_sar_rgb.astype(np.float32)
# # 归一化到[0, 1]
# tile_sar_rgb = (tile_sar_rgb - tile_sar_rgb.min()) / (tile_sar_rgb.max() - tile_sar_rgb.min())
# plt.imshow(tile_sar_rgb)
# save_name = f"rgb_sar_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}_1.png"
# plt.imsave(save_name, tile_sar_rgb)
# print(f"保存为: {save_name}")
# plt.close()

# base_dir = "/scratch/zf281/downstream_dataset/austrian_whole_year/representation_retiled"
# # 获取下面的所有npy
# import os
# npy_files = []
# for root, dirs, files in os.walk(base_dir):
#     for file in files:
#         if file.endswith(".npy"):
#             npy_files.append(os.path.join(root, file))
# print(f"找到 {len(npy_files)} 个npy文件")

# # for band_file_path in npy_files:
# #     print(f"使用文件: {band_file_path}")
# #     band = np.load(band_file_path, mmap_mode='r')
# #     print(band.shape)
# #     # 可视化前三个波段
# #     rgb = band[:, :, :3]
# #     # 归一化到[0, 1]
# #     rgb = rgb.astype(np.float32)
# #     for i in range(3):
# #         rgb[:, :, i] = (rgb[:, :, i] - rgb[:, :, i].min()) / (rgb[:, :, i].max() - rgb[:, :, i].min())
# #     import matplotlib.pyplot as plt
# #     plt.imshow(rgb)
# #     save_name = f"temp_log/rgb_{os.path.basename(band_file_path)}.png"
# #     plt.imsave(save_name, rgb)
# #     plt.close()

# band_file_path = f"/scratch/zf281/downstream_dataset/austrian_whole_year/representation_retiled/{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}.npy"
# # 取第一个
# # band_file_path = npy_files[0]
# print(f"使用文件: {band_file_path}")
# # band = np.load(band_file_path, mmap_mode='r')
# band = np.load(band_file_path)
# print(band.shape)
# # 可视化前三个波段
# rgb = band[:, :, :3]
# # 归一化到[0, 1]
# rgb = rgb.astype(np.float32)
# for i in range(3):
#     rgb[:, :, i] = (rgb[:, :, i] - rgb[:, :, i].min()) / (rgb[:, :, i].max() - rgb[:, :, i].min())
# import matplotlib.pyplot as plt
# plt.imshow(rgb)
# save_name = f"temp_log/rgb_{os.path.basename(band_file_path)}.png"
# plt.imsave(save_name, rgb)
# print(f"保存为: {save_name}")
# plt.close()

import numpy as np
file_path = "/maps/zf281/btfm4rs/data/tmp/tmp_map_10m_utm28n_rgb.npy"
band = np.load(file_path, mmap_mode='r')
print(band.shape)
# # 打印最大最小值
# print(f"最大值: {band.max()}")
# print(f"最小值: {band.min()}")

# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt

# def visualize_tiff(tiff_path):
#     """
#     读取TIFF文件的前三个波段，将其值从[-127, 127]转换为[0, 255]并进行可视化。

#     参数:
#     tiff_path (str): TIFF文件的路径。
#     """
#     try:
#         with rasterio.open(tiff_path) as src:
#             # 读取前三个波段
#             # rasterio读取的波段索引从1开始
#             img_data = src.read([1, 2, 3])

#             # 将数据从int8 (范围-127到127) 转换为 uint8 (范围0-255)
#             # 首先将数据类型转换为int16以避免溢出，然后加上127
#             img_data_scaled = (img_data.astype(np.int16) + 127).astype(np.uint8)

#             # rasterio读取的数组形状为 (通道, 高度, 宽度)
#             # matplotlib.pyplot.imshow 需要的形状为 (高度, 宽度, 通道)
#             # 因此需要转换数组的维度
#             img_to_show = np.transpose(img_data_scaled, (1, 2, 0))

#             # 使用matplotlib显示图像
#             plt.figure(figsize=(10, 10))
#             plt.imshow(img_to_show)
#             # plt.title('TIFF Image - First three bands')
#             # plt.xlabel('Width')
#             # plt.ylabel('Height')
#             # plt.show()
#             plt.imsave('visualized_image.png', img_to_show)
#             print("图像已保存为 'visualized_image.png'")
#             plt.close()

#     except Exception as e:
#         print(f"发生错误: {e}")

# if __name__ == '__main__':
#     # 请将这个路径替换为您TIFF文件的实际路径
#     tiff_file_path = '/maps/zf281/btfm4rs/senegal_map_10m_wgs84_128bands.tiff'
#     visualize_tiff(tiff_file_path)
    
# import asf_search as asf

# results = asf.granule_search(['ALPSRS279162400', 'ALPSRS279162200'])
# print(results)

# wkt = 'POLYGON((-135.7 58.2,-136.6 58.1,-135.8 56.9,-134.6 56.1,-134.9 58.0,-135.7 58.2))'
# results = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], intersectsWith=wkt, maxResults=10)
# print(results)

# import os
# import numpy as np

# # 设置根目录路径
# base_dir = "/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled"

# # 遍历所有子文件夹
# for subdir in os.listdir(base_dir):
#     subdir_path = os.path.join(base_dir, subdir)
    
#     # 确保是目录且包含bands.npy
#     if os.path.isdir(subdir_path):
#         bands_path = os.path.join(subdir_path, "bands.npy")
        
#         if os.path.exists(bands_path):
#             # 使用内存映射加载数据（避免加载大文件到内存）
#             bands = np.load(bands_path, mmap_mode='r')
#             print(f"{subdir}: {bands.shape}")
#             sub_data = bands[:, 150, 160, 0]
#             print(f"{subdir}: {sub_data}")
#         else:
#             print(f"{subdir}: 未找到bands.npy文件")

# import os
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.patches import Rectangle

# def parse_coordinates(folder_name):
#     """解析文件夹名中的坐标信息 (格式: x_offset_y_offset_width_height)"""
#     match = re.match(r"^(\d+)_(\d+)_(\d+)_(\d+)$", folder_name)
#     if match:
#         return list(map(int, match.groups()))  # [x, y, width, height]
#     return None

# # ================= 配置参数 =================
# base_dir = "/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled"
# big_tiff_path = "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/roi.tif"
# output_path = "tiff_coverage_check.png"
# # ============================================

# # 创建画布
# fig, ax = plt.subplots(figsize=(15, 15))

# # 步骤1：绘制大TIFF（填充内容）
# try:
#     with Image.open(big_tiff_path) as img:
#         big_data = np.array(img)
#         ax.imshow(big_data, 
#                  extent=[0, big_data.shape[1], 0, big_data.shape[0]],  # 假设坐标原点在左下角
#                  cmap='viridis', 
#                  alpha=0.4,  # 半透明填充
#                  label='Reference TIFF')
# except Exception as e:
#     print(f"大TIFF加载失败: {str(e)}")

# # 步骤2：绘制所有小tile的边界框
# total_tiles = 0
# for folder in os.listdir(base_dir):
#     coords = parse_coordinates(folder)
#     if not coords:
#         continue
    
#     tile_path = os.path.join(base_dir, folder, "roi.tiff")
#     if not os.path.exists(tile_path):
#         continue
    
#     try:
#         # 获取实际tile尺寸
#         with Image.open(tile_path) as img:
#             w, h = img.size
        
#         # 绘制边界框
#         rect = Rectangle(
#             (coords[0], coords[1]),  # 左下角坐标
#             w,                      # 宽度
#             h,                      # 高度
#             linewidth=1.5,
#             edgecolor='red',
#             facecolor='none',       # 关键设置：无填充
#             alpha=0.8,
#             label='Tile Boundary' if total_tiles == 0 else ""  # 避免重复图例
#         )
#         ax.add_patch(rect)
#         total_tiles += 1
        
#     except Exception as e:
#         print(f"处理 {folder} 失败: {str(e)}")

# # 步骤3：自动调整坐标轴
# ax.autoscale_view()
# ax.set_xlabel('X Coordinate (pixels)')
# ax.set_ylabel('Y Coordinate (pixels)')
# ax.set_title('Spatial Coverage Verification\n(Red: Tile Boundaries, Blue: Reference Area)', pad=20)

# # 添加智能图例
# handles, labels = ax.get_legend_handles_labels()
# unique_labels = dict(zip(labels, handles))
# ax.legend(unique_labels.values(), unique_labels.keys(), 
#          loc='lower right', framealpha=0.7)

# # 保存并显示
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"验证结果已保存至: {output_path}")
# plt.close()

# sample_index = 20
# time_index = 0
# time_sar_index = 1
# r = band[sample_index, :, :, time_index, 0]
# # 归一化
# r = r.astype(np.float32)
# r = (r - r.min()) / (r.max() - r.min())

# import matplotlib.pyplot as plt
# plt.imshow(r, cmap='gray')
# plt.imsave("r.png", r)
# plt.close()

# sar_band_file_path = "data/ssl_training/ready_to_use_patch/aug1/s1/data_B1_F1.npy"
# sar_band = np.load(sar_band_file_path, mmap_mode='r')
# print(sar_band.shape)
# sar_r = sar_band[sample_index, :, :, time_sar_index, 0]
# # 归一化
# sar_r = sar_r.astype(np.float32)
# sar_r = (sar_r - sar_r.min()) / (sar_r.max() - sar_r.min())
# plt.imshow(sar_r, cmap='gray')
# plt.imsave("sar_r.png", sar_r)
# plt.close()

# filepath = "/scratch/zf281/robin/fungal/data_processed/34VFL/bands.npy"
# doy_filepath = "/scratch/zf281/robin/fungal/data_processed/34VFL/doys.npy"
# # filepath1 = f"/scratch/zf281/global/{tile_code}/sar_descending_doy.npy"
# data = np.load(filepath,mmap_mode='r')
# # data = np.load(filepath)
# print(data.shape)
# print(data.dtype)
# print(data[0,5000:5010,5000:5010, 0])
# print(data[1,5000:5010,5000:5010, 0])

# doy = np.load(doy_filepath)
# print(doy)
# 统计不为0的元素的个数
# print(np.count_nonzero(data))
# print(data[0])

# import numpy as np

# filepath = "/maps/zf281/btfm4rs/data/ssl_training/ready_to_use_64/s1/data_B1_F1.npy"
# # # doy_filepath = "/scratch/zf281/robin/fungal/data_processed/35VLF/doys.npy"
# # # filepath1 = f"/scratch/zf281/global/{tile_code}/sar_descending_doy.npy"
# data = np.load(filepath,mmap_mode='r')
# # data = np.load(filepath)
# print(data.shape)
# print(data.dtype)
# print(data[0,5000:5010,5000:5010, 0])
# print(data[1,5000:5010,5000:5010, 0])

# doy = np.load(doy_filepath)
# print(doy)
# 统计不为0的元素的个数
# print(np.count_nonzero(data))
# print(data[0])

# 定义新的doy数组，注意数据类型与原文件一致为uint16
# new_doys = np.array(
#     [27, 40, 47, 50, 60, 67, 70, 80, 87, 107, 120, 130, 140, 147, 167, 170, 177, 180, 187, 197, 200, 217, 227, 250],
#     dtype=np.uint16
# )

# # 保存为 new_doys.npy 文件
# np.save("new_doys.npy", new_doys)

# # 提取前三个通道
# rbg_time_0 = data[10, :, :, :3]
# # rbg变为rgb
# rbg_time_0 = rbg_time_0[..., [2, 1, 0]]
# # 转为float
# rbg_time_0 = rbg_time_0.astype(np.float32)
# # 归一化后保存为png
# for i in range(3):
#     rbg_time_0[:, :, i] = rbg_time_0[:, :, i] / np.max(rbg_time_0[:, :, i])
# import matplotlib.pyplot as plt
# plt.imshow(rbg_time_0)
# plt.imsave("rbg_time_0.png", rbg_time_0)
# plt.close()

# print(data[20, 5000:5100, 5000:5100, ...])
# 检查第二个通道是否全为0
# print(np.all(data[:,5000:5500,5000:5500, 1] == 0))
# print(data[0].shape)
# print(data)
# print(data1)


# import rasterio
# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_tiff(tiff_path, output_filename='visualized_image.png'):
#     with rasterio.open(tiff_path) as src:
#         count = src.count
#         # 如果有至少3个波段，则认为是RGB图像，否则取第一波段显示
#         if count >= 3:
#             # 读取前三个波段
#             r = src.read(1)
#             g = src.read(2)
#             b = src.read(3)
#             # 简单归一化处理
#             def normalize(array):
#                 array = array.astype(np.float32)
#                 array -= array.min()
#                 if array.max() > 0:
#                     array /= array.max()
#                 return array
#             rgb = np.dstack((normalize(r), normalize(g), normalize(b)))
#             plt.figure(figsize=(10, 10))
#             plt.imshow(rgb)
#             plt.title("RGB Composite")
#         else:
#             # 只读取第一个波段
#             band = src.read(1)
#             plt.figure(figsize=(10, 10))
#             plt.imshow(band, cmap='gray')
#             plt.title("Single Band")
#             plt.colorbar()
    
#     plt.axis('off')
#     plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
#     plt.show()
#     print(f"图像已保存为：{output_filename}")

# if __name__ == '__main__':
#     # 修改为你的 tiff 文件路径
#     tiff_path = '/scratch/zf281/robin/fungal/estonia_roi.tif'
#     visualize_tiff(tiff_path)
