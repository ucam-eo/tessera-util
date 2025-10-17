# # import numpy as np

# # left_down_x = 1500
# # left_down_y = 1500
# # right_up_x = 2000
# # right_up_y = 2000

# # x = 0
# # y = 0

# # rgb_time_index = 8

# # band_file_path = "/scratch/zf281/downstream_dataset/austrian_whole_year/data_processed/33UXP/bands.npy"
# # band = np.load(band_file_path, mmap_mode='r')
# # print(band.dtype)
# # print(band.shape)

# # # # 遍历所有时间步
# # # for i in range(band.shape[0]):
# # #     # 取出当前时间步的数据
# # #     current_time_step = band[i,::20, ::20, 6:9].copy()
# # #     # 转为float
# # #     current_time_step = current_time_step.astype(np.float32)
# # #     for j in range(3):
# # #         # 归一化到[0, 1]
# # #         current_time_step[:, :, j] = (current_time_step[:, :, j] - current_time_step[:, :, j].min()) / (current_time_step[:, :, j].max() - current_time_step[:, :, j].min()+1e-8)
# # #     # 单通道归一化
# # #     # current_time_step = (current_time_step - current_time_step.min()) / (current_time_step.max() - current_time_step.min())
# # #     # 可视化
# # #     import matplotlib.pyplot as plt
# # #     plt.imshow(current_time_step)
# # #     save_name = f"temp_log/{i}_timestep.png"
# # #     plt.imsave(save_name, current_time_step)
# # #     print(f"保存为: {save_name}")

# # mask_path = "/scratch/zf281/downstream_dataset/austrian_whole_year/data_processed/33UXP/masks.npy"
# # mask = np.load(mask_path, mmap_mode='r') # （T, H, W）
# # print(mask.shape)

# # # 可视化rgb
# # tile_rgb = band[rgb_time_index, left_down_y:left_down_y + 500, left_down_x:left_down_x + 500, 3:6].copy()
# # # 转为float
# # tile_rgb = tile_rgb.astype(np.float32)
# # # 归一化到[0, 1]
# # for i in range(3):
# #     tile_rgb[:, :, i] = (tile_rgb[:, :, i] - tile_rgb[:, :, i].min()) / (tile_rgb[:, :, i].max() - tile_rgb[:, :, i].min())
# # import matplotlib.pyplot as plt
# # # plt.imshow(tile_rgb.transpose(1, 2, 0))
# # save_name = f"rgb_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}.png"
# # plt.imsave(save_name, tile_rgb)
# # print(f"保存为: {save_name}")

# # tile_file_path = f"/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled/{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}/bands.npy"
# # tile_band = np.load(tile_file_path, mmap_mode='r')


# # # 可视化rgb
# # tile_rgb = tile_band[rgb_time_index, :, :, 3:6].copy()
# # # 转为float
# # tile_rgb = tile_rgb.astype(np.float32)
# # # 归一化到[0, 1]
# # for i in range(3):
# #     tile_rgb[:, :, i] = (tile_rgb[:, :, i] - tile_rgb[:, :, i].min()) / (tile_rgb[:, :, i].max() - tile_rgb[:, :, i].min())
# # # plt.imshow(tile_rgb.transpose(1, 2, 0))
# # save_name = f"rgb_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}_1.png"
# # plt.imsave(save_name, tile_rgb)
# # print(f"保存为: {save_name}")



# # tile_mask_path = f"/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled/{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}/masks.npy"
# # tile_mask = np.load(tile_mask_path, mmap_mode='r')
# # print(tile_mask.shape)
# # tile_mask_roi = tile_mask[:, y, x]
# # print(tile_mask_roi)
# # print(np.sum(tile_mask_roi))

# # print(f"mask for time step: {rgb_time_index}")
# # tile_mask_timestep = tile_mask[rgb_time_index, :, :].copy()
# # # 变为0-255
# # tile_mask_timestep = tile_mask_timestep* 255
# # # 转为uint8
# # tile_mask_timestep = tile_mask_timestep.astype(np.uint8)
# # plt.imshow(tile_mask_timestep, cmap='gray')
# # save_name = f"mask_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}.png"
# # plt.imsave(save_name, tile_mask_timestep)
# # print(f"保存为: {save_name}")
# # plt.close()

# # # Sar
# # time_sar_index = 100

# # sar_band_file_path = "/scratch/zf281/downstream_dataset/austrian_whole_year/data_processed/33UXP/sar_ascending.npy"
# # sar_band = np.load(sar_band_file_path, mmap_mode='r')
# # print(sar_band.shape)
# # # 可视化
# # sar_rgb = sar_band[time_sar_index, left_down_y:left_down_y + 500, left_down_x:left_down_x + 500, 0].copy()
# # # 转为float
# # sar_rgb = sar_rgb.astype(np.float32)
# # # 归一化到[0, 1]
# # sar_rgb = (sar_rgb - sar_rgb.min()) / (sar_rgb.max() - sar_rgb.min())
# # plt.imshow(sar_rgb)
# # save_name = f"rgb_sar_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}.png"
# # plt.imsave(save_name, sar_rgb)
# # print(f"保存为: {save_name}")
# # plt.close()

# # # tile sar
# # tile_sar_band_file_path = f"/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled/{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}/sar_ascending.npy"
# # tile_sar_band = np.load(tile_sar_band_file_path, mmap_mode='r')
# # print(tile_sar_band.shape)
# # # 可视化
# # tile_sar_rgb = tile_sar_band[time_sar_index, :, :, 0].copy()
# # # 转为float
# # tile_sar_rgb = tile_sar_rgb.astype(np.float32)
# # # 归一化到[0, 1]
# # tile_sar_rgb = (tile_sar_rgb - tile_sar_rgb.min()) / (tile_sar_rgb.max() - tile_sar_rgb.min())
# # plt.imshow(tile_sar_rgb)
# # save_name = f"rgb_sar_{left_down_x}_{left_down_y}_{right_up_x}_{right_up_y}_1.png"
# # plt.imsave(save_name, tile_sar_rgb)
# # print(f"保存为: {save_name}")
# # plt.close()

# # base_dir = "/scratch/zf281/downstream_dataset/austrian_whole_year/representation_retiled"
# # # 获取下面的所有npy
# # import os
# # npy_files = []
# # for root, dirs, files in os.walk(base_dir):
# #     for file in files:
# #         if file.endswith(".npy"):
# #             npy_files.append(os.path.join(root, file))
# # print(f"找到 {len(npy_files)} 个npy文件")

# # # for band_file_path in npy_files:
# # #     print(f"使用文件: {band_file_path}")
# # #     band = np.load(band_file_path, mmap_mode='r')
# # #     print(band.shape)
# # #     # 可视化前三个波段
# # #     rgb = band[:, :, :3]
# # #     # 归一化到[0, 1]
# # #     rgb = rgb.astype(np.float32)
# # #     for i in range(3):
# # #         rgb[:, :, i] = (rgb[:, :, i] - rgb[:, :, i].min()) / (rgb[:, :, i].max() - rgb[:, :, i].min())
# # #     import matplotlib.pyplot as plt
# # #     plt.imshow(rgb)
# # #     save_name = f"temp_log/rgb_{os.path.basename(band_file_path)}.png"
# # #     plt.imsave(save_name, rgb)
# # #     plt.close()
import numpy as np
import os
import matplotlib.pyplot as plt

band_file_path = "/maps/zf281/btfm4rs/data/downstream/austrian_crop/raw/patch_64_64/train/s2/1001.npy"
# 取第一个
# band_file_path = npy_files[0]
print(f"使用文件: {band_file_path}")
band = np.load(band_file_path, mmap_mode='r')
# band = np.load(band_file_path)
print(band.shape)
# 最大最小值
# print(f"最大值: {band.max()}")
# print(f"最小值: {band.min()}")
# 可视化前三个波段
# rgb = band[0, :3, :, :] # (3, H, W)
# # transpose to HWC
# rgb = np.transpose(rgb, (1, 2, 0))
# rgb = band[:, :, :3] # (H, W, 3)
# # 归一化到[0, 1]
# rgb = rgb.astype(np.float32)
# for i in range(3):
#     rgb[:, :, i] = (rgb[:, :, i] - rgb[:, :, i].min()) / (rgb[:, :, i].max() - rgb[:, :, i].min())
# import matplotlib.pyplot as plt
# plt.imshow(rgb)
# save_name = "/maps/zf281/btfm4rs/test0.png"
# plt.imsave(save_name, rgb)
# print(f"保存为: {save_name}")
# plt.close()

# band_file_path = "/maps/zf281/btfm4rs/data/downstream/austrian_crop/alphaearth/patch_64_64/test/label_20.npy"
# # 取第一个
# # band_file_path = npy_files[0]
# print(f"使用文件: {band_file_path}")
# band = np.load(band_file_path, mmap_mode='r')
# # band = np.load(band_file_path)
# print(band.shape)
# #最大最小值
# print(f"最大值: {band.max()}")
# print(f"最小值: {band.min()}")

# import matplotlib.pyplot as plt
# plt.imshow(band)
# save_name = "/maps/zf281/btfm4rs/test1.png"
# plt.imsave(save_name, band)
# print(f"保存为: {save_name}")
# plt.close()



# band_file_path = "/scratch/zf281/pangaea-bench/data/PASTIS-HD/DATA_S2/S2_10002.npy"
# # 取第一个
# # band_file_path = npy_files[0]
# print(f"使用文件: {band_file_path}")
# # band = np.load(band_file_path, mmap_mode='r')
# band = np.load(band_file_path)
# print(band.shape)
# #最大最小值
# print(f"最大值: {band.max()}")
# print(f"最小值: {band.min()}")
# # 可视化前三个波段
# rgb = band[1, :3, :, :] # (3, H, W)
# # # transpose to HWC
# rgb = np.transpose(rgb, (1, 2, 0))
# # rgb = band[:, :, :3] # (H, W, 3)
# # 归一化到[0, 1]
# rgb = rgb.astype(np.float32)
# for i in range(3):
#     rgb[:, :, i] = (rgb[:, :, i] - rgb[:, :, i].min()) / (rgb[:, :, i].max() - rgb[:, :, i].min())
# import matplotlib.pyplot as plt
# plt.imshow(rgb)
# save_name = "/maps/zf281/btfm4rs/test1.png"
# plt.imsave(save_name, rgb)
# print(f"保存为: {save_name}")
# plt.close()

# import numpy as np
# import sys

# def calculate_average_cosine_similarity(file1, file2):
#     """
#     加载两个Numpy embedding文件，计算它们在每个像素上的余弦相似度，
#     然后返回所有像素的平均相似度。

#     参数:
#     file1 (str): 第一个.npy文件的路径。
#     file2 (str): 第二个.npy文件的路径。

#     返回:
#     float: 平均余弦相似度，如果发生错误则返回None。
#     """
#     # --- 数据加载与验证 ---
#     try:
#         embedding1 = np.load(file1)
#         embedding2 = np.load(file2)
#     except FileNotFoundError as e:
#         print(f"错误: 找不到文件 {e.filename}", file=sys.stderr)
#         return None
#     except Exception as e:
#         print(f"加载文件时出错: {e}", file=sys.stderr)
#         return None

#     # 验证形状是否匹配
#     if embedding1.shape != embedding2.shape:
#         print(f"错误: 输入文件的形状不匹配: {embedding1.shape} vs {embedding2.shape}", file=sys.stderr)
#         return None
        
#     # 为保证计算精度，将数据类型转为浮点数
#     embedding1 = embedding1.astype(np.float32)
#     embedding2 = embedding2.astype(np.float32)

#     # --- 使用矩阵运算计算余弦相似度 ---

#     # 1. 计算点积 (A dot B)
#     # np.sum(embedding1 * embedding2, axis=-1) 会沿着最后一个维度(C)进行元素乘积并求和
#     # 结果是一个形状为 (H, W) 的矩阵
#     dot_product = np.sum(embedding1 * embedding2, axis=-1)

#     # 2. 计算每个向量的范数 (||A|| 和 ||B||)
#     # np.linalg.norm会计算每个像素向量的L2范数，结果是两个(H, W)形状的矩阵
#     norm1 = np.linalg.norm(embedding1, axis=-1)
#     norm2 = np.linalg.norm(embedding2, axis=-1)

#     # 3. 计算余弦相似度: (A dot B) / (||A|| * ||B||)
#     # 为了防止除以零，在分母中添加一个很小的数 epsilon
#     epsilon = 1e-8
#     cosine_similarity_map = dot_product / (norm1 * norm2 + epsilon)

#     # 4. 计算所有像素的平均余弦相似度
#     average_similarity = np.mean(cosine_similarity_map)

#     return average_similarity

# # --- 主程序入口 ---
# if __name__ == "__main__":
#     # !!!重要!!! 请将下面的路径替换为您文件的真实路径
#     file_path1 = '/scratch/zf281/downstream_dataset/discard_timesteps/40_fsdp_20250408_101211.npy'
#     file_path2 = '/scratch/zf281/downstream_dataset/discard_timesteps/5_100_fsdp_20250408_101211.npy'

#     # 执行计算
#     avg_sim = calculate_average_cosine_similarity(file_path1, file_path2)

#     # 如果计算成功，则打印结果
#     if avg_sim is not None:
#         print(f"文件1: {file_path1}")
#         print(f"文件2: {file_path2}")
#         print(f"平均余弦相似度为: {avg_sim:.8f}")




# # import numpy as np
# # file_path = "/scratch/zf281/btfm_representation/senegal/representation/2018_representation_map_10m_utm28n_scales.npy"
# # band = np.load(file_path, mmap_mode='r')
# # print(band.shape)
# # # # 打印最大最小值
# # print(f"最大值: {band.max()}")
# # print(f"最小值: {band.min()}")

# # import rasterio
# # import numpy as np
# # import matplotlib.pyplot as plt

# # def visualize_tiff(tiff_path):
# #     """
# #     读取TIFF文件的前三个波段，将其值从[-127, 127]转换为[0, 255]并进行可视化。

# #     参数:
# #     tiff_path (str): TIFF文件的路径。
# #     """
# #     try:
# #         with rasterio.open(tiff_path) as src:
# #             # 读取前三个波段
# #             # rasterio读取的波段索引从1开始
# #             img_data = src.read([1, 2, 3])

# #             # 将数据从int8 (范围-127到127) 转换为 uint8 (范围0-255)
# #             # 首先将数据类型转换为int16以避免溢出，然后加上127
# #             img_data_scaled = (img_data.astype(np.int16) + 127).astype(np.uint8)

# #             # rasterio读取的数组形状为 (通道, 高度, 宽度)
# #             # matplotlib.pyplot.imshow 需要的形状为 (高度, 宽度, 通道)
# #             # 因此需要转换数组的维度
# #             img_to_show = np.transpose(img_data_scaled, (1, 2, 0))

# #             # 使用matplotlib显示图像
# #             plt.figure(figsize=(10, 10))
# #             plt.imshow(img_to_show)
# #             # plt.title('TIFF Image - First three bands')
# #             # plt.xlabel('Width')
# #             # plt.ylabel('Height')
# #             # plt.show()
# #             plt.imsave('visualized_image.png', img_to_show)
# #             print("图像已保存为 'visualized_image.png'")
# #             plt.close()

# #     except Exception as e:
# #         print(f"发生错误: {e}")

# # if __name__ == '__main__':
# #     # 请将这个路径替换为您TIFF文件的实际路径
# #     tiff_file_path = '/maps/zf281/btfm4rs/senegal_map_10m_wgs84_128bands.tiff'
# #     visualize_tiff(tiff_file_path)
    
# # import asf_search as asf

# # results = asf.granule_search(['ALPSRS279162400', 'ALPSRS279162200'])
# # print(results)

# # wkt = 'POLYGON((-135.7 58.2,-136.6 58.1,-135.8 56.9,-134.6 56.1,-134.9 58.0,-135.7 58.2))'
# # results = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], intersectsWith=wkt, maxResults=10)
# # print(results)

# # import os
# # import numpy as np

# # # 设置根目录路径
# # base_dir = "/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled"

# # # 遍历所有子文件夹
# # for subdir in os.listdir(base_dir):
# #     subdir_path = os.path.join(base_dir, subdir)
    
# #     # 确保是目录且包含bands.npy
# #     if os.path.isdir(subdir_path):
# #         bands_path = os.path.join(subdir_path, "bands.npy")
        
# #         if os.path.exists(bands_path):
# #             # 使用内存映射加载数据（避免加载大文件到内存）
# #             bands = np.load(bands_path, mmap_mode='r')
# #             print(f"{subdir}: {bands.shape}")
# #             sub_data = bands[:, 150, 160, 0]
# #             print(f"{subdir}: {sub_data}")
# #         else:
# #             print(f"{subdir}: 未找到bands.npy文件")

# # import os
# # import re
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from PIL import Image
# # from matplotlib.patches import Rectangle

# # def parse_coordinates(folder_name):
# #     """解析文件夹名中的坐标信息 (格式: x_offset_y_offset_width_height)"""
# #     match = re.match(r"^(\d+)_(\d+)_(\d+)_(\d+)$", folder_name)
# #     if match:
# #         return list(map(int, match.groups()))  # [x, y, width, height]
# #     return None

# # # ================= 配置参数 =================
# # base_dir = "/scratch/zf281/downstream_dataset/austrian_whole_year/d_pixel_retiled"
# # big_tiff_path = "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/roi.tif"
# # output_path = "tiff_coverage_check.png"
# # # ============================================

# # # 创建画布
# # fig, ax = plt.subplots(figsize=(15, 15))

# # # 步骤1：绘制大TIFF（填充内容）
# # try:
# #     with Image.open(big_tiff_path) as img:
# #         big_data = np.array(img)
# #         ax.imshow(big_data, 
# #                  extent=[0, big_data.shape[1], 0, big_data.shape[0]],  # 假设坐标原点在左下角
# #                  cmap='viridis', 
# #                  alpha=0.4,  # 半透明填充
# #                  label='Reference TIFF')
# # except Exception as e:
# #     print(f"大TIFF加载失败: {str(e)}")

# # # 步骤2：绘制所有小tile的边界框
# # total_tiles = 0
# # for folder in os.listdir(base_dir):
# #     coords = parse_coordinates(folder)
# #     if not coords:
# #         continue
    
# #     tile_path = os.path.join(base_dir, folder, "roi.tiff")
# #     if not os.path.exists(tile_path):
# #         continue
    
# #     try:
# #         # 获取实际tile尺寸
# #         with Image.open(tile_path) as img:
# #             w, h = img.size
        
# #         # 绘制边界框
# #         rect = Rectangle(
# #             (coords[0], coords[1]),  # 左下角坐标
# #             w,                      # 宽度
# #             h,                      # 高度
# #             linewidth=1.5,
# #             edgecolor='red',
# #             facecolor='none',       # 关键设置：无填充
# #             alpha=0.8,
# #             label='Tile Boundary' if total_tiles == 0 else ""  # 避免重复图例
# #         )
# #         ax.add_patch(rect)
# #         total_tiles += 1
        
# #     except Exception as e:
# #         print(f"处理 {folder} 失败: {str(e)}")

# # # 步骤3：自动调整坐标轴
# # ax.autoscale_view()
# # ax.set_xlabel('X Coordinate (pixels)')
# # ax.set_ylabel('Y Coordinate (pixels)')
# # ax.set_title('Spatial Coverage Verification\n(Red: Tile Boundaries, Blue: Reference Area)', pad=20)

# # # 添加智能图例
# # handles, labels = ax.get_legend_handles_labels()
# # unique_labels = dict(zip(labels, handles))
# # ax.legend(unique_labels.values(), unique_labels.keys(), 
# #          loc='lower right', framealpha=0.7)

# # # 保存并显示
# # plt.savefig(output_path, dpi=300, bbox_inches='tight')
# # print(f"验证结果已保存至: {output_path}")
# # plt.close()

# # sample_index = 20
# # time_index = 0
# # time_sar_index = 1
# # r = band[sample_index, :, :, time_index, 0]
# # # 归一化
# # r = r.astype(np.float32)
# # r = (r - r.min()) / (r.max() - r.min())

# # import matplotlib.pyplot as plt
# # plt.imshow(r, cmap='gray')
# # plt.imsave("r.png", r)
# # plt.close()

# # sar_band_file_path = "data/ssl_training/ready_to_use_patch/aug1/s1/data_B1_F1.npy"
# # sar_band = np.load(sar_band_file_path, mmap_mode='r')
# # print(sar_band.shape)
# # sar_r = sar_band[sample_index, :, :, time_sar_index, 0]
# # # 归一化
# # sar_r = sar_r.astype(np.float32)
# # sar_r = (sar_r - sar_r.min()) / (sar_r.max() - sar_r.min())
# # plt.imshow(sar_r, cmap='gray')
# # plt.imsave("sar_r.png", sar_r)
# # plt.close()

# # filepath = "/scratch/zf281/robin/fungal/data_processed/34VFL/bands.npy"
# # doy_filepath = "/scratch/zf281/robin/fungal/data_processed/34VFL/doys.npy"
# # # filepath1 = f"/scratch/zf281/global/{tile_code}/sar_descending_doy.npy"
# # data = np.load(filepath,mmap_mode='r')
# # # data = np.load(filepath)
# # print(data.shape)
# # print(data.dtype)
# # print(data[0,5000:5010,5000:5010, 0])
# # print(data[1,5000:5010,5000:5010, 0])

# # doy = np.load(doy_filepath)
# # print(doy)
# # 统计不为0的元素的个数
# # print(np.count_nonzero(data))
# # print(data[0])

# # import numpy as np

# # filepath = "/maps/zf281/btfm4rs/data/ssl_training/ready_to_use_64/s1/data_B1_F1.npy"
# # # # doy_filepath = "/scratch/zf281/robin/fungal/data_processed/35VLF/doys.npy"
# # # # filepath1 = f"/scratch/zf281/global/{tile_code}/sar_descending_doy.npy"
# # data = np.load(filepath,mmap_mode='r')
# # # data = np.load(filepath)
# # print(data.shape)
# # print(data.dtype)
# # print(data[0,5000:5010,5000:5010, 0])
# # print(data[1,5000:5010,5000:5010, 0])

# # doy = np.load(doy_filepath)
# # print(doy)
# # 统计不为0的元素的个数
# # print(np.count_nonzero(data))
# # print(data[0])

# # 定义新的doy数组，注意数据类型与原文件一致为uint16
# # new_doys = np.array(
# #     [27, 40, 47, 50, 60, 67, 70, 80, 87, 107, 120, 130, 140, 147, 167, 170, 177, 180, 187, 197, 200, 217, 227, 250],
# #     dtype=np.uint16
# # )

# # # 保存为 new_doys.npy 文件
# # np.save("new_doys.npy", new_doys)

# # # 提取前三个通道
# # rbg_time_0 = data[10, :, :, :3]
# # # rbg变为rgb
# # rbg_time_0 = rbg_time_0[..., [2, 1, 0]]
# # # 转为float
# # rbg_time_0 = rbg_time_0.astype(np.float32)
# # # 归一化后保存为png
# # for i in range(3):
# #     rbg_time_0[:, :, i] = rbg_time_0[:, :, i] / np.max(rbg_time_0[:, :, i])
# # import matplotlib.pyplot as plt
# # plt.imshow(rbg_time_0)
# # plt.imsave("rbg_time_0.png", rbg_time_0)
# # plt.close()

# # print(data[20, 5000:5100, 5000:5100, ...])
# # 检查第二个通道是否全为0
# # print(np.all(data[:,5000:5500,5000:5500, 1] == 0))
# # print(data[0].shape)
# # print(data)
# # print(data1)


# # import rasterio
# # import matplotlib.pyplot as plt
# # import numpy as np

# # def visualize_tiff(tiff_path, output_filename='visualized_image.png'):
# #     with rasterio.open(tiff_path) as src:
# #         count = src.count
# #         # 如果有至少3个波段，则认为是RGB图像，否则取第一波段显示
# #         if count >= 3:
# #             # 读取前三个波段
# #             r = src.read(1)
# #             g = src.read(2)
# #             b = src.read(3)
# #             # 简单归一化处理
# #             def normalize(array):
# #                 array = array.astype(np.float32)
# #                 array -= array.min()
# #                 if array.max() > 0:
# #                     array /= array.max()
# #                 return array
# #             rgb = np.dstack((normalize(r), normalize(g), normalize(b)))
# #             plt.figure(figsize=(10, 10))
# #             plt.imshow(rgb)
# #             plt.title("RGB Composite")
# #         else:
# #             # 只读取第一个波段
# #             band = src.read(1)
# #             plt.figure(figsize=(10, 10))
# #             plt.imshow(band, cmap='gray')
# #             plt.title("Single Band")
# #             plt.colorbar()
    
# #     plt.axis('off')
# #     plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
# #     plt.show()
# #     print(f"图像已保存为：{output_filename}")

# # if __name__ == '__main__':
# #     # 修改为你的 tiff 文件路径
# #     tiff_path = '/scratch/zf281/robin/fungal/estonia_roi.tif'
# #     visualize_tiff(tiff_path)


# import numpy as np
# import os

# def dequantize_representation(representation_file_path, scales_file_path):
#     """
#     加载 int8 表示和其缩放因子，并将它们反量化回 float32 格式。

#     Args:
#         representation_file_path (str): int8 表示文件的路径 (H,W,C)。
#         scales_file_path (str): float32 缩放因子文件的路径 (H,W)。

#     Returns:
#         numpy.ndarray: float32 格式的 ndarray，形状为 (H,W,C)。
#     """
#     print(f"正在加载 int8 表示文件: {representation_file_path}")
#     # 加载文件
#     representation_int8 = np.load(representation_file_path)  # (H, W, C), dtype=int8
    
#     print(f"正在加载 scales 文件: {scales_file_path}")
#     scales = np.load(scales_file_path)  # (H, W), dtype=float32
    
#     print("文件加载完成。")
#     print(f"  - int8 表示形状: {representation_int8.shape}")
#     print(f"  - Scales 形状: {scales.shape}")
    
#     # 为了计算，将 int8 转换为 float32
#     print("正在将 int8 转换为 float32...")
#     representation_f32 = representation_int8.astype(np.float32)
    
#     # 扩展 scales 的维度以匹配表示的形状
#     # scales shape: (H, W) -> (H, W, 1)
#     print("正在扩展 scales 的维度用于广播...")
#     scales_expanded = scales[..., np.newaxis]
    
#     # 通过乘以 scales 来进行反量化
#     print("正在执行反量化操作 (乘法)...")
#     representation_f32 = representation_f32 * scales_expanded
    
#     print("反量化完成。")
#     print(f"  - 生成的 float32 表示形状: {representation_f32.shape}")
    
#     return representation_f32

# def main():
#     """
#     主执行函数
#     """
#     # --- 文件路径配置 ---
#     int8_file = '/maps/zf281/btfm4rs/data/representation/austrian_crop_v1.0_pipeline_downsample_100_int8.npy'
#     scales_file = '/maps/zf281/btfm4rs/data/representation/austrian_crop_v1.0_pipeline_downsample_100_int8_scales.npy'
#     output_file = '/maps/zf281/btfm4rs/data/representation/austrian_crop_v1.0_pipeline_downsample_100_f32_from_int8.npy'

#     # 检查输入文件是否存在
#     if not os.path.exists(int8_file):
#         print(f"错误: 输入文件不存在: {int8_file}")
#         return
#     if not os.path.exists(scales_file):
#         print(f"错误: 输入文件不存在: {scales_file}")
#         return

#     # 执行转换
#     dequantized_representation = dequantize_representation(int8_file, scales_file)
    
#     # 保存结果
#     try:
#         print(f"\n正在将转换后的 float32 数组保存到: {output_file}")
#         # 确保输出目录存在
#         output_dir = os.path.dirname(output_file)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#             print(f"已创建目录: {output_dir}")
            
#         np.save(output_file, dequantized_representation)
#         print("="*50)
#         print("成功！")
#         print(f"文件已成功保存。")
#         print("="*50)
#     except Exception as e:
#         print(f"保存文件时发生错误: {e}")

# if __name__ == '__main__':
#     main()


# import rasterio
# from rasterio.windows import from_bounds
# import numpy as np
# import logging

# # --- 配置日志 ---
# # 设置日志记录，以便我们可以看到详细的输出信息
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- 文件路径定义 ---
# # 定义输入的大 TIFF 文件路径
# large_tiff_path = '/maps/zf281/btfm4rs/data/downstream/pv_detection/london/london_map_10m_utm30n_128bands.tiff'
# # 定义与大 TIFF 对应的 Numpy npy 文件路径
# scales_npy_path = '/maps/zf281/btfm4rs/data/downstream/pv_detection/london/london_map_10m_utm30n_scales.npy'
# # 定义参考 TIFF 文件路径
# reference_tiff_path = '/maps/zf281/btfm4rs/data/downstream/pv_detection/london/london_rooftop_point_and_polygon.tif'

# # 定义裁剪后输出的文件路径
# output_tiff_path = '/maps/zf281/btfm4rs/data/downstream/pv_detection/london/london_map_10m_utm30n_128bands_cropped.tiff'
# output_npy_path = '/maps/zf281/btfm4rs/data/downstream/pv_detection/london/london_map_10m_utm30n_scales_cropped.npy'

# try:
#     # --- 步骤 1: 读取参考文件的地理范围 ---
#     logging.info(f"开始处理，参考文件为: {reference_tiff_path}")
#     with rasterio.open(reference_tiff_path) as ref_ds:
#         ref_bounds = ref_ds.bounds
#         ref_crs = ref_ds.crs
#         logging.info(f"参考文件的尺寸 (波段数, 高, 宽): ({ref_ds.count}, {ref_ds.height}, {ref_ds.width})")
#         logging.info(f"参考文件的地理范围 (Bounds): Left={ref_bounds.left}, Bottom={ref_bounds.bottom}, Right={ref_bounds.right}, Top={ref_bounds.top}")
#         logging.info(f"参考文件的坐标参考系 (CRS): {ref_crs}")

#     # 在处理主文件之前，先加载NPY文件并检查尺寸
#     logging.info(f"正在加载NPY文件: {scales_npy_path}")
#     scales_array = np.load(scales_npy_path)
#     logging.info(f"原始NPY文件的尺寸 (Shape): {scales_array.shape}")


#     # --- 步骤 2: 读取大文件的元数据并计算裁剪窗口 ---
#     logging.info(f"正在读取待裁剪的TIFF文件: {large_tiff_path}")
#     with rasterio.open(large_tiff_path) as src_ds:
#         # 打印原始尺寸信息
#         logging.info(f"待裁剪TIFF的原始尺寸 (波段数, 高, 宽): ({src_ds.count}, {src_ds.height}, {src_ds.width})")
        
#         # 验证NPY文件和TIFF文件的空间尺寸是否一致
#         if src_ds.height != scales_array.shape[0] or src_ds.width != scales_array.shape[1]:
#             logging.error("错误：TIFF文件和NPY文件的空间尺寸 (高度/宽度) 不匹配！")
#             logging.error(f"TIFF (H, W): ({src_ds.height}, {src_ds.width})")
#             logging.error(f"NPY (H, W): ({scales_array.shape[0]}, {scales_array.shape[1]})")
#             exit()
        
#         logging.info("TIFF和NPY文件的空间尺寸匹配，继续处理。")

#         # 检查两个TIFF文件的坐标参考系是否一致
#         if src_ds.crs != ref_crs:
#             logging.error("错误：两个TIFF文件的坐标参考系 (CRS) 不匹配。无法进行裁剪。")
#             exit()
        
#         logging.info("坐标参考系匹配，继续处理。")

#         # 根据参考文件的地理范围计算裁剪窗口
#         crop_window = from_bounds(
#             left=ref_bounds.left,
#             bottom=ref_bounds.bottom,
#             right=ref_bounds.right,
#             top=ref_bounds.top,
#             transform=src_ds.transform
#         )
        
#         logging.info(f"计算出的裁剪窗口 (Window): {crop_window}")

#         # --- 步骤 3: 裁剪TIFF文件并写入 ---
#         logging.info("正在从待裁剪TIFF文件中读取数据...")
#         cropped_tiff_data = src_ds.read(window=crop_window)
        
#         logging.info(f"正在将裁剪后的TIFF数据写入到: {output_tiff_path}")
#         out_transform = src_ds.window_transform(crop_window)
#         out_meta = src_ds.meta.copy()
        
#         out_meta.update({
#             "driver": "GTiff",
#             "height": cropped_tiff_data.shape[1],
#             "width": cropped_tiff_data.shape[2],
#             "transform": out_transform
#         })

#         with rasterio.open(output_tiff_path, "w", **out_meta) as dest_ds:
#             dest_ds.write(cropped_tiff_data)
            
#     # --- 步骤 4: 裁剪NPY文件并写入 ---
#     logging.info("正在裁剪NPY数组...")
#     # 从 window 对象中获取行和列的偏移量及高宽
#     row_off = int(crop_window.row_off)
#     col_off = int(crop_window.col_off)
#     height = int(crop_window.height)
#     width = int(crop_window.width)

#     # 使用Numpy切片语法裁剪数组
#     # array[start_row:end_row, start_col:end_col]
#     cropped_scales_array = scales_array[row_off : row_off + height, col_off : col_off + width]
    
#     logging.info(f"正在将裁剪后的NPY数组写入到: {output_npy_path}")
#     np.save(output_npy_path, cropped_scales_array)
            
#     # --- 步骤 5: 打印输出文件的信息进行最终验证 ---
#     with rasterio.open(output_tiff_path) as out_ds:
#         logging.info("--- 操作完成 ---")
#         logging.info(f"已成功创建裁剪后的TIFF文件: {output_tiff_path}")
#         logging.info(f"输出TIFF文件的尺寸 (波段数, 高, 宽): ({out_ds.count}, {out_ds.height}, {out_ds.width})")
#         logging.info(f"输出TIFF文件的地理范围 (Bounds): Left={out_ds.bounds.left}, Bottom={out_ds.bounds.bottom}, Right={out_ds.bounds.right}, Top={out_ds.bounds.top}")

#     final_npy = np.load(output_npy_path)
#     logging.info(f"已成功创建裁剪后的NPY文件: {output_npy_path}")
#     logging.info(f"输出NPY文件的尺寸 (Shape): {final_npy.shape}")

#     # 最终一致性检查
#     if out_ds.height == final_npy.shape[0] and out_ds.width == final_npy.shape[1]:
#         logging.info("最终验证成功：裁剪后的TIFF和NPY文件空间尺寸一致。")
#     else:
#         logging.warning("警告：裁剪后的TIFF和NPY文件空间尺寸不一致！请检查日志。")


# except FileNotFoundError as e:
#     logging.error(f"文件未找到错误: {e}")
#     logging.error("请确保输入的文件路径正确无误。")
# except Exception as e:
#     logging.error(f"处理过程中发生未知错误: {e}")



# import numpy as np
# import os

# file_path = "/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile/2024_change_detection_test_tile_labels.npy"
# data = np.load(file_path, mmap_mode='r')
# print(data.shape)
# # print(data[0:10,0:10,:])

# unique, counts = np.unique(data, return_counts=True)
# value_counts = dict(zip(unique, counts))
# print("值的分布情况:")
# for val, cnt in value_counts.items():
#     print(f"值 {val}: 出现 {cnt} 次")

# import rasterio
# import numpy as np
# import cv2  # 导入OpenCV库
# from pathlib import Path

# # --- 1. 设置文件路径和目标尺寸 ---
# input_filepath = Path('/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile/2024_change_detection_test_tile_labels.tif')

# # 设置您期望的最终输出尺寸 (高度, 宽度)
# target_shape = (1137, 734)

# # --- 2. 动态生成输出文件路径 ---
# output_filepath = input_filepath.with_suffix('.npy')

# print(f"输入文件: {input_filepath}")
# print(f"输出文件: {output_filepath}")
# print(f"目标尺寸 (H, W): {target_shape}")

# try:
#     # --- 3. 读取原始栅格数据 ---
#     with rasterio.open(input_filepath) as src:
#         original_data = src.read(1)

#     print(f"\n成功读取图像，原始尺寸: {original_data.shape}")

#     # --- 4. 转换值 (标签为1，其余为0) ---
#     print("正在转换值...")
#     binary_mask = (original_data == 1).astype(np.uint8)

#     # --- 5. 【新增】强制重置尺寸 ---
#     # 检查当前尺寸是否与目标尺寸不同
#     if binary_mask.shape != target_shape:
#         print(f"当前尺寸 {binary_mask.shape} 与目标尺寸 {target_shape} 不同，开始重置尺寸...")

#         # 注意：cv2.resize函数需要 (宽度, 高度) 格式的尺寸
#         target_size_for_cv2 = (target_shape[1], target_shape[0])

#         # 使用cv2.resize进行缩放
#         # interpolation=cv2.INTER_NEAREST (最近邻插值) 对于标签/掩码数据至关重要，
#         # 它可以保证缩放后的值仍然是0或1。
#         resized_data = cv2.resize(
#             binary_mask,
#             dsize=target_size_for_cv2,
#             interpolation=cv2.INTER_NEAREST
#         )
#         print(f"尺寸重置成功，新尺寸: {resized_data.shape}")
#         final_data_to_save = resized_data
#     else:
#         print("当前尺寸与目标尺寸相同，无需重置。")
#         final_data_to_save = binary_mask

#     # --- 6. 保存为.npy文件 ---
#     np.save(output_filepath, final_data_to_save)
#     print(f"\n操作完成！已将结果保存到: {output_filepath}")

#     # --- 7. (可选) 验证 ---
#     print("\n正在验证输出文件...")
#     loaded_array = np.load(output_filepath)
#     unique_values = np.unique(loaded_array)
    
#     print(f"验证: 文件中唯一值为: {unique_values}")
#     print(f"验证: 文件尺寸为: {loaded_array.shape}")

#     if loaded_array.shape == target_shape:
#         print("尺寸验证成功！")
#     else:
#         print(f"警告：输出尺寸 {loaded_array.shape} 与目标尺寸 {target_shape} 不符！")

# except FileNotFoundError:
#     print(f"错误：文件未找到，请检查路径是否正确: {input_filepath}")
# except Exception as e:
#     print(f"处理文件时发生错误: {e}")