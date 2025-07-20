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

# import rasterio
# import numpy as np

# # 1. 读取TIFF文件
# with rasterio.open('/mnt/e/Codes/btfm4rs/data/downstream/cambridge/cambridge.tiff') as src:
#     data = src.read(1)  # 读取第一个波段（单波段TIFF）

# # 打印形状
# print(data.shape)  # 输出数组的形状

# # 2. 统计唯一值及其出现次数
# unique_values, counts = np.unique(data, return_counts=True)

# # 3. 计算每个值的占比（百分比）
# total_pixels = data.size
# percentages = (counts / total_pixels) * 100

# # 4. 打印结果
# print("Value distribution:")
# for val, cnt, pct in zip(unique_values, counts, percentages):
#     print(f"Value {val}: {cnt} pixels ({pct:.2f}%)")

# 可选：可视化（需要matplotlib）
# import matplotlib.pyplot as plt
# plt.bar(unique_values, percentages)
# plt.xlabel('Pixel Value')
# plt.ylabel('Percentage (%)')
# plt.title('TIFF Pixel Value Distribution')
# plt.show()

# import os
# import rasterio
# from rasterio.windows import Window

# def tile_raster(input_path: str, output_dir_name: str = 'smaller_tif', tile_size: int = 500):
#     """
#     将一个大型TIF文件分割成指定大小的图块。

#     Args:
#         input_path (str): 输入的TIF文件完整路径。
#         output_dir_name (str): 用于存放图块的文件夹名称。
#         tile_size (int): 每个图块的边长（像素）。
#     """
#     try:
#         # 检查输入文件是否存在
#         if not os.path.exists(input_path):
#             print(f"错误：找不到文件 '{input_path}'。请检查路径是否正确。")
#             return

#         # 创建输出目录
#         base_dir = os.path.dirname(input_path)
#         output_dir = os.path.join(base_dir, output_dir_name)
#         os.makedirs(output_dir, exist_ok=True)
#         print(f"输出目录已创建：{output_dir}")

#         # 打开TIF文件
#         with rasterio.open(input_path) as src:
#             width = src.width
#             height = src.height
#             profile = src.profile.copy() # 复制元数据

#             print(f"输入影像尺寸: Width={width}, Height={height}")
#             print(f"开始以 {tile_size}x{tile_size} 像素大小进行分割...")

#             # 遍历并创建图块
#             for j in range(0, height, tile_size):
#                 for i in range(0, width, tile_size):
#                     # 计算当前窗口的实际宽高，以处理边缘情况
#                     window_width = min(tile_size, width - i)
#                     window_height = min(tile_size, height - j)
                    
#                     # 定义读取窗口
#                     window = Window(col_off=i, row_off=j, width=window_width, height=window_height)

#                     # 计算新图块的地理空间变换信息
#                     transform = src.window_transform(window)

#                     # 定义输出文件名
#                     y1, x1 = j, i
#                     y2, x2 = j + window_height, i + window_width
#                     output_filename = f"{y1}_{x1}_{y2}_{x2}.tif"
#                     output_filepath = os.path.join(output_dir, output_filename)

#                     # 更新新图块的元数据
#                     profile.update({
#                         'height': window_height,
#                         'width': window_width,
#                         'transform': transform
#                     })

#                     # 读取窗口数据并写入新的TIF文件
#                     with rasterio.open(output_filepath, 'w', **profile) as dst:
#                         dst.write(src.read(window=window))

#             print(f"\n处理完成！所有图块已保存至 '{output_dir}'")

#     except Exception as e:
#         print(f"处理过程中发生错误: {e}")

# # --- 主程序执行 ---
# if __name__ == '__main__':
#     # 请在这里定义你的TIF文件路径
#     tif_file_path = '/mnt/e/Codes/btfm4rs/data/downstream/austrian_crop/austrian_roi.tif'
    
#     # 执行分割函数
#     tile_raster(tif_file_path)

import numpy as np

file_path = "/mnt/e/Codes/btfm4rs/data/representation/Austria_EFM_Embeddings_2021_method1_100m.npy"
data = np.load(file_path, mmap_mode='r')
print(data.shape)  # 输出数组的形状
print(data)

# rgb = data[:, :, :3,0].copy()  # 获取前3个波段
rgb = data[:, :, :3].copy()  # 获取前3个波段
rgb = rgb.astype(np.float32)  # 转为float
# #归一化
for i in range(3):
    rgb[:, :, i] = (rgb[:, :, i] - np.min(rgb[:, :, i])) / (np.max(rgb[:, :, i]) - np.min(rgb[:, :, i]))
# 可视化
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.imshow(rgb)
plt.savefig('repr_efm.png')
plt.close()

# file_path1 = "data/representation/austrian_crop_downsample_100_fsdp_20250407_195912.npy"
# data1 = np.load(file_path1) # (527, 602, 64)
# rgb1 = data1[:, :, :3]  # 获取前3个波段
# # 打印形状
# print(data1.shape)  # 输出数组的形状
# #归一化
# for i in range(3):
#     rgb1[:, :, i] = (rgb1[:, :, i] - np.min(rgb1[:, :, i])) / (np.max(rgb1[:, :, i]) - np.min(rgb1[:, :, i]))
# # 可视化
# plt.imshow(rgb1)
# plt.savefig('repr_btfm.png')
# plt.close()

# 将边缘的1个像素裁剪掉，变为(525, 600, 64)
# data = data[1:-1, 1:-1, :]
# print(data.shape)
# # 查看数据中是否有nan
# print(np.isnan(data).any()) # True
# 将nan替换为0
# data[np.isnan(data)] = 0
# # 查看数据中是否有nan
# print(np.isnan(data).any()) # False
# 将数据中
# 保存
# np.save(file_path.replace(".npy", "_cropped.npy"), data)
# print(data[20, 100:110, 100:110, 1])  # 0.0

# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
# import matplotlib.pyplot as plt

# file_path = "/mnt/e/Codes/btfm4rs/data/representation/borneo_representations_fsdp_20250407_195912.npy"
# data = np.load(file_path, mmap_mode='r')
# print(data.shape)  # 输出数组的形状
# # 用memmap
# # data = np.memmap(file_path, dtype='int16', mode='r', shape=(142, 10980, 10980, 10))
# sar_time_step = 3

# print(data.shape)  # 输出数组的形状
# print(data.dtype)  # 输出数组的数据类型
# print(data[sar_time_step,100:120,100:120,:])

# single_image = data[sar_time_step, :, :, 0]  # 获取第一个时间步的第一个波段
# plt.imshow(single_image, cmap='gray')
# plt.savefig('sar_0.png')
# plt.close()

# single_image = data[sar_time_step, :, :, 1]  # 获取第一个时间步的第一个波段
# plt.imshow(single_image, cmap='gray')
# plt.savefig('sar_1.png')
# plt.close()

# band_file_path = "/mnt/e/Codes/btfm/maddy_code/data_processed/MGRS_33UXP/bands.npy"
# band_file_path = "/mnt/e/Codes/btfm4rs/data/ready_to_use_40_steps/aug1/s1/data_B1_F1.npy"

# band_data = np.load(band_file_path, mmap_mode='r')
# print(band_data.shape)  # 输出数组的形状
# print(band_data[0,:,:,0,-1])

# patch_rbg = band_data[0, :, :, 0, :1]  # 获取第一个时间步的RGB波段


# RGB_MEAN = np.array([1711.0938, 1308.8511, 1546.4543])
# RGB_STD = np.array([1926.1026, 1862.9751, 1803.1792])
# # 反归一化
# patch_rbg = patch_rbg * RGB_STD + RGB_MEAN
# # 转为float
# patch_rbg = patch_rbg.astype(np.float32)
# # 转为rgb
# single_rbg_image = patch_rbg[:, :, [2, 1, 0]]
# # 归一化
# for i in range(3):
#     single_rbg_image[:, :, i] = (single_rbg_image[:, :, i] - np.min(single_rbg_image[:, :, i])) / (np.max(single_rbg_image[:, :, i]) - np.min(single_rbg_image[:, :, i]))

# # 转为int
# single_rbg_image = (single_rbg_image * 255).astype(np.uint8)

# import cv2

# def histogram_equalization(image):
#     for i in range(3):
#         image[:,:,i] = cv2.equalizeHist((image[:,:,i] * 255).astype(np.uint8)) / 255.0
#     return image

# single_rbg_image = histogram_equalization(single_rbg_image)    

# 可视化
# plt.imshow(patch_rbg)
# plt.savefig('rgb.png')
# plt.close()

# rgb_time_step = 6

# single_rbg_image = band_data[rgb_time_step, :, :, :3]  # 获取第一个时间步的RGB波段
# # 转为float
# single_rbg_image = single_rbg_image.astype(np.float32)
# # 转为rgb
# single_rbg_image = single_rbg_image[:, :, [2, 1, 0]]
# # 归一化
# for i in range(3):
#     single_rbg_image[:, :, i] = (single_rbg_image[:, :, i] - np.min(single_rbg_image[:, :, i])) / (np.max(single_rbg_image[:, :, i]) - np.min(single_rbg_image[:, :, i]))
# plt.imshow(single_rbg_image)
# plt.savefig('rgb.png')
# plt.close()

# import rasterio

# # 打开 TIFF 文件
# file_path = "/mnt/e/Codes/btfm4rs/data/downstream/borneo/Danum_2020_chm_lspikefree.tif"
# # file_path = "data/Jovana/SEKI ROI/seki_convex_hull.tiff"
# with rasterio.open(file_path) as dataset:
#     # 获取坐标参考系
#     crs = dataset.crs
#     print(f"坐标参考系（CRS）：\n{crs}\n")

#     # 获取仿射变换参数
#     transform = dataset.transform
#     print(f"仿射变换（Affine）：\n{transform}\n")

#     # 计算分辨率
#     resolution = (transform.a, transform.e)
#     print(f"分辨率：\n{resolution}\n")

#     # 获取地理范围
#     bounds = dataset.bounds
#     print(f"地理范围：\n{bounds}\n")

#!/usr/bin/env python3
# import os
# import rasterio
# from rasterio.warp import calculate_default_transform, reproject, Resampling
# from shapely.geometry import box, mapping
# import rasterio.mask

# def reproject_tiff(input_tiff: str, output_tiff: str, dst_crs: str = 'EPSG:32650') -> str:
#     """
#     将输入 TIFF 重投影到目标 CRS（默认 EPSG:32650），如果已在目标 CRS，则直接拷贝。
#     """
#     with rasterio.open(input_tiff) as src:
#         # 如果当前 CRS已经是目标 CRS，则直接复制文件
#         if src.crs.to_string() == dst_crs:
#             print(f"{input_tiff} 已经在 {dst_crs} 坐标系统，无需重投影。")
#             return input_tiff
        
#         transform, width, height = calculate_default_transform(
#             src.crs, dst_crs, src.width, src.height, *src.bounds
#         )
#         kwargs = src.meta.copy()
#         kwargs.update({
#             'crs': dst_crs,
#             'transform': transform,
#             'width': width,
#             'height': height
#         })

#         with rasterio.open(output_tiff, 'w', **kwargs) as dst:
#             for i in range(1, src.count + 1):
#                 reproject(
#                     source=rasterio.band(src, i),
#                     destination=rasterio.band(dst, i),
#                     src_transform=src.transform,
#                     src_crs=src.crs,
#                     dst_transform=transform,
#                     dst_crs=dst_crs,
#                     resampling=Resampling.nearest
#                 )
#     print(f"重投影后的 TIFF 已保存至: {output_tiff}")
#     return output_tiff

# def main():
#     # 定义原始文件路径
#     converted_tiff = "data/Jovana/SEKI ROI/seki_convex_hull.tiff"
#     reference_tiff = "data/Jovana/S2A_11SLA_20220108_0_L2A.tiff"
    
#     # 定义重投影后保存的文件路径
#     # reprojected_converted = "data/Jovana/SEKI ROI/seki_convex_hull_32650.tiff"
#     reprojected_converted = "data/Jovana/SEKI ROI/seki_convex_hull.tiff"
#     reprojected_reference = "data/Jovana/S2A_11SLA_20220108_0_L2A_32650.tiff"
    
#     dst_crs = 'EPSG:32650'
    
#     # 对两个 TIFF 进行重投影（如果需要）
#     reproject_tiff(converted_tiff, reprojected_converted, dst_crs)
#     reproject_tiff(reference_tiff, reprojected_reference, dst_crs)
    
#     # 分别读取重投影后的 TIFF 信息
#     with rasterio.open(reprojected_converted) as src1:
#         bounds1 = src1.bounds
#         crs1 = src1.crs
#         print("\n转换后 TIFF (SEKI ROI) 信息:")
#         print(f"  CRS: {crs1}")
#         print(f"  Bounds: {bounds1}")
    
#     with rasterio.open(reprojected_reference) as src2:
#         bounds2 = src2.bounds
#         crs2 = src2.crs
#         print("\n参考 TIFF 信息:")
#         print(f"  CRS: {crs2}")
#         print(f"  Bounds: {bounds2}")
    
#     # 构造两个 TIFF 的空间多边形（box 参数顺序：minx, miny, maxx, maxy）
#     poly1 = box(bounds1.left, bounds1.bottom, bounds1.right, bounds1.top)
#     poly2 = box(bounds2.left, bounds2.bottom, bounds2.right, bounds2.top)
    
#     # 计算交集区域
#     intersection = poly1.intersection(poly2)
#     if intersection.is_empty:
#         print("\n两个影像没有空间交集。")
#     else:
#         print("\n两个影像的交集区域信息：")
#         print(f"  交集边界: {intersection.bounds}")
#         print(f"  交集面积: {intersection.area}")
        
#         # 可选：利用交集区域对参考 TIFF 进行裁剪
#         with rasterio.open(reprojected_reference) as src:
#             out_image, out_transform = rasterio.mask.mask(src, [mapping(intersection)], crop=True)
#             out_meta = src.meta.copy()
#             out_meta.update({
#                 "height": out_image.shape[1],
#                 "width": out_image.shape[2],
#                 "transform": out_transform
#             })
#         subset_tiff = "data/Jovana/intersection_subset.tiff"
#         with rasterio.open(subset_tiff, "w", **out_meta) as dest:
#             dest.write(out_image)
#         print(f"\n裁剪后的交集 TIFF 已保存：{subset_tiff}")

# if __name__ == '__main__':
#     main()
