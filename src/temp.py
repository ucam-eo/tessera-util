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

# file_path = "/mnt/e/Codes/btfm4rs/data/ssl_training/ready_to_use/aug1/s1/data_B1_F1.npy"
# data = np.load(file_path, mmap_mode='r')
# print(data.shape)
# # print(data[20, 100:110, 100:110, 1])  # 0.0

import rasterio

# 打开 TIFF 文件
file_path = "/mnt/e/Codes/btfm4rs/data/downstream/borneo/Danum_2020_chm_lspikefree.tif"
# file_path = "data/Jovana/SEKI ROI/seki_convex_hull.tiff"
with rasterio.open(file_path) as dataset:
    # 获取坐标参考系
    crs = dataset.crs
    print(f"坐标参考系（CRS）：\n{crs}\n")

    # 获取仿射变换参数
    transform = dataset.transform
    print(f"仿射变换（Affine）：\n{transform}\n")

    # 计算分辨率
    resolution = (transform.a, transform.e)
    print(f"分辨率：\n{resolution}\n")

    # 获取地理范围
    bounds = dataset.bounds
    print(f"地理范围：\n{bounds}\n")

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
