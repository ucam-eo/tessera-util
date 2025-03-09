import numpy as np

filepath = "/scratch/zf281/robin/fungal/representation_from_dawn/34VFM/representation.npy"
# filepath1 = f"/scratch/zf281/global/{tile_code}/sar_descending_doy.npy"
data = np.load(filepath, mmap_mode='r') #(n, h, w, c)
# data1 = np.load(filepath1)
print(data.shape)
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
