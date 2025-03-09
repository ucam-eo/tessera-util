import numpy as np
import matplotlib.pyplot as plt

# 文件路径
sar_file = '/scratch/zf281/robin/fungal/data_processed/34VFL/sar_ascending.npy'
bands_file = '/scratch/zf281/robin/fungal/data_processed/34VFL/bands.npy'

# 选择需要可视化的时间节点（这里示例取第0个时间步）
t1_index = 80  # 对应 sar_ascending.npy 的时间节点
t2_index = 100  # 对应 bands.npy 的时间节点

# 定义 patch 区域
row_start, row_end = 5000, 5500
col_start, col_end = 5000, 5500

# ----------------------------
# 处理 sar_ascending.npy
# ----------------------------
# 使用 mmap 模式读取大文件
sar_data = np.load(sar_file, mmap_mode='r')
# 提取指定时间和区域，并只取第一个波段（VV），结果为灰度图像
sar_patch = sar_data[t1_index, row_start:row_end, col_start:col_end, 0]

# 直方图均衡化
sar_patch = (sar_patch - sar_patch.min()) / (sar_patch.max() - sar_patch.min()) * 255
sar_patch = sar_patch.astype(np.uint8)

# 可视化并保存灰度图
plt.figure()
plt.imshow(sar_patch, cmap='gray')
plt.title(f'SAR Patch at time index {t1_index} (VV channel)')
plt.colorbar()
plt.savefig('sar_patch.png', dpi=300)
plt.close()

# ----------------------------
# 处理 bands.npy
# ----------------------------
# 使用 mmap 模式读取大文件
bands_data = np.load(bands_file, mmap_mode='r')
# 提取指定时间和区域，并取前三个波段 (R, G, B)
bands_patch = bands_data[t2_index, row_start:row_end, col_start:col_end, 0:3]

# 对每个通道进行逐一归一化
rgb_patch = np.empty_like(bands_patch, dtype=np.float32)
for i in range(3):
    channel = bands_patch[..., i].astype(np.float32)
    c_min = channel.min()
    c_max = channel.max()
    # 防止除零
    if c_max > c_min:
        normalized = (channel - c_min) / (c_max - c_min)
    else:
        normalized = channel - c_min
    rgb_patch[..., i] = normalized

# 可视化并保存 RGB 图像
plt.figure()
plt.imshow(rgb_patch)
plt.title(f'Bands Patch at time index {t2_index} (R,G,B channels)')
plt.axis('off')
plt.savefig('bands_patch.png', dpi=300)
plt.close()