import os
import logging
from glob import glob

import numpy as np
import rasterio
import matplotlib.pyplot as plt

# 设置日志输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据和输出目录
base_dir = '/scratch/zf281/robin/2021-2022/data_raw/32UQB'
output_dir = '/maps/zf281/btfm4rs/data/rgb/2021_32UQB'
mask_file = '/scratch/zf281/robin/2021-2022/32UQB/masks.npy'
crop_size = (549, 549)  # (height, width)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Created output directory: {output_dir}")

# 定义中心裁剪函数
def crop_center(img, crop_size):
    h, w = img.shape[:2]
    ch, cw = crop_size
    start_y = (h - ch) // 2
    start_x = (w - cw) // 2
    return img[start_y:start_y+ch, start_x:start_x+cw]

# 加载 masks
logging.info("Loading masks...")
masks = np.load(mask_file)  # shape: (T, H, W)
logging.info(f"Masks shape: {masks.shape}")

# 获取 red 文件夹下所有 tiff 文件，并排序以确保时间一致
red_files = glob(os.path.join(base_dir, 'red', '*.tiff'))
red_files.sort()
logging.info(f"Found {len(red_files)} red files.")

# 检查 red 文件数与 masks 的时间步匹配
if len(red_files) != masks.shape[0]:
    logging.warning("Number of red files does not match masks time dimension!")
    T = min(len(red_files), masks.shape[0])
    logging.info(f"Processing first {T} files based on minimum count.")
else:
    T = len(red_files)

def normalize(array):
    """
    简单归一化，将数据按照2%和98%的分位值拉伸到[0,1]
    """
    a_min, a_max = np.percentile(array, (2, 98))
    array = np.clip(array, a_min, a_max)
    return (array - a_min) / (a_max - a_min)

for i in range(T):
    red_path = red_files[i]
    filename = os.path.basename(red_path)
    
    # 构造对应 blue 和 green 文件的路径
    blue_path = os.path.join(base_dir, 'blue', filename)
    green_path = os.path.join(base_dir, 'green', filename)
    
    if not (os.path.exists(blue_path) and os.path.exists(green_path)):
        logging.info(f"Skipping {filename} because corresponding blue or green file is missing.")
        continue

    # 裁剪 masks 中的正中心区域
    current_mask = masks[i]
    mask_crop = crop_center(current_mask, crop_size)
    clear_ratio = np.mean(mask_crop)
    logging.info(f"File {filename}: clear ratio in cropped area = {clear_ratio:.2f}")
    if clear_ratio < 0.8:
        logging.info(f"Skipping {filename} due to low clear pixel ratio in cropped area ({clear_ratio:.2f}).")
        continue

    # 从文件名中提取日期（假设格式为：S2A_32UQB_YYYYMMDD_...）
    parts = filename.split('_')
    date = parts[2]

    # 读取三个波段数据，并裁剪中心区域
    try:
        with rasterio.open(red_path) as src_red:
            red_data = crop_center(src_red.read(1), crop_size)
        with rasterio.open(green_path) as src_green:
            green_data = crop_center(src_green.read(1), crop_size)
        with rasterio.open(blue_path) as src_blue:
            blue_data = crop_center(src_blue.read(1), crop_size)
    except Exception as e:
        logging.error(f"Error reading file {filename}: {e}")
        continue

    # 将三个波段合并成 RGB 图像
    rgb = np.stack([red_data, green_data, blue_data], axis=2)
    
    # 对每个通道分别归一化
    rgb_norm = np.empty_like(rgb, dtype=np.float32)
    for j in range(3):
        rgb_norm[..., j] = normalize(rgb[..., j])
    
    # 保存 PNG 文件
    output_path = os.path.join(output_dir, f'{date}.png')
    plt.imsave(output_path, rgb_norm)
    logging.info(f"Saved RGB image for date {date} to {output_path}")

logging.info("Processing complete.")
