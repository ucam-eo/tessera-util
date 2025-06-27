import os
import imageio
from glob import glob
import logging

# 设置日志输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义 PNG 文件所在目录和输出 GIF 文件路径
png_dir = '/maps/zf281/btfm4rs/data/rgb/2021_32UQB'
output_gif = os.path.join(png_dir, 'animation.gif')

# 获取目录下所有 png 文件，并按文件名排序（时间顺序）
png_files = glob(os.path.join(png_dir, '*.png'))
png_files.sort()  # 假设文件名格式为 YYYYMMDD.png，自然排序就是时间顺序

if not png_files:
    logging.error("未在目录中找到 png 文件。")
    exit(1)
else:
    logging.info(f"共找到 {len(png_files)} 个 png 文件，开始生成 GIF 动画...")

frames = []
for file in png_files:
    logging.info(f"读取图像: {file}")
    try:
        img = imageio.imread(file)
        frames.append(img)
    except Exception as e:
        logging.error(f"读取 {file} 时出错: {e}")

# 生成 GIF 动画，间隔时间为 0.5 秒
try:
    imageio.mimsave(output_gif, frames, duration=0.5)
    logging.info(f"GIF 动画已保存到: {output_gif}")
except Exception as e:
    logging.error(f"保存 GIF 时出错: {e}")
