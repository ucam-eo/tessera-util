import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import glob

# 定义目录路径
data_dir = "/mnt/e/Codes/btfm4rs/data/downstream/cambridge"
tiff_path = os.path.join(data_dir, "cambridge.tiff")
output_gif = os.path.join(data_dir, "cambridge_timelapse.gif")

# 优化输出分辨率设置
dpi = 600
figsize = (8, 6)  # 基础尺寸，会根据DPI进行调整

# 字体大小调整参数
title_fontsize_base = 24
title_fontsize_scale = 3  # 放大系数，用户要求的约3倍

# 加载ROI掩膜
print("加载ROI掩膜...")
roi_mask = tifffile.imread(tiff_path).astype(bool)

# 查找所有npy文件
npy_files = sorted(glob.glob(os.path.join(data_dir, "*_fsdp_*.npy")))

# 加载所有npy文件并提取前3个通道，使用memmap模式
data_list = []
years = []

for npy_file in npy_files:
    # 从文件名中提取年份
    year = os.path.basename(npy_file).split('_')[0]
    years.append(year)
    
    # 使用memmap模式加载npy文件
    print(f"加载文件: {os.path.basename(npy_file)}...")
    data = np.load(npy_file, mmap_mode='r')
    # 只读取前三个通道而不是整个数据
    data_first_3_channels = data[:, :, :3].copy()  # 复制以避免内存共享问题
    data_list.append(data_first_3_channels)
    
    # 关闭memmap连接以释放资源
    del data

print(f"找到 {len(years)} 个npy文件，年份: {', '.join(years)}")

# 为GIF创建帧
frames = []

for i, (data, year) in enumerate(zip(data_list, years)):
    print(f"处理年份 {year}...")
    
    # 创建归一化的RGB图像
    normalized_data = np.zeros_like(data, dtype=np.float32)
    
    # 为当前时间步计算每个通道的最小值和最大值
    time_step_min = np.zeros(3)
    time_step_max = np.zeros(3)
    
    for c in range(3):
        channel_data = data[:, :, c]
        roi_values = channel_data[roi_mask]
        
        # 计算当前时间步ROI内的最小值和最大值
        time_step_min[c] = np.min(roi_values)
        time_step_max[c] = np.max(roi_values)
        
        print(f"  通道 {c} - 最小值: {time_step_min[c]:.4f}, 最大值: {time_step_max[c]:.4f}")
        
        # 单独对该时间步归一化
        norm_channel = np.zeros_like(channel_data, dtype=np.float32)
        
        channel_range = time_step_max[c] - time_step_min[c]
        if channel_range > 0:  # 避免除以零
            norm_channel[roi_mask] = (channel_data[roi_mask] - time_step_min[c]) / channel_range
        
        normalized_data[:, :, c] = norm_channel
    
    # 创建RGBA图像，ROI外为透明
    rgba = np.zeros((*data.shape[:2], 4))
    rgba[:, :, :3] = normalized_data  # RGB通道
    rgba[:, :, 3] = roi_mask.astype(float)  # Alpha通道（ROI内为1，外为0）
    
    # 创建带标题的图形，高DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(rgba)
    
    # 根据DPI和用户要求的放大系数调整标题字体大小
    # 确保字体大小不会过小
    title_fontsize = max(15, int(title_fontsize_base * title_fontsize_scale * (100/dpi)))
    print(f"  标题字体大小: {title_fontsize}")
    
    # ax.set_title(f"Year: {year}", fontsize=title_fontsize)
    ax.axis('off')
    
    # 保存为临时PNG（带透明度）
    temp_file = os.path.join(data_dir, f"temp_{year}.png")
    plt.savefig(temp_file, transparent=True, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    
    # 加载保存的图像用于GIF
    frames.append(Image.open(temp_file))

# 调整所有帧到相同大小（以第一帧为标准）
first_frame_size = frames[0].size
print(f"输出图像大小: {first_frame_size[0]}x{first_frame_size[1]}像素")
resized_frames = []
for frame in frames:
    if frame.size != first_frame_size:
        frame = frame.resize(first_frame_size, Image.LANCZOS)
    resized_frames.append(frame)

# 保存为GIF
print(f"创建带有 {len(resized_frames)} 帧的GIF...")
resized_frames[0].save(
    output_gif,
    save_all=True,
    append_images=resized_frames[1:],
    duration=1000,  # 每帧1秒
    loop=0,  # 无限循环
    disposal=2,  # 每帧替换前一帧
    optimize=False  # 禁用优化以保持高质量
)

# 清理临时文件
# for year in years:
#     temp_file = os.path.join(data_dir, f"temp_{year}.png")
#     if os.path.exists(temp_file):
#         os.remove(temp_file)

print(f"GIF创建成功: {output_gif}")