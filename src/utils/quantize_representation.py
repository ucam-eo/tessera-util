import os
import numpy as np

# 原始文件路径
original_file = "/mnt/e/Codes/btfm4rs/data/representation/borneo_representations_val_acc_7852_4090pc.npy"

# 分离出文件名的前缀和后缀(.npy)
prefix, ext = os.path.splitext(original_file)

# 读入数据 (float32)
arr = np.load(original_file)  # shape: (H, W, C), dtype=float32

# 1) 转换为 float16 并保存
arr_f16 = arr.astype(np.float16)
float16_file = prefix + "_f16" + ext   # 拼接文件名
np.save(float16_file, arr_f16)
print(f"Float16 文件已保存至: {float16_file}")

# 2) 转换为 int8 并保存
#    这里以最大绝对值做统一缩放，保证数据能够尽量利用 [-128, 127] 范围
max_abs = np.max(np.abs(arr))
scale = 127.0 / max_abs    # 计算缩放因子
arr_int8 = np.round(arr * scale).clip(-128, 127).astype(np.int8)

int8_file = prefix + "_int8" + ext   # 拼接文件名
int8_scale_file = prefix + "_int8_scale" + ext  # 保存缩放因子
np.save(int8_file, arr_int8)
np.save(int8_scale_file, scale)  # 保存缩放因子
print(f"Int8 文件已保存至: {int8_file}")
print(f"缩放因子已保存至: {int8_scale_file}")
