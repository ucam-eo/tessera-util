import os
import numpy as np
import rasterio
from rasterio.transform import Affine
 
def convert_npy_to_tiff(npy_path, ref_tiff_path, out_dir, downsample_rate=1):
    # 加载 npy 数据，假设数据形状为 (H, W, C)
    data = np.load(npy_path)
    H, W, C = data.shape
    
    # 如果需要下采样
    if downsample_rate > 1:
        # 计算下采样后的尺寸
        new_H = H // downsample_rate
        new_W = W // downsample_rate
        
        # 创建下采样后的数组
        downsampled_data = np.zeros((new_H, new_W, C), dtype=data.dtype)
        
        # 执行下采样，使用区域平均方法
        for i in range(new_H):
            for j in range(new_W):
                # 计算当前块的索引范围
                i_start = i * downsample_rate
                i_end = min((i + 1) * downsample_rate, H)
                j_start = j * downsample_rate
                j_end = min((j + 1) * downsample_rate, W)
                
                # 计算当前块的平均值
                block = data[i_start:i_end, j_start:j_end, :]
                downsampled_data[i, j, :] = np.mean(block, axis=(0, 1)).astype(data.dtype)
        
        # 使用下采样后的数据替换原数据
        data = downsampled_data
        H, W = new_H, new_W
   
    # 打开参考 tiff 文件，获取空间参考、仿射变换、宽度和高度等信息
    with rasterio.open(ref_tiff_path) as ref:
        ref_meta = ref.meta.copy()
        # 保留参考 tiff 的坐标系和仿射变换信息
        transform = ref.transform
        crs = ref.crs
        
        # 如果需要下采样，修改仿射变换中的像素大小
        if downsample_rate > 1:
            # 创建新的仿射变换，增加像素大小
            transform = Affine(
                transform.a * downsample_rate, transform.b, transform.c,
                transform.d, transform.e * downsample_rate, transform.f
            )
 
    # 更新元数据，新tiff的波段数为 C, 数据类型以 npy 数据为准
    new_meta = ref_meta.copy()
    new_meta.update({
        'driver': 'GTiff',
        'height': H,
        'width': W,
        'count': C,
        'dtype': data.dtype,
        'transform': transform  # 更新仿射变换信息
    })
 
    # 构造输出文件名，与 npy 文件同名但后缀改为 .tif
    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}.tif")
 
    # 写入新 tiff 文件
    with rasterio.open(out_path, 'w', **new_meta) as dst:
        # 假设 npy 数据的第3个维度为波段，对每个波段写入数据
        for i in range(C):
            dst.write(data[:, :, i], i + 1)
            print(f"波段 {i + 1} 写入完成")
 
    print(f"输出文件已保存为：{out_path}")
    print(f"分辨率：原始 10m，下采样后 {10 * downsample_rate}m")
 
if __name__ == "__main__":
    npy_path = "/media/12TBNVME/frankfeng/btfm4rs/data/representation/clement_agb/22MHC/representations_fsdp_20250417_101636.npy"  # 修改为实际 npy 文件路径
    ref_tiff_path = "/media/12TBNVME/frankfeng/btfm4rs/data/downstream/clement_agb/22MHC/red/S2B_22MHC_20180131_0_L2A.tiff"  # 修改为实际参考 tiff 文件路径
    out_dir = "/media/12TBNVME/frankfeng/btfm4rs/data/representation/clement_agb/22MHC/"  # 修改为实际输出目录
    downsample_rate = 1  # 默认不进行下采样，可根据需要修改
    convert_npy_to_tiff(npy_path, ref_tiff_path, out_dir, downsample_rate)