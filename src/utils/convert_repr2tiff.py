import os
import numpy as np
import rasterio

def convert_npy_to_tiff(npy_path, ref_tiff_path, out_dir):
    # 加载 npy 数据，假设数据形状为 (H, W, C)
    data = np.load(npy_path)
    H, W, C = data.shape
    
    # 打开参考 tiff 文件，获取空间参考、仿射变换、宽度和高度等信息
    with rasterio.open(ref_tiff_path) as ref:
        ref_meta = ref.meta.copy()
        # 保留参考 tiff 的坐标系、仿射变换和尺寸信息
        transform = ref.transform
        crs = ref.crs
        width = ref.width
        height = ref.height

    # 更新元数据，新tiff的波段数为 C, 数据类型以 npy 数据为准
    new_meta = ref_meta.copy()
    new_meta.update({
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': C,
        'dtype': data.dtype
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


if __name__ == "__main__":
    npy_path = "/scratch/zf281/robin/fungal/representation_from_dawn/34VEH/representation.npy"         # 修改为实际 npy 文件路径
    ref_tiff_path = "/scratch/zf281/robin/fungal/data_raw/34VEH/red/S2A_34VEH_20210107_0_L2A.tiff"      # 修改为实际参考 tiff 文件路径
    out_dir = "/scratch/zf281/robin/fungal/representation_from_dawn/34VEH"         # 修改为实际输出目录
    convert_npy_to_tiff(npy_path, ref_tiff_path, out_dir)