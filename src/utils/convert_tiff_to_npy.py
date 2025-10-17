import os
import numpy as np
import rasterio
from scipy.ndimage import zoom

def convert_tiff_to_npy(tiff_path, out_dir, out_dim=(5643, 5565)):
    # 获取输出文件的基本名称
    base_name = os.path.splitext(os.path.basename(tiff_path))[0]
    
    # 打开TIFF文件
    with rasterio.open(tiff_path) as src:
        # 获取元数据
        meta = src.meta
        height = meta['height']
        width = meta['width']
        count = meta['count']  # 波段数量

        # 读取所有波段
        data = np.zeros((height, width, count), dtype=meta['dtype'])
        for i in range(count):
            data[:, :, i] = src.read(i + 1)  # rasterio的波段索引从1开始
            print(f"读取波段 {i + 1} 完成", f"形状: {data[:, :, i].shape}")

    # 如果指定了输出尺寸，进行resize
    if out_dim != (height, width):
        target_height, target_width = out_dim
        # 计算缩放因子
        zoom_factors = (target_height / height, target_width / width, 1)
        # 对数据进行resize
        data = zoom(data, zoom_factors, order=1)  # order=1 表示双线性插值
        print(f"数据已resize从 {(height, width)} 到 {out_dim}")

    # 创建输出路径
    out_path = os.path.join(out_dir, f"{base_name}.npy")
    
    # 保存为numpy文件
    np.save(out_path, data)
    print(f"NumPy 文件已保存为：{out_path}")

if __name__ == "__main__":
    outdim = (4509, 5826)  # 修改为所需的输出尺寸
    tiff_path = "/maps/zf281/btfm4rs/data/downstream/pv_detection/london/london_map_10m_utm30n_128bands_cropped.tiff"  # 修改为实际TIFF文件路径
    out_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/london"  # 修改为实际输出目录
    convert_tiff_to_npy(tiff_path, out_dir, out_dim=outdim)