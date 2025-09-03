import os
import numpy as np
import rasterio

def convert_tiff_to_npy(tiff_path, out_dir):
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
    
    # 创建输出路径
    out_path = os.path.join(out_dir, f"{base_name}.npy")
    
    # 保存为numpy文件
    np.save(out_path, data)
    print(f"NumPy 文件已保存为：{out_path}")

if __name__ == "__main__":
    tiff_path = "/mnt/e/Codes/btfm4rs/data/representation/austrian_crop_EFM_v1.0_Embeddings_2022_100x_downsampled.tif"  # 修改为实际TIFF文件路径
    out_dir = "/mnt/e/Codes/btfm4rs/data/representation"  # 修改为实际输出目录
    convert_tiff_to_npy(tiff_path, out_dir)