import os
import numpy as np
import rasterio
from sklearn.decomposition import PCA

def convert_npy_to_tiff(npy_path, ref_tiff_path, out_dir):
    # 加载 npy 数据，数据形状为 (H, W, 128)
    data = np.load(npy_path)
    H, W, C = data.shape
    
    base_name = os.path.splitext(os.path.basename(npy_path))[0]

    # 打开参考 tiff 文件，获取空间参考、仿射变换、宽度和高度等信息
    with rasterio.open(ref_tiff_path) as ref:
        ref_meta = ref.meta.copy()
        transform = ref.transform
        crs = ref.crs
        width = ref.width
        height = ref.height

    # -------------------------------
    # 生成 128 维 tiff 文件
    # -------------------------------
    new_meta = ref_meta.copy()
    new_meta.update({
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': C,
        'dtype': data.dtype
    })

    out_path = os.path.join(out_dir, f"{base_name}.tif")
    with rasterio.open(out_path, 'w', **new_meta) as dst:
        # 假设 npy 数据的第3个维度为波段，对每个波段写入数据
        for i in range(C):
            dst.write(data[:, :, i], i + 1)
            print(f"波段 {i + 1} 写入完成")
    print(f"输出文件已保存为：{out_path}")

    # -------------------------------
    # PCA 降维，将 128 维数据降至 3 维，并生成新的 tiff 文件
    # -------------------------------
    # 将数据 reshape 为二维矩阵，形状 (H*W, 128)
    reshaped_data = data.reshape(-1, C)
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(reshaped_data)
    # 如果需要，可将 PCA 结果转换为 float32（此处根据需求调整）
    data_pca = data_pca.astype(np.float32)
    # 重新 reshape 成 (H, W, 3)
    data_pca = data_pca.reshape(H, W, 3)

    # 更新元数据，新 tiff 的波段数为 3
    new_meta_pca = ref_meta.copy()
    new_meta_pca.update({
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 3,
        'dtype': data_pca.dtype
    })

    out_path_pca = os.path.join(out_dir, f"{base_name}_pca.tif")
    with rasterio.open(out_path_pca, 'w', **new_meta_pca) as dst:
        for i in range(3):
            dst.write(data_pca[:, :, i], i + 1)
            print(f"PCA 波段 {i + 1} 写入完成")
    print(f"PCA 输出文件已保存为：{out_path_pca}")

if __name__ == "__main__":
    npy_path = "/mnt/e/Codes/btfm4rs/data/downstream/cambridge/2017_fsdp_20250407_195912.npy"         # 修改为实际 npy 文件路径
    ref_tiff_path = "/mnt/e/Codes/btfm4rs/data/downstream/cambridge/cambridge.tiff"      # 修改为实际参考 tiff 文件路径
    out_dir = "/mnt/e/Codes/btfm4rs/data/downstream/cambridge"         # 修改为实际输出目录
    convert_npy_to_tiff(npy_path, ref_tiff_path, out_dir)
