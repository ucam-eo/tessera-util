import argparse
import rasterio
import numpy as np
from rasterio.transform import Affine

def downsample_tiff(input_path, scale_factor, output_path):
    print("\n[INFO] 开始处理文件:", input_path)
    
    with rasterio.open(input_path) as src:
        # 记录原始元数据
        print(f"[原始文件] 波段数={src.count}, 宽度={src.width}, 高度={src.height}")
        print(f"[原始文件] 分辨率: X={src.transform[0]:.2f} m, Y={-src.transform[4]:.2f} m")
        print(f"[参数] 降采样因子={scale_factor:.1f}x")

        # 读取原始数据
        print("[操作] 正在读取原始数据...")
        data = src.read()
        
        # 通过选择每隔scale_factor个像素进行降采样
        print("[操作] 正在执行像素抽取降采样...")
        scale_factor_int = int(scale_factor)
        
        # 计算实际降采样后的维度（使用切片实际会产生的大小）
        sampled_height = len(range(0, src.height, scale_factor_int))
        sampled_width = len(range(0, src.width, scale_factor_int))
        print(f"[计算] 实际新宽度={sampled_width}, 实际新高度={sampled_height}")
        
        downsampled_data = np.zeros((src.count, sampled_height, sampled_width), dtype=data.dtype)
        
        # 对每个波段执行相同的操作
        for i in range(src.count):
            downsampled_data[i] = data[i, ::scale_factor_int, ::scale_factor_int]
            
        # 更新变换矩阵
        new_transform = src.transform * Affine.scale(scale_factor, scale_factor)
        print(f"[变换] 新地理参考矩阵:\n{new_transform}")

        # 更新元数据
        kwargs = src.meta.copy()
        kwargs.update({
            'height': sampled_height,
            'width': sampled_width,
            'transform': new_transform
        })

    # 写入新文件
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        dst.write(downsampled_data)
        print(f"[输出] 文件已保存: {output_path}")
        print(f"[输出] 波段数={kwargs['count']}, 宽度={kwargs['width']}, 高度={kwargs['height']}")
        print(f"[输出] 新分辨率: X={new_transform[0]:.2f} m, Y={-new_transform[4]:.2f} m\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 rasterio 降低 TIFF 文件分辨率")
    parser.add_argument("--input_path", default="data/representation/austrian_crop_EFM_Embeddings_2022.tif", help="输入 TIFF 文件路径")
    parser.add_argument("--scale_factor", type=float, default=10.0, help="降采样因子（例如 20 表示分辨率降低为原来的1/20）")
    parser.add_argument("--output_path", default="data/representation/austrian_crop_EFM_Embeddings_2022_downsample_100.tif", help="输出 TIFF 文件路径")
    args = parser.parse_args()

    downsample_tiff(args.input_path, args.scale_factor, args.output_path)