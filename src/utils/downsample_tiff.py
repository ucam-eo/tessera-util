import argparse
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

def downsample_tiff(input_path, scale_factor, output_path):
    print("\n[INFO] 开始处理文件:", input_path)
    
    with rasterio.open(input_path) as src:
        # 记录原始元数据
        print(f"[原始文件] 波段数={src.count}, 宽度={src.width}, 高度={src.height}")
        print(f"[原始文件] 分辨率: X={src.transform[0]:.2f} m, Y={-src.transform[4]:.2f} m")
        print(f"[参数] 降采样因子={scale_factor:.1f}x")

        # 计算新尺寸
        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)
        print(f"[计算] 新宽度={new_width}, 新高度={new_height}")

        # 重采样读取
        print("[操作] 正在执行重采样...")
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.average
        )

        # 更新变换矩阵
        new_transform = src.transform * Affine.scale(
            src.width / new_width, 
            src.height / new_height
        )
        print(f"[变换] 新地理参考矩阵:\n{new_transform}")

        # 更新元数据
        kwargs = src.meta.copy()
        kwargs.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })

    # 写入新文件
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        dst.write(data)
        print(f"[输出] 文件已保存: {output_path}")
        print(f"[输出] 波段数={kwargs['count']}, 宽度={kwargs['width']}, 高度={kwargs['height']}")
        print(f"[输出] 新分辨率: X={new_transform[0]:.2f} m, Y={-new_transform[4]:.2f} m\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 rasterio 降低 TIFF 文件分辨率")
    parser.add_argument("input_path", help="输入 TIFF 文件路径")
    parser.add_argument("scale_factor", type=float, help="降采样因子（例如 20 表示分辨率降低为原来的1/20）")
    parser.add_argument("output_path", help="输出 TIFF 文件路径")
    args = parser.parse_args()

    downsample_tiff(args.input_path, args.scale_factor, args.output_path)