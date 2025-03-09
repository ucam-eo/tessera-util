import argparse
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

def downsample_tiff(input_path, scale_factor, output_path):
    with rasterio.open(input_path) as src:
        # 计算新的宽度和高度
        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)

        # 使用重采样读取数据，默认采用平均重采样方法（适用于连续数据）
        data = src.read(
            out_shape=(
                src.count,
                new_height,
                new_width
            ),
            resampling=Resampling.average
        )

        # 计算新的仿射变换，新像元尺寸变大
        new_transform = src.transform * Affine.scale(src.width / new_width, src.height / new_height)

        # 更新元数据
        kwargs = src.meta.copy()
        kwargs.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })

    # 写入新的 TIFF 文件
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        dst.write(data)
    print(f"降采样后的 TIFF 文件已保存：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 rasterio 降低 TIFF 文件分辨率")
    parser.add_argument("input_path", help="输入 TIFF 文件路径")
    parser.add_argument("scale_factor", type=float, help="降采样因子（例如 20 表示分辨率降低为原来的1/20）")
    parser.add_argument("output_path", help="输出 TIFF 文件路径")
    args = parser.parse_args()

    downsample_tiff(args.input_path, args.scale_factor, args.output_path)