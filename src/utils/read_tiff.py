import rasterio

def print_tiff_info(tiff_path):
    with rasterio.open(tiff_path) as src:
        width = src.width
        height = src.height
        crs = src.crs
        transform = src.transform  # Affine对象

        # 像元分辨率通常存储在仿射变换中：
        # transform.a 为 x 分辨率，transform.e 为 y 分辨率（通常为负值）
        pixel_size_x = transform.a
        pixel_size_y = -transform.e  # 取绝对值

        # 左上角坐标
        ul_x, ul_y = transform * (0, 0)

        print(f"宽度: {width}")
        print(f"高度: {height}")
        print(f"坐标参考系: {crs}")
        print(f"仿射变换: {transform}")
        print(f"像元分辨率: x方向 = {pixel_size_x}, y方向 = {pixel_size_y}")
        print(f"左上角坐标: ({ul_x}, {ul_y})")

if __name__ == "__main__":
    tiff_path = '/scratch/zf281/robin/fungal/estonia_roi.tif'  # 修改为你的tiff文件路径
    print_tiff_info(tiff_path)
