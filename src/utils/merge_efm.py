import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.plot import show
import glob
import os
# 添加这个导入
from shapely.geometry import box, mapping

def merge_and_crop_to_roi(input_dir, roi_path, output_path):
    # 定义临时文件路径
    temp_merged_path = os.path.join(input_dir, "temp_merged.tif")
    
    # 检查临时文件是否已经存在
    if not os.path.exists(temp_merged_path):
        # 临时文件不存在，执行合并步骤
        # 读取所有分块文件（假设文件名按坐标命名）
        tiff_files = glob.glob(os.path.join(input_dir, "*embeddings_2022*.tif"))
        print(f"找到 {len(tiff_files)} 个分块文件:")
        for f in tiff_files:
            print(f" - {os.path.basename(f)}")

        # 按文件名排序（确保合并顺序正确）
        tiff_files.sort()
        
        # 打开所有分块文件
        srcs = [rasterio.open(f) for f in tiff_files]
        
        # 合并分块
        print("\n[合并] 开始合并分块...")
        merged_data, merged_transform = merge(srcs)
        merged_meta = srcs[0].meta.copy()
        merged_meta.update({
            "height": merged_data.shape[1],
            "width": merged_data.shape[2],
            "transform": merged_transform,
            "driver": "GTiff"
        })
        for src in srcs:
            src.close()
        print(f"[合并] 合并完成，形状={merged_data.shape}")

        # 临时保存合并后的 TIFF（后续裁剪用）
        with rasterio.open(temp_merged_path, "w", **merged_meta) as dst:
            dst.write(merged_data)
        print(f"[临时文件] 已保存: {temp_merged_path}")
    else:
        # 临时文件已存在，跳过合并步骤
        print(f"[临时文件] 发现已存在: {temp_merged_path}，跳过合并步骤")

    # 读取 ROI 文件的空间范围
    with rasterio.open(roi_path) as roi_src:
        roi_bounds = roi_src.bounds
        roi_crs = roi_src.crs
        roi_transform = roi_src.transform
        roi_height = roi_src.height
        roi_width = roi_src.width
        roi_data = roi_src.read(1)
        print(f"\n[ROI] 空间范围={roi_bounds}")
        print(f"[ROI] 分辨率={roi_transform[0]} m, 尺寸={roi_width}x{roi_height}")

    # 裁剪合并后的文件到 ROI 范围
    with rasterio.open(temp_merged_path) as merged_src:
        # 确保坐标系一致
        if merged_src.crs != roi_crs:
            raise ValueError("ROI 与合并文件的坐标系不一致！")
        
        # 从边界框创建一个几何对象
        roi_box = box(roi_bounds.left, roi_bounds.bottom, roi_bounds.right, roi_bounds.top)
        roi_feature = mapping(roi_box)  # 转换为GeoJSON格式
        
        # 使用正确格式的几何对象进行裁剪
        print("\n[裁剪] 正在裁剪到 ROI 范围...")
        out_image, out_transform = mask(
            merged_src,
            [roi_feature],  # 使用GeoJSON格式的几何对象
            crop=True,
            all_touched=True
        )
        # 更新元数据
        out_meta = merged_src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": roi_crs
        })
        print(f"[裁剪] 裁剪后形状={out_image.shape}")

    # 写入最终输出文件
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)
    print(f"\n[输出] 最终文件已保存: {output_path}")

    # 删除临时文件
    os.remove(temp_merged_path)
    print("[清理] 临时文件已删除")

if __name__ == "__main__":
    input_dir = r"/mnt/d/btfm_temp_files/22MHB"
    roi_path = r"/mnt/d/btfm_temp_files/S2B_22MHB_20180131_0_L2A.tiff"  # 注意这里已修正为tiff
    output_path = r"/mnt/d/btfm_temp_files/22MHB_EFM_Embeddings_2022_ROI.tif"
    
    merge_and_crop_to_roi(input_dir, roi_path, output_path)