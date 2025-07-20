import os
import numpy as np
import rasterio

def tiff_to_numpy(input_tiff, output_npy=None):
    """
    将TIFF文件转换为NumPy数组，并将通道顺序从(B,H,W)转为(H,W,B)
    
    参数:
        input_tiff: 输入TIFF文件路径
        output_npy: 输出NumPy文件路径，如果不指定，则自动将.tif替换为.npy
    """
    # 如果没有指定输出路径，自动生成
    if output_npy is None:
        output_npy = os.path.splitext(input_tiff)[0] + '.npy'
    
    print(f"\n[INFO] 正在处理: {input_tiff}")
    
    # 读取TIFF文件
    with rasterio.open(input_tiff) as src:
        print(f"[读取] 波段数={src.count}, 宽度={src.width}, 高度={src.height}")
        
        # 读取所有波段
        data = src.read()
        print(f"[数据] 原始形状: {data.shape} (B,H,W)")
        
        # 转置为(H,W,B)
        data = np.transpose(data, (1, 2, 0))
        print(f"[数据] 转置后形状: {data.shape} (H,W,B)")
        
        # 保存为NumPy文件
        np.save(output_npy, data)
        print(f"[输出] NumPy数组已保存至: {output_npy}")
        
        # 计算文件大小
        input_size = os.path.getsize(input_tiff) / (1024 * 1024)
        output_size = os.path.getsize(output_npy) / (1024 * 1024)
        print(f"[统计] 输入文件大小: {input_size:.2f} MB")
        print(f"[统计] 输出文件大小: {output_size:.2f} MB")
        
        return data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将TIFF文件转换为NumPy数组并调整通道顺序")
    parser.add_argument("input_tiff", help="输入TIFF文件路径")
    parser.add_argument("--output_npy", help="输出NumPy文件路径（可选，默认替换扩展名为.npy）")
    args = parser.parse_args()
    
    tiff_to_numpy(args.input_tiff, args.output_npy)