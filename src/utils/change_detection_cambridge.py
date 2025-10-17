import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# --- 1. 参数设置 ---
# 定义分析的年份
YEAR1 = 2022
YEAR2 = 2023

# 定义文件路径
BASE_DIR = '/scratch/zf281/btfm_representation/cambridge'
GEOJSON_PATH = '/scratch/zf281/geotessera/example/CB.geojson'
OUTPUT_DIR = './' # 可以指定您希望保存输出的目录

# 定义文件名模板
TIFF_TEMPLATE = os.path.join(BASE_DIR, '{year}_cambridge_map_10m_utm31n_128bands.tiff')
SCALES_TEMPLATE = os.path.join(BASE_DIR, '{year}_cambridge_map_10m_utm31n_scales.npy')
OUTPUT_PNG_PATH = os.path.join(OUTPUT_DIR, f'cambridge_change_detection_{YEAR1}_{YEAR2}_geojson_roi.png')

# 变化检测阈值 (重缩放后, 0表示无变化, 2表示最大变化)
CHANGE_THRESHOLD = 0.5 

# --- 2. 辅助函数 ---

def load_and_dequantize_representation(representation_file_path, scales_file_path):
    """
    加载int8 TIFF表征并将其反量化为float32。
    注意：此函数已修改为使用rasterio读取TIFF文件。

    Args:
        representation_file_path (str): int8表征文件的路径 (TIFF格式)
        scales_file_path (str): float32缩放文件的路径 (.npy格式)

    Returns:
        np.ndarray: float32类型的表征数组，形状为 (H, W, C)
    """
    # 使用rasterio加载TIFF文件
    with rasterio.open(representation_file_path) as src:
        # rasterio读取的形状是(C, H, W)，需要转换为(H, W, C)
        representation_int8 = src.read().transpose((1, 2, 0))
    
    # 加载缩放因子
    scales = np.load(scales_file_path)  # (H, W), dtype=float32
    
    # 将int8转换为float32进行计算
    representation_f32 = representation_int8.astype(np.float32)
    
    # 扩展缩放因子的维度以匹配表征的形状
    # scales shape: (H, W) -> (H, W, 1)
    scales_expanded = scales[..., np.newaxis]
    
    # 通过乘以缩放因子进行反量化
    representation_f32 = representation_f32 * scales_expanded
    
    return representation_f32

def normalize_embeddings_with_mask(embeddings, mask):
    """
    对嵌入向量进行归一化，只处理ROI内的像素。

    Args:
        embeddings (np.ndarray): 输入的嵌入向量数组，形状为 (h, w, c)
        mask (np.ndarray): ROI掩码，形状为 (h, w)，布尔类型

    Returns:
        np.ndarray: 归一化后的嵌入向量，ROI外的像素保持为零
    """
    # 创建一个与输入形状相同的零数组
    normalized = np.zeros_like(embeddings, dtype=np.float32)
    
    # 仅提取ROI内的像素进行处理
    roi_pixels = embeddings[mask]  # 形状为 (num_roi_pixels, c)
    
    # 计算每个向量的L2范数（模长）
    magnitudes = np.linalg.norm(roi_pixels, axis=1, keepdims=True)
    
    # 避免除以零
    # 创建一个与roi_pixels形状相同的零数组用于存放结果
    normalized_roi = np.zeros_like(roi_pixels, dtype=np.float32)
    # 只在模长不为零的地方进行除法操作
    np.divide(roi_pixels, magnitudes, out=normalized_roi, where=magnitudes != 0)
    
    # 将归一化后的值放回掩码指定的位置
    normalized[mask] = normalized_roi
    
    return normalized

# --- 3. 主流程 ---

print("--- 开始变化检测分析 ---")
print(f"比较年份: {YEAR1} vs {YEAR2}")

# --- 3.1. 从GeoJSON创建ROI掩码 ---
print(f"正在从 {GEOJSON_PATH} 创建ROI掩码...")
ref_tiff_path = TIFF_TEMPLATE.format(year=YEAR1)
with rasterio.open(ref_tiff_path) as src:
    ref_meta = src.meta
    ref_transform = src.transform
    ref_crs = src.crs
    height, width = src.height, src.width

# 读取GeoJSON
gdf = gpd.read_file(GEOJSON_PATH)

# 确保GeoJSON和栅格数据的坐标参考系（CRS）一致
if gdf.crs != ref_crs:
    print(f"GeoJSON CRS ({gdf.crs}) 与栅格CRS ({ref_crs})不匹配，正在转换...")
    gdf = gdf.to_crs(ref_crs)

# 将GeoJSON几何图形栅格化为掩码
roi_mask = rasterize(
    shapes=gdf.geometry,
    out_shape=(height, width),
    transform=ref_transform,
    fill=0,
    default_value=1,
    dtype=np.uint8
).astype(bool)

print(f"ROI掩码形状: {roi_mask.shape}")
print(f"ROI内像素数量: {np.sum(roi_mask)}")

# --- 3.2. 加载、反量化和归一化数据 ---
print(f"正在加载并反量化 {YEAR1} 年的数据...")
embedding1 = load_and_dequantize_representation(
    TIFF_TEMPLATE.format(year=YEAR1),
    SCALES_TEMPLATE.format(year=YEAR1)
)
print(f"{YEAR1}年表征形状: {embedding1.shape}")

print(f"正在加载并反量化 {YEAR2} 年的数据...")
embedding2 = load_and_dequantize_representation(
    TIFF_TEMPLATE.format(year=YEAR2),
    SCALES_TEMPLATE.format(year=YEAR2)
)
print(f"{YEAR2}年表征形状: {embedding2.shape}")


print("正在对ROI内的嵌入向量进行归一化...")
normalized1 = normalize_embeddings_with_mask(embedding1, roi_mask)
normalized2 = normalize_embeddings_with_mask(embedding2, roi_mask)

# --- 3.3. 计算变化 ---
print("正在计算点积和变化幅度...")
# 初始化一个全零数组
dot_product = np.zeros(roi_mask.shape, dtype=np.float32)

# 仅在ROI内部计算点积
# 点积是余弦相似度，因为向量已经归一化
dot_product[roi_mask] = np.sum(normalized1[roi_mask] * normalized2[roi_mask], axis=1)

# 重缩放结果：(cos_sim * -1) + 1
# 这样，无变化(cos_sim=1) -> 0, 最大变化(cos_sim=-1) -> 2
rescaled_change = np.zeros_like(dot_product, dtype=np.float32)
rescaled_change[roi_mask] = (dot_product[roi_mask] * -1) + 1

# 创建显著变化区域的掩码
change_mask = (rescaled_change > CHANGE_THRESHOLD) & roi_mask

# --- 3.4. 可视化和保存 ---
print("正在生成并保存变化图...")
plt.figure(figsize=(12, 10))

# 创建一个用于显示的数组，ROI外部设为NaN以实现透明背景
display_change = np.full(rescaled_change.shape, np.nan, dtype=float)
display_change[roi_mask] = rescaled_change[roi_mask]

# 绘制底层的连续变化图 (可选，如果只想看高亮可以注释掉)
plt.imshow(display_change, cmap='viridis', vmin=0, vmax=2)

# 创建自定义颜色映射：从完全透明到不透明的黄色
colors = [(1, 1, 0, 0), (1, 1, 0, 1)]  # RGBA: (Red, Green, Blue, Alpha)
cmap_overlay = LinearSegmentedColormap.from_list('change_cmap', colors)

# 创建一个用于高亮变化的叠加层
overlay_data = np.full(change_mask.shape, np.nan, dtype=float)
overlay_data[change_mask] = 1.0  # 在变化区域赋值

# 在原图上叠加高亮层
plt.imshow(overlay_data, cmap=cmap_overlay, vmin=0, vmax=1, alpha=0.8)

plt.title(f'Land Type Change Detection ({YEAR1} vs {YEAR2}) for Cambridge (CB.geojson)', fontsize=16)
plt.axis('off')
plt.tight_layout()

# 保存图形，而不是显示
plt.savefig(OUTPUT_PNG_PATH, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
plt.close() # 关闭图形，释放内存

print(f"变化检测图已成功保存至: {OUTPUT_PNG_PATH}")

# --- 3.5. 统计分析 ---
print("\n--- 统计分析报告 ---")
roi_pixels_count = np.sum(roi_mask)
change_pixels_count = np.sum(change_mask)

if roi_pixels_count > 0:
    change_percentage = (change_pixels_count / roi_pixels_count) * 100
    roi_change_values = rescaled_change[roi_mask]

    print(f"ROI总像素数: {roi_pixels_count}")
    print(f"检测到的变化像素数 (阈值 > {CHANGE_THRESHOLD}): {change_pixels_count}")
    print(f"ROI内发生显著变化的区域百分比: {change_percentage:.2f}%")
    
    print(f"\nROI内变化值统计 (0=无变化, 2=最大变化):")
    print(f"  最小值: {np.min(roi_change_values):.4f}")
    print(f"  最大值: {np.max(roi_change_values):.4f}")
    print(f"  平均值: {np.mean(roi_change_values):.4f}")
    print(f"  中位数: {np.median(roi_change_values):.4f}")
    print(f"  标准差: {np.std(roi_change_values):.4f}")
else:
    print("ROI掩码为空，无法进行统计分析。请检查GeoJSON文件和栅格数据是否重叠。")

print("\n分析完成！")