import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt错误
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.ndimage import zoom

def align_shapes(representations, labels):
    """调整representations的空间维度以匹配labels"""
    rep_h, rep_w, rep_d = representations.shape
    label_h, label_w = labels.shape
    
    print(f"Original representations shape: ({rep_h}, {rep_w}, {rep_d})")
    print(f"Labels shape: ({label_h}, {label_w})")
    
    if rep_h == label_h and rep_w == label_w:
        print("Shapes already match, no alignment needed")
        return representations
    
    # 计算缩放因子
    zoom_h = label_h / rep_h
    zoom_w = label_w / rep_w
    
    print(f"Zoom factors: height={zoom_h:.4f}, width={zoom_w:.4f}")
    
    # 使用scipy的zoom函数进行调整
    aligned_representations = zoom(representations, (zoom_h, zoom_w, 1), order=1)
    
    print(f"Aligned representations shape: {aligned_representations.shape}")
    
    return aligned_representations

# 加载数据
# representation_path = "data/representation/austrian_crop_EFM_v1.0_Embeddings_2022_100x_downsampled.npy"
representation_path = "data/representation/austria_Presto_embeddings_100m.npy"
label_path = "data/downstream/austrian_crop/fieldtype_17classes_downsample_100.npy"

print("Loading data...")
representations = np.load(representation_path)  # (H, W, D)
# 把nan值替换为0
representations = np.nan_to_num(representations, nan=0.0)
labels = np.load(label_path).astype(np.int64)  # (H, W)

# 对齐空间维度
representations = align_shapes(representations, labels)

# 1. 可视化representation - 取前三个维度作为RGB
print("\nVisualizing representation...")
# 取前三个维度
rgb_representation = representations[:, :, :3].copy()
# 转为float
rgb_representation = rgb_representation.astype(np.float32)

# 归一化到[0, 1]范围
for i in range(3):
    channel = rgb_representation[:, :, i]
    min_val = np.min(channel)
    max_val = np.max(channel)
    print(f"Channel {i+1} - min: {min_val}, max: {max_val}")
    if max_val > min_val:
        rgb_representation[:, :, i] = (channel - min_val) / (max_val - min_val)
    else:
        rgb_representation[:, :, i] = 0

# 保存representation图像
plt.figure(figsize=(10, 10))
plt.imshow(rgb_representation)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('representation_rgb.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

# 2. 可视化label
print("Visualizing labels...")
# 为每个类别分配颜色
unique_labels = np.unique(labels)
n_classes = len(unique_labels)

# 创建颜色映射
# 使用tab20颜色映射，适合多类别
if n_classes <= 20:
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_classes]
else:
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

# 创建颜色映射字典
color_map = {}
for i, label in enumerate(unique_labels):
    color_map[label] = colors[i]

# 将标签转换为RGB图像
label_rgb = np.zeros((labels.shape[0], labels.shape[1], 3))
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        label_rgb[i, j] = color_map[labels[i, j]][:3]  # 只取RGB通道

# 保存label图像
plt.figure(figsize=(10, 10))
plt.imshow(label_rgb)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('labels_colored.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print("\nVisualization complete!")
print("Saved: representation_rgb.png")
print("Saved: labels_colored.png")

# 打印一些统计信息
print(f"\nFinal shape check:")
print(f"Representation (after alignment): {representations.shape}")
print(f"Labels: {labels.shape}")
print(f"Number of unique labels: {n_classes}")
print(f"Label values: {unique_labels}")