# import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端
# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# training_ratios = [0.01, 0.05, 0.1, 0.3]
# data = {
#     "Representation Map": [(11.7468, 0.3611), (10.3378, 0.4357), (8.7605, 0.5529), (5.2077, 0.7822)],
#     "S2 RGB": [(15.1084, 0.0038), (13.0802, 0.2113), (10.0813, 0.4440), (9.97, 0.4479)],
#     "S2 MSI": [(15.3084, 0.0071), (13.1827, 0.2215), (10.0245, 0.4592), (8.78, 0.4457)],
#     "S2 MSI + S1 SAR": [(14.628, 0.0169), (12.862, 0.2154), (10.146, 0.4676), (8.11, 0.4982)]
# }

# # 颜色字典
# colors = {
#     "Representation Map": "tab:blue",
#     "S2 RGB": "tab:orange",
#     "S2 MSI": "tab:green",
#     "S2 MSI + S1 SAR": "tab:red"
# }

# # Figure 1: MAE
# plt.figure(figsize=(8, 6))
# for label, values in data.items():
#     mae_values = [v[0] for v in values]
#     plt.plot(training_ratios, mae_values, marker='o', markersize=6, alpha=0.6,
#              linestyle='-', color=colors[label], label=label)
# plt.xlabel("Training Ratio")
# plt.ylabel("MAE")
# plt.title("Mean Absolute Error (MAE)")
# plt.xticks(training_ratios)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.savefig("figure1.png", dpi=300)
# plt.close()

# # Figure 2: R²
# plt.figure(figsize=(8, 6))
# for label, values in data.items():
#     r2_values = [v[1] for v in values]
#     plt.plot(training_ratios, r2_values, marker='o', markersize=6, alpha=0.6,
#              linestyle='-', color=colors[label], label=label)
# plt.xlabel("Training Ratio")
# plt.ylabel("R²")
# plt.title("R² Score")
# plt.xticks(training_ratios)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.savefig("figure2.png", dpi=300)
# plt.close()

import numpy as np

file_path = "data/ssl_training/global/11TLK/bands.npy"
data = np.load(file_path, mmap_mode='r')
print(data.shape)
print(data[20, 100:110, 100:110, 1])  # 0.0