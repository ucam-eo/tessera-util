import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set the style to match Nature journal guidelines
plt.style.use('default')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.dpi'] = 300

# Data
ratios = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]
ratio_labels = ['0.01', '0.03', '0.05', '0.07', '0.10', '0.15', '0.20', '0.25', '0.30']

# Accuracy data
acc_transformer = [0.6721, 0.739, 0.7526, 0.7722, 0.7813, 0.8013, 0.8111, 0.8202, 0.827]
acc_rf = [0.5871, 0.6511, 0.6765, 0.6932, 0.7117, 0.7291, 0.7444, 0.7538, 0.76]
acc_lr = [0.4612, 0.5476, 0.5906, 0.6065, 0.6278, 0.6667, 0.7059, 0.7265, 0.75]
# Add new transformer* data
acc_transformer_star = [0.7077, 0.7604, 0.7746, 0.789, 0.8039, 0.8182, 0.8272, 0.834, 0.8409]

# F1 data
f1_transformer = [0.6618, 0.7254, 0.7408, 0.7622, 0.7724, 0.7937, 0.8046, 0.8143, 0.8211]
f1_rf = [0.5271, 0.6038, 0.6377, 0.6602, 0.6814, 0.7035, 0.722, 0.7335, 0.7422]
f1_lr = [0.4722, 0.549, 0.5948, 0.6122, 0.6337, 0.6705, 0.7076, 0.7272, 0.7493]
# Add new transformer* data
f1_transformer_star = [0.6909, 0.7432, 0.7638, 0.7799, 0.7956, 0.81, 0.8217, 0.8279, 0.8357]

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)

# Define colors - using Nature-style colors (colorblind-friendly)
colors = {
    'transformer': '#0072B2',  # Blue
    'rf': '#009E73',           # Green
    'lr': '#D55E00',           # Orange/Red
    'transformer_star': '#CC79A7'  # Purple (new color for transformer*)
}

# Plot Accuracy
ax1.plot(ratios, acc_transformer, color=colors['transformer'], label='Transformer', 
         marker='o', markersize=5, linewidth=1.5, alpha=0.7)
ax1.plot(ratios, acc_rf, color=colors['rf'], label='RF', 
         marker='o', markersize=5, linewidth=1.5, alpha=0.7)
ax1.plot(ratios, acc_lr, color=colors['lr'], label='LR', 
         marker='o', markersize=5, linewidth=1.5, alpha=0.7)
# Add new transformer* plot
ax1.plot(ratios, acc_transformer_star, color=colors['transformer_star'], label='Transformer*', 
         marker='o', markersize=5, linewidth=1.5, alpha=0.7)

# Plot F1 Score
ax2.plot(ratios, f1_transformer, color=colors['transformer'], label='Transformer', 
         marker='o', markersize=5, linewidth=1.5, alpha=0.7)
ax2.plot(ratios, f1_rf, color=colors['rf'], label='RF', 
         marker='o', markersize=5, linewidth=1.5, alpha=0.7)
ax2.plot(ratios, f1_lr, color=colors['lr'], label='LR', 
         marker='o', markersize=5, linewidth=1.5, alpha=0.7)
# Add new transformer* plot
ax2.plot(ratios, f1_transformer_star, color=colors['transformer_star'], label='Transformer*', 
         marker='o', markersize=5, linewidth=1.5, alpha=0.7)

# Set y-axis limits
y_min = 0.4  # Starting from 0.4 to better show the differences
y_max = 0.85  # Just above the maximum value
# Check if we need to adjust y_max for the new model data
max_value = max(max(acc_transformer_star), max(f1_transformer_star))
if max_value > y_max:
    y_max = max_value + 0.01  # Add a small margin

# Configure axes for Accuracy plot
ax1.set_xlabel('Ratio')
ax1.set_ylabel('Accuracy')
ax1.set_xlim(-0.01, 0.31)
ax1.set_ylim(y_min, y_max)
ax1.set_xticks(ratios)
ax1.set_xticklabels(ratio_labels, rotation=45, ha='right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Configure axes for F1 plot
ax2.set_xlabel('Ratio')
ax2.set_ylabel('F1 Score')
ax2.set_xlim(-0.01, 0.31)
ax2.set_ylim(y_min, y_max)
ax2.set_xticks(ratios)
ax2.set_xticklabels(ratio_labels, rotation=45, ha='right')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add legends
ax1.legend(loc='lower right', frameon=False)
ax2.legend(loc='lower right', frameon=False)

# Add subplot labels - Nature style
ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold')
ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold')

# Add title
# fig.suptitle('Small-scaled Austrian Crop Dataset Performance', fontsize=10, y=0.98)
plt.savefig('crop_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('crop_performance_comparison.pdf', bbox_inches='tight')

# plt.show() removed - not needed with Agg backend