import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib as mpl

# Set random seed for reproducibility
np.random.seed(42)
TRAINING_RATIO = 0.3

# Argument parser for model selection
parser = argparse.ArgumentParser(description="Train a model on remote sensing data")
parser.add_argument("--model", type=str, choices=["lr", "rf"], default="rf",
                   help="Model type: 'lr' for Logistic Regression, 'rf' for Random Forest")
args = parser.parse_args()

# Logging setup
log_file = "feature_analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
)
logging.info(f"Program started. Using model: {args.model}")

# Configuration parameters
njobs = 12
bands_file_path = "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/bands_downsample_100.npy" # shape (T,H,W,C)
label_file_path = "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes_downsample_100.npy" # shape (H,W)
# Added cloud mask file path
masks_file_path = "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/MGRS_33UXP/masks_downsample_100.npy" # shape (T,H,W)

# Sentinel-2 normalization parameters
S2_BAND_MEAN = np.array([1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
                        2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
                       1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779], dtype=np.float32)

# Class names mapping
class_names = [
    "Legume",
    "Soy",
    "Summer Grain",
    "Winter Grain",
    "Corn",
    "Sunflower",
    "Mustard",
    "Potato",
    "Beet",
    "Squash",
    "Grapes",
    "Tree Fruit",
    "Cover Crop",
    "Grass",
    "Fallow",
    "Other (Plants)",
    "Other (Non Plants)"
]

# ----------------- Data Loading & Preprocessing -----------------
logging.info("Loading labels...")
labels = np.load(label_file_path).astype(np.int64)
H, W = labels.shape
logging.info(f"Data dimensions: {H}x{W}")

# Valid class selection
logging.info("Identifying valid classes...")
class_counts = Counter(labels.ravel())
valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
valid_classes.discard(0)  # Remove background/no-data class if present
logging.info(f"Valid classes: {sorted(valid_classes)}")

# Create a mask for valid pixels
valid_mask = np.isin(labels, list(valid_classes))
valid_indices = np.where(valid_mask.ravel())[0]
logging.info(f"Found {len(valid_indices)} valid pixels")

# Generate random train/test split on valid pixels only
np.random.shuffle(valid_indices)
split_idx = int(TRAINING_RATIO * len(valid_indices))
train_indices = valid_indices[:split_idx]
test_indices = valid_indices[split_idx:]
logging.info(f"Selected {len(train_indices)} pixels for training ({split_idx/len(valid_indices):.2%})")

# ----------------- Process Full Dataset -----------------
logging.info("Loading and processing the full dataset...")

# Load the full dataset
tile_data = np.load(bands_file_path)
time_steps, h, w, bands = tile_data.shape
logging.info(f"Loaded data with shape: {tile_data.shape}")

# Load cloud masks
logging.info("Loading and applying cloud masks...")
cloud_masks = np.load(masks_file_path)
logging.info(f"Loaded cloud masks with shape: {cloud_masks.shape}")

# Apply cloud masks to zero out invalid pixels (before normalization)
# Create a copy of the data to avoid modifying the original
tile_data_masked = tile_data.copy()

# Apply the mask - expand dimensions for broadcasting
# cloud_masks shape is (T,H,W), need to reshape to (T,H,W,1) for broadcasting
expanded_masks = cloud_masks.reshape(time_steps, h, w, 1)

# Zero out invalid pixels (where mask is 0)
tile_data_masked = tile_data_masked * expanded_masks

logging.info("Cloud masks applied successfully")

# Normalize data - properly reshape for broadcasting
# S2_BAND_MEAN and S2_BAND_STD need to be reshaped to (1, 1, 1, 10)
# For shape (T, H, W, C) = (99, 459, 518, 10)

means = S2_BAND_MEAN.reshape(1, 1, 1, -1)
stds = S2_BAND_STD.reshape(1, 1, 1, -1)
tile_data_normalized = (tile_data_masked - means) / stds

# Get coordinates for all valid pixels
valid_coords = np.unravel_index(valid_indices, (H, W))

# Extract normalized features for valid pixels
X = []
y = []

logging.info("Extracting features for valid pixels...")
start_time = time.time()
for idx in range(len(valid_indices)):
    h_idx, w_idx = valid_coords[0][idx], valid_coords[1][idx]
    # Extract all timesteps and bands for this pixel
    pixel_features = tile_data_normalized[:, h_idx, w_idx, :].reshape(-1)
    X.append(pixel_features)
    y.append(labels[h_idx, w_idx])

X = np.array(X)
y = np.array(y)
end_time = time.time()
logging.info(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
logging.info(f"Processed data shape: {X.shape}")

# Create train and test masks based on indices
train_mask = np.zeros(len(valid_indices), dtype=bool)
train_mask[:split_idx] = True  # First split_idx indices are for training

# Split data into train and test sets
X_train = X[train_mask]
y_train = y[train_mask]
X_val = X[~train_mask]
y_val = y[~train_mask]

logging.info(f"Training data shape: {X_train.shape}, labels: {y_train.shape}")
logging.info(f"Validation data shape: {X_val.shape}, labels: {y_val.shape}")

# ----------------- Model Training -----------------
bands_per_time = len(S2_BAND_MEAN)
n_times = X_train.shape[1] // bands_per_time
logging.info(f"Features: {bands_per_time} bands × {n_times} time steps")

# Generate feature names for later use
feature_names = [f'band{b+1}_time{t+1}' 
                for t in range(n_times) 
                for b in range(bands_per_time)]

# Train model based on selected model type
if args.model == "lr":
    # Logistic Regression Model
    logging.info("\nTraining Logistic Regression...")
    start_time = time.time()
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=1e4,  # Reduced regularization
        max_iter=100000,
        n_jobs=njobs,
        random_state=42
    )
    model.fit(X_train, y_train)
    end_time = time.time()
    logging.info(f"Logistic Regression training completed in {end_time - start_time:.2f} seconds")
    
else:
    # Random Forest Model
    logging.info("\nTraining Random Forest Classifier...")
    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=njobs,
        random_state=42
    )
    model.fit(X_train, y_train)
    end_time = time.time()
    logging.info(f"Random Forest training completed in {end_time - start_time:.2f} seconds")
    
    # Additional RF-specific logging
    logging.info(f"Number of trees: {model.n_estimators}")
    logging.info(f"OOB Score: {model.oob_score_:.4f}" if hasattr(model, 'oob_score_') else "OOB Score not computed")

# ----------------- Evaluation -----------------
logging.info("Evaluating model...")
start_time = time.time()
y_pred = model.predict(X_val)
end_time = time.time()
logging.info(f"Prediction completed in {end_time - start_time:.2f} seconds")
logging.info("Classification Report:\n" + classification_report(y_val, y_pred, digits=4))

# ----------------- Feature Analysis -----------------
logging.info("\nStarting Feature Coefficient Analysis...")

if args.model == "lr":
    # LR-specific feature importance analysis using coefficients
    # Create coefficient matrix
    coef_matrix = np.abs(model.coef_)
    mean_coef = coef_matrix.mean(axis=0)
    
    # Create analysis DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_coef
    })
    
    # Sign direction analysis
    sign_matrix = pd.DataFrame(
        np.sign(model.coef_.mean(axis=0).reshape(bands_per_time, n_times)),
        index=range(1, bands_per_time+1),
        columns=range(1, n_times+1)
    )
    logging.info("\n[Sign Matrix] Dominant Effect Directions:")
    logging.info(sign_matrix.round(2).to_string())
    
else:
    # RF-specific feature importance analysis
    logging.info("Extracting feature importances from Random Forest...")
    feature_importances = model.feature_importances_
    
    # Create analysis DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    
    # Log top 20 most important features
    top_features = df.sort_values('importance', ascending=False).head(20)
    logging.info("\n[Top 20 Features] RF Feature Importance:")
    logging.info(top_features.to_string(index=False))

# Extract band and time info
df[['band', 'time']] = df['feature'].str.extract(r'band(\d+)_time(\d+)').astype(int)

# Band-wise aggregation
band_agg = df.groupby('band', as_index=False)['importance'].mean()
# logging.info("\n[Band Importance] Average importance:")
# logging.info(band_agg.round(4).sort_values('importance', ascending=False).to_string(index=False))

# Time-wise aggregation
time_agg = df.groupby('time', as_index=False)['importance'].mean()
# logging.info("\n[Time Importance] Average importance:")
# logging.info(time_agg.round(4).sort_values('importance', ascending=False).to_string(index=False))

# Heatmap generation
heatmap_data = df.pivot_table(
    index='band',
    columns='time',
    values='importance',
    aggfunc=np.mean
)
# logging.info("\n[Heatmap Matrix] Band-Time Importance:")
# logging.info(heatmap_data.round(3).to_string())

# Generate heatmap visualization
plt.figure(figsize=(50, 5))  # Adjust width based on number of time points
ax = sns.heatmap(
    heatmap_data,
    cmap="viridis",
    linewidths=0.5,
    cbar_kws={'label': 'Feature Importance'}
)

model_name = "LogisticRegression" if args.model == "lr" else "RandomForest"
ax.set_title(f"Band-Time Importance Heatmap ({model_name})", fontsize=14, pad=20)
ax.set_xlabel("Time Index", fontsize=12)
ax.set_ylabel("Band Number", fontsize=12)
ax.figure.axes[-1].yaxis.label.set_size(12)  # Color bar label font

# Auto-rotate x-axis labels
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

# Save and close
plt.savefig(
    f'band_time_heatmap_{args.model}.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()  # Free memory

# Add RF-specific visualizations
# if args.model == "rf":
#     # Plot feature importance
#     plt.figure(figsize=(12, 8))
#     top_n = 30  # Show top 30 features
#     top_features = df.sort_values('importance', ascending=False).head(top_n)
    
#     plt.barh(range(top_n), top_features['importance'], align='center')
#     plt.yticks(range(top_n), top_features['feature'])
#     plt.xlabel('Importance')
#     plt.ylabel('Feature')
#     plt.title(f'Top {top_n} Feature Importances (Random Forest)')
#     plt.tight_layout()
#     plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Additional analysis only available for RF
#     logging.info("\nRandom Forest - Additional Analysis:")
    
#     # Per-class importance analysis
#     if hasattr(model, 'estimators_'):
#         n_classes = len(model.classes_)
#         logging.info(f"Analyzing per-class feature importance for {n_classes} classes...")
        
#         # For brevity, just analyze the top 3 classes by sample count
#         class_counts = Counter(y_train)
#         top_classes = [cls for cls, _ in class_counts.most_common(3)]
        
#         for target_class in top_classes:
#             class_trees = [tree for i, tree in enumerate(model.estimators_) 
#                          if model.classes_[np.argmax(model.estimators_samples_[i], axis=1)[0]] == target_class]
            
#             if class_trees:
#                 class_importance = np.mean([tree.feature_importances_ for tree in class_trees], axis=0)
#                 logging.info(f"\nClass {target_class} - Top 10 Important Features:")
#                 top_idx = np.argsort(class_importance)[-10:][::-1]
#                 for i, idx in enumerate(top_idx):
#                     logging.info(f"{i+1}. {feature_names[idx]}: {class_importance[idx]:.4f}")

# Add mask statistics to the log
invalid_mask_count = np.sum(cloud_masks == 0)
total_pixels = cloud_masks.size
percent_invalid = (invalid_mask_count / total_pixels) * 100
logging.info(f"\nCloud mask statistics:")
logging.info(f"Total invalid (cloud-covered) pixels: {invalid_mask_count} out of {total_pixels} ({percent_invalid:.2f}%)")

# Calculate coverage per time step
for t in range(time_steps):
    time_invalid = np.sum(cloud_masks[t] == 0)
    time_total = h * w
    time_percent_invalid = (time_invalid / time_total) * 100
    logging.info(f"Time step {t+1}: {time_percent_invalid:.2f}% cloud-covered")

logging.info("\nAnalysis completed!")

# ----------------- Classification Map Generation -----------------
logging.info("\nGenerating classification maps...")

# Generate a color palette (using tab20 and extending if needed)
def get_color_palette(n_classes):
    """Generate a color palette for classification maps."""
    # Start with tab20 which has 20 distinct colors
    base_cmap = plt.cm.get_cmap('tab20')
    colors = [base_cmap(i) for i in range(20)]
    
    # If we need more colors, add from other colormaps
    if n_classes > 20:
        extra_cmap = plt.cm.get_cmap('tab20b')
        colors.extend([extra_cmap(i) for i in range(n_classes - 20)])
    
    # Return only the number of colors we need
    return colors[:n_classes]

# Setup for visualization
# Add 1 to max class for background (0)
max_class = max(valid_classes) 
n_classes = len(valid_classes)
logging.info(f"Creating color mapping for {n_classes} classes (1-{max_class})")

# Generate color palette
colors = get_color_palette(max_class + 1)  # +1 for background class (0)
# Set background (0) to white
colors[0] = (1, 1, 1, 1)  # White for background

# Create colormap
cmap = ListedColormap(colors)

# Create a mask to represent training and testing pixels
train_test_mask = np.zeros((H, W), dtype=np.int8)
# Convert valid_coords back to 2D indices
h_train_indices = np.array([valid_coords[0][i] for i in range(len(train_mask)) if train_mask[i]])
w_train_indices = np.array([valid_coords[1][i] for i in range(len(train_mask)) if train_mask[i]])
h_test_indices = np.array([valid_coords[0][i] for i in range(len(train_mask)) if not train_mask[i]])
w_test_indices = np.array([valid_coords[1][i] for i in range(len(train_mask)) if not train_mask[i]])

# Mark training pixels as 1, testing pixels as 2
train_test_mask[h_train_indices, w_train_indices] = 1  # Training
train_test_mask[h_test_indices, w_test_indices] = 2    # Testing

# Create an empty prediction map of the same shape as labels
pred_map = np.zeros_like(labels)

# Place validation predictions in the correct spatial locations
for i, pred_val in enumerate(y_pred):
    # Get the spatial indices for this validation sample
    h_idx, w_idx = valid_coords[0][split_idx + i], valid_coords[1][split_idx + i]
    pred_map[h_idx, w_idx] = pred_val

# Place training values in the prediction map (these pixels were used for training)
for i, train_val in enumerate(y_train):
    h_idx, w_idx = valid_coords[0][i], valid_coords[1][i]
    pred_map[h_idx, w_idx] = train_val

# Define a plotting function for consistent formatting
def plot_classification_map(data, title, cmap, class_names, save_path, figsize=(12, 10)):
    """Create a nicely formatted classification map."""
    plt.figure(figsize=figsize, dpi=300)
    
    # Set up the plot with publication quality
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.5
    })
    
    # Plot the data
    im = plt.imshow(data, cmap=cmap, interpolation='nearest')
    
    # Add a color bar
    cbar = plt.colorbar(im, ticks=range(len(class_names)), shrink=0.8)
    
    # Create a legend with class names
    if class_names:
        # Get the number of unique classes in the data
        unique_classes = sorted(np.unique(data))
        # Filter out 0 if it's background
        if 0 in unique_classes and len(unique_classes) > 1:
            unique_classes = [c for c in unique_classes if c > 0]
        
        # Create legend patches for each class
        legend_patches = []
        for cls in unique_classes:
            if cls == 0:
                continue  # Skip background
            if cls <= len(class_names):
                # Use class color from colormap
                color = cmap(cls / max(unique_classes))
                label = class_names[cls-1] if cls-1 < len(class_names) else f"Class {cls}"
                legend_patches.append(mpatches.Patch(color=color, label=label))
        
        # Add legend outside the plot
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), 
                  loc='upper left', fontsize=10, frameon=True)
    
    # Add title and style adjustments
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Add scale bar (simple 100-pixel scale)
    scale_bar_size = 100  # pixels
    scalebar_x = 20
    scalebar_y = data.shape[0] - 30
    plt.plot([scalebar_x, scalebar_x + scale_bar_size], 
             [scalebar_y, scalebar_y], 'k-', linewidth=3)
    plt.text(scalebar_x + scale_bar_size/2, scalebar_y + 15, 
             f"{scale_bar_size} pixels", ha='center', fontsize=10)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved classification map to {save_path}")

# 1. Original Ground Truth Map
logging.info("Generating ground truth classification map...")
plot_classification_map(
    labels, 
    f"Ground Truth Land Cover Classification", 
    cmap, 
    class_names, 
    f"ground_truth_map.png"
)

# 2. Model Prediction Map
logging.info("Generating model prediction classification map...")
plot_classification_map(
    pred_map, 
    f"{model_name} Classification Predictions", 
    cmap, 
    class_names, 
    f"prediction_map_{args.model}.png"
)

# 3. Training/Testing Split Map
logging.info("Generating training/testing split map...")
train_test_cmap = ListedColormap(['white', 'blue', 'red'])  # White for background, blue for training, red for testing
train_test_names = ["Background", "Training Set", "Testing Set"]

plot_classification_map(
    train_test_mask, 
    f"Training and Testing Sample Distribution", 
    train_test_cmap, 
    train_test_names, 
    f"train_test_split_map.png"
)

# Generate a composite map that shows the differences between prediction and ground truth
logging.info("Generating prediction difference map...")
diff_map = np.zeros_like(labels)
# Only compute difference for test pixels
diff_map[h_test_indices, w_test_indices] = (pred_map[h_test_indices, w_test_indices] != 
                                           labels[h_test_indices, w_test_indices]).astype(int)

diff_cmap = ListedColormap(['white', 'lightgray', 'red'])  # White for background, gray for correct, red for incorrect
diff_names = ["Background", "Correct Prediction", "Incorrect Prediction"]

plot_classification_map(
    diff_map, 
    f"{model_name} Prediction Accuracy", 
    diff_cmap, 
    diff_names, 
    f"prediction_difference_map_{args.model}.png"
)

# Calculate and log the accuracy statistics for the test set
test_correct = np.sum(diff_map[h_test_indices, w_test_indices] == 0)
test_incorrect = np.sum(diff_map[h_test_indices, w_test_indices] == 1)
test_accuracy = test_correct / (test_correct + test_incorrect) * 100

logging.info(f"\nTest set prediction statistics:")
logging.info(f"Correct predictions: {test_correct} pixels")
logging.info(f"Incorrect predictions: {test_incorrect} pixels")
logging.info(f"Overall accuracy: {test_accuracy:.2f}%")

# Per-class accuracy statistics
class_accuracies = {}
for cls in sorted(valid_classes):
    # Get indices where ground truth is this class
    cls_indices = np.where((labels == cls) & (train_test_mask == 2))
    if len(cls_indices[0]) > 0:
        cls_correct = np.sum(pred_map[cls_indices] == cls)
        cls_accuracy = cls_correct / len(cls_indices[0]) * 100
        class_accuracies[cls] = cls_accuracy
        class_name = class_names[cls-1] if cls-1 < len(class_names) else f"Class {cls}"
        logging.info(f"Class {cls} ({class_name}) accuracy: {cls_accuracy:.2f}% ({cls_correct}/{len(cls_indices[0])} pixels)")

# Generate a class accuracy bar plot
plt.figure(figsize=(14, 8))
classes = list(class_accuracies.keys())
accuracies = list(class_accuracies.values())
class_labels = [class_names[cls-1] if cls-1 < len(class_names) else f"Class {cls}" for cls in classes]

# Sort by accuracy for better visualization
sorted_indices = np.argsort(accuracies)
sorted_classes = [classes[i] for i in sorted_indices]
sorted_accuracies = [accuracies[i] for i in sorted_indices]
sorted_labels = [class_labels[i] for i in sorted_indices]

# Plot horizontal bar chart
bars = plt.barh(range(len(sorted_classes)), sorted_accuracies, color=[cmap(cls/max(classes)) for cls in sorted_classes])
plt.yticks(range(len(sorted_classes)), sorted_labels)
plt.xlabel('Accuracy (%)')
plt.title(f'{model_name} - Per-Class Accuracy', fontsize=16)
plt.axvline(test_accuracy, color='red', linestyle='--', label=f'Overall Accuracy: {test_accuracy:.1f}%')
plt.legend()
plt.tight_layout()
plt.savefig(f'class_accuracy_{args.model}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Process finished. Model: {args.model}. Logs saved to: {log_file}")
print(f"Classification maps saved as PNG files.")