import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from collections import Counter
import logging
from contextlib import contextmanager
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib as mpl
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

TRAINING_RATIO = 0.01
VAL_TEST_SPLIT_RATIO = 1/7.0  # Validation to test set split ratio
MODEL = "MLP"  # Options: "LogisticRegression", "RandomForest", or "MLP"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# MLP hyperparameters
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 5  # Early stopping parameter

# Log settings
log_file = "feature_analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
)
logging.info("Program started.")
logging.info(f"Using device: {DEVICE}")

# Configuration parameters
njobs = 12
chunk_size = 1000

# bands_file_path = "/scratch/zf281/austrian_crop_whole_year/bands_downsample_100.npy"
# label_file_path = "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes_downsample_100.npy"
# field_id_file_path = "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldid_downsample_100.npy"
# updated_fielddata_path = '/maps/zf281/btfm-training-10.4/maddy_code/data/updated_fielddata.csv'

# # sar_asc_bands_file_path = "/maps/zf281/btfm4rs/data/ssl_training/austrian_crop/sar_ascending_downsample_100.npy"
# sar_asc_bands_file_path = "/scratch/zf281/austrian_crop_whole_year/sar_ascending_downsample_100.npy"
# # sar_desc_bands_file_path = "/maps/zf281/btfm4rs/data/ssl_training/austrian_crop/sar_descending_downsample_100.npy"
# sar_desc_bands_file_path = "/scratch/zf281/austrian_crop_whole_year/sar_descending_downsample_100.npy"

bands_file_path = "/maps/zf281/btfm4rs/data/downstream/austrian_crop_v1.0_pipeline/bands_downsample_100.npy"
label_file_path = "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldtype_17classes_downsample_100.npy"
field_id_file_path = "/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldid_downsample_100.npy"
updated_fielddata_path = '/maps/zf281/btfm-training-10.4/maddy_code/data/updated_fielddata.csv'

# sar_asc_bands_file_path = "/maps/zf281/btfm4rs/data/ssl_training/austrian_crop/sar_ascending_downsample_100.npy"
sar_asc_bands_file_path = "/maps/zf281/btfm4rs/data/downstream/austrian_crop_v1.0_pipeline/sar_ascending_downsample_100.npy"
# sar_desc_bands_file_path = "/maps/zf281/btfm4rs/data/ssl_training/austrian_crop/sar_descending_downsample_100.npy"
sar_desc_bands_file_path = "/maps/zf281/btfm4rs/data/downstream/austrian_crop_v1.0_pipeline/sar_descending_downsample_100.npy"

# Sentinel-2 normalization parameters
S2_BAND_MEAN = np.array([1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
                        2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
                       1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779], dtype=np.float32)
S1_BAND_MEAN = np.array([5484.0407,3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334,1726.0670], dtype=np.float32)

# Class names for visualization
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

# ----------------- Define MLP model -----------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(MLP, self).__init__()
        
        # First layer with BatchNorm and ReLU
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Second layer with BatchNorm and ReLU
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Third layer with BatchNorm and ReLU
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[2], num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

# Function to train MLP model
def train_mlp(X_train, y_train, X_val, y_val, num_classes, input_size):
    """Train MLP model and return the trained model"""
    logging.info(f"Starting MLP training with input size: {input_size}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    hidden_sizes = [512, 256, 128]  # Three hidden layer sizes
    model = MLP(input_size, hidden_sizes, num_classes).to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Train the model
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    logging.info(f"MLP model architecture:\n{model}")
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            logging.info(f"Saving best model with validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= PATIENCE:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

# MLP model prediction function
def mlp_predict(model, X):
    """Make predictions using MLP model"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    
    # Batch predict for large datasets
    batch_size = 1024
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.cpu().numpy())
    
    return np.concatenate(predictions)

# ----------------- Data loading and preprocessing -----------------
logging.info(f"Training ratio: {TRAINING_RATIO}")
logging.info(f"Validation/Test split ratio: {VAL_TEST_SPLIT_RATIO}")
logging.info(f"Selected model: {MODEL}")
logging.info("Loading labels and field IDs...")
labels = np.load(label_file_path).astype(np.int64)
field_ids = np.load(field_id_file_path)
H, W = labels.shape
logging.info(f"Data dimensions: {H}x{W}")

# Select valid classes
logging.info("Identifying valid classes...")
class_counts = Counter(labels.ravel())
valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
valid_classes.discard(0)
logging.info(f"Valid classes: {sorted(valid_classes)}")

# ----------------- Train/validation/test set split -----------------
logging.info("Splitting data into train/val/test sets...")
fielddata_df = pd.read_csv(updated_fielddata_path)
area_summary = fielddata_df.groupby('SNAR_CODE')['area_m2'].sum().reset_index()
area_summary.rename(columns={'area_m2': 'total_area'}, inplace=True)

# Collect training set field IDs
train_fids = []
for _, row in area_summary.iterrows():
    sn_code = row['SNAR_CODE']
    total_area = row['total_area']
    target_area = total_area * TRAINING_RATIO
    rows_sncode = fielddata_df[fielddata_df['SNAR_CODE'] == sn_code].sort_values(by='area_m2')
    selected_fids = []
    selected_area_sum = 0
    for _, r2 in rows_sncode.iterrows():
        if selected_area_sum < target_area:
            selected_fids.append(int(r2['fid_1']))
            selected_area_sum += r2['area_m2']
        else:
            break
    train_fids.extend(selected_fids)

train_fids = list(set(train_fids))
logging.info(f"Number of selected train field IDs: {len(train_fids)}")

# Split remaining field IDs into validation and test sets
all_fields = fielddata_df['fid_1'].unique().astype(int)
set_train = set(train_fids)
set_all = set(all_fields)
remaining = list(set_all - set_train)
remaining = np.array(remaining)
np.random.shuffle(remaining)
val_count = int(len(remaining) * VAL_TEST_SPLIT_RATIO)
val_fids = remaining[:val_count]
test_fids = remaining[val_count:]
train_fids = np.array(train_fids)
logging.info(f"Train fields: {len(train_fids)}, Val fields: {len(val_fids)}, Test fields: {len(test_fids)}")

# ----------------- Create training/validation/testing split map -----------------
logging.info("Creating train/val/test split map for visualization...")
# Now we need to create the train/test/val mask using field_ids
train_test_mask = np.zeros((H, W), dtype=np.int8)

# Fill in the mask based on field IDs
for i in range(H):
    for j in range(W):
        fid = field_ids[i, j]
        if fid in train_fids:
            train_test_mask[i, j] = 1  # Training
        elif fid in val_fids:
            train_test_mask[i, j] = 2  # Validation
        elif fid in test_fids:
            train_test_mask[i, j] = 3  # Testing

# ----------------- Chunk processing -----------------
def process_chunk(h_start, h_end, w_start, w_end, file_path):
    logging.info(f"Processing chunk: h[{h_start}:{h_end}], w[{w_start}:{w_end}]")
    
    # Load S2
    # Load and normalize data
    tile_chunk = np.load(file_path)[:, h_start:h_end, w_start:w_end, :] # (time, h, w, bands)
    tile_chunk = (tile_chunk - S2_BAND_MEAN) / S2_BAND_STD
    # Reshape data
    time_steps, h, w, bands = tile_chunk.shape
    s2_band_chunk = tile_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands) # (h*w, time_steps*bands)
    
    # Load S1
    sar_asc_chunk = np.load(sar_asc_bands_file_path)[:, h_start:h_end, w_start:w_end]
    sar_desc_chunk = np.load(sar_desc_bands_file_path)[:, h_start:h_end, w_start:w_end]
    # Concatenate along time dimension
    sar_chunk = np.concatenate((sar_asc_chunk, sar_desc_chunk), axis=0)
    # Normalize
    sar_chunk = (sar_chunk - S1_BAND_MEAN) / S1_BAND_STD
    # Reshape data
    time_steps, h, w, bands = sar_chunk.shape
    sar_band_chunk = sar_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands) # (h*w, time_steps*bands)
    # Concatenate S2 and S1
    X_chunk = np.concatenate((s2_band_chunk, sar_band_chunk), axis=1) # (h*w, time_steps*bands*2)
    
    y_chunk = labels[h_start:h_end, w_start:w_end].ravel()
    fieldid_chunk = field_ids[h_start:h_end, w_start:w_end].ravel()
    
    # Filter valid data
    valid_mask = np.isin(y_chunk, list(valid_classes))
    X_chunk, y_chunk, fieldid_chunk = X_chunk[valid_mask], y_chunk[valid_mask], fieldid_chunk[valid_mask]
    
    # Split into train/val/test sets based on field_id
    train_mask = np.isin(fieldid_chunk, train_fids)
    val_mask = np.isin(fieldid_chunk, val_fids)
    test_mask = np.isin(fieldid_chunk, test_fids)
    
    return (X_chunk[train_mask], y_chunk[train_mask], 
            X_chunk[val_mask], y_chunk[val_mask],
            X_chunk[test_mask], y_chunk[test_mask])

# Parallel processing
chunks = [(h, min(h+chunk_size, H), w, min(w+chunk_size, W))
         for h in range(0, H, chunk_size)
         for w in range(0, W, chunk_size)]
logging.info(f"Total chunks: {len(chunks)}")

results = Parallel(n_jobs=njobs)(
    delayed(process_chunk)(h_start, h_end, w_start, w_end, bands_file_path)
    for h_start, h_end, w_start, w_end in chunks
)

# Combine results
X_train = np.vstack([res[0] for res in results if res[0].size > 0])
y_train = np.hstack([res[1] for res in results if res[1].size > 0])
X_val = np.vstack([res[2] for res in results if res[2].size > 0])
y_val = np.hstack([res[3] for res in results if res[3].size > 0])
X_test = np.vstack([res[4] for res in results if res[4].size > 0])
y_test = np.hstack([res[5] for res in results if res[5].size > 0])

logging.info(f"Data split summary:")
logging.info(f"  Train set: {X_train.shape[0]} samples")
logging.info(f"  Validation set: {X_val.shape[0]} samples")
logging.info(f"  Test set: {X_test.shape[0]} samples")

# Print data shapes
logging.info(f"X_train shape: {X_train.shape}")
input_size = X_train.shape[1]
logging.info(f"Input feature dimension: {input_size}")

# ----------------- Model training -----------------
logging.info(f"\nTraining {MODEL}...")

if MODEL == "LogisticRegression":
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=1e4,
        max_iter=100000,
        n_jobs=njobs,
        random_state=42
    )
    model.fit(X_train, y_train)
    
elif MODEL == "RandomForest":
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=njobs,
        random_state=42
    )
    model.fit(X_train, y_train)
    
elif MODEL == "MLP":
    # Get number of classes
    num_classes = max(valid_classes) + 1
    logging.info(f"Number of classes: {num_classes}")
    
    # Train MLP model
    mlp_model = train_mlp(X_train, y_train, X_val, y_val, num_classes, input_size)
    
    # Create a wrapper class for consistency with other models
    class MLPWrapper:
        def __init__(self, model):
            self.model = model
            
        def predict(self, X):
            return mlp_predict(self.model, X)
    
    model = MLPWrapper(mlp_model)
    
else:
    raise ValueError(f"Unknown model type: {MODEL}. Use 'LogisticRegression', 'RandomForest', or 'MLP'")

# ----------------- Evaluation -----------------
logging.info("Evaluating model on test set...")
y_pred = model.predict(X_test)
logging.info("Classification Report (Test Set):\n" + classification_report(y_test, y_pred, digits=4))

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

# Define a plotting function for consistent formatting
def plot_classification_map(data, title, cmap, class_names, save_path, figsize=(12, 10)):
    """Create a nicely formatted classification map without colorbar."""
    plt.figure(figsize=figsize, dpi=300)
    
    # Set up the plot with publication quality
    plt.rcParams.update({
        'font.family': 'sans-serif',  # Use a generic font family available everywhere
        'font.size': 12,
        'axes.linewidth': 1.5
    })
    
    # Plot the data
    im = plt.imshow(data, cmap=cmap, interpolation='nearest')
    
    # No colorbar - removed as requested
    
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
        
        # Add legend outside the plot with larger text and make it more prominent
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), 
                   loc='upper left', fontsize=14, frameon=True, fancybox=True, 
                   shadow=True, title="Classes", title_fontsize=15)
    
    # Add title and style adjustments
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
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

# 3. Training/Testing Split Map
logging.info("Generating training/testing/validation split map...")
train_test_cmap = ListedColormap(['white', 'blue', 'green', 'red'])  # White for background, blue for training, green for validation, red for testing
train_test_names = ["Background", "Training Set", "Validation Set", "Testing Set"]

plot_classification_map(
    train_test_mask, 
    f"Training, Validation and Testing Sample Distribution", 
    train_test_cmap, 
    train_test_names, 
    f"train_test_split_map.png"
)

# ----------------- Generating Prediction Map (Optimized) -----------------
logging.info("Generating prediction map - this may take some time...")

# Create prediction map
pred_map = np.zeros_like(labels)

# Optimized batch prediction for a whole chunk
def batch_predict_chunk(h_start, h_end, w_start, w_end):
    """Process and predict a chunk of the image more efficiently."""
    # Create mask for valid classes in this chunk
    chunk_labels = labels[h_start:h_end, w_start:w_end]
    chunk_fieldids = field_ids[h_start:h_end, w_start:w_end]
    
    # Create empty prediction array for this chunk
    chunk_pred = np.zeros_like(chunk_labels)
    
    # Identify valid pixels that need prediction (non-training pixels with valid classes)
    valid_mask = np.isin(chunk_labels, list(valid_classes))
    # Get mask for test/val pixels (those not in training)
    non_train_mask = ~np.isin(chunk_fieldids, train_fids)
    # Combine masks to get pixels that need prediction
    predict_mask = valid_mask & non_train_mask
    
    # Copy training pixels directly (they should match ground truth)
    train_mask = valid_mask & ~non_train_mask
    chunk_pred[train_mask] = chunk_labels[train_mask]
    
    # If there are no pixels to predict, return early
    if not np.any(predict_mask):
        return h_start, h_end, w_start, w_end, chunk_pred
    
    # Get coordinates of pixels that need prediction
    h_indices, w_indices = np.where(predict_mask)
    
    # Load data for feature extraction (only once per chunk)
    s2_data = np.load(bands_file_path)[:, h_start:h_end, w_start:w_end, :]
    sar_asc_data = np.load(sar_asc_bands_file_path)[:, h_start:h_end, w_start:w_end]
    sar_desc_data = np.load(sar_desc_bands_file_path)[:, h_start:h_end, w_start:w_end]
    
    # Batch size for processing within chunk
    batch_size = 1000
    for i in range(0, len(h_indices), batch_size):
        batch_h = h_indices[i:i+batch_size]
        batch_w = w_indices[i:i+batch_size]
        
        # Extract features for this batch of pixels
        batch_features = []
        for j in range(len(batch_h)):
            h_idx, w_idx = batch_h[j], batch_w[j]
            
            # S2 feature extraction
            s2_pixel = s2_data[:, h_idx, w_idx, :]
            s2_norm = (s2_pixel - S2_BAND_MEAN) / S2_BAND_STD
            s2_features = s2_norm.reshape(-1)
            
            # S1 feature extraction
            sar_asc_pixel = sar_asc_data[:, h_idx, w_idx]
            sar_desc_pixel = sar_desc_data[:, h_idx, w_idx]
            sar_pixel = np.concatenate((sar_asc_pixel, sar_desc_pixel))
            sar_norm = (sar_pixel - S1_BAND_MEAN) / S1_BAND_STD
            sar_features = sar_norm.reshape(-1)
            
            # Combine features
            features = np.concatenate((s2_features, sar_features))
            
            batch_features.append(features)
        
        # Convert to numpy array
        batch_features = np.array(batch_features)
        
        # Batch prediction
        batch_preds = model.predict(batch_features)
        
        # Place predictions into chunk
        for j in range(len(batch_h)):
            h_idx, w_idx = batch_h[j], batch_w[j]
            chunk_pred[h_idx, w_idx] = batch_preds[j]
    
    return h_start, h_end, w_start, w_end, chunk_pred

# Define chunks for parallel processing of prediction map
pred_chunks = [(h, min(h+chunk_size, H), w, min(w+chunk_size, W))
              for h in range(0, H, chunk_size)
              for w in range(0, W, chunk_size)]

# Process prediction map in parallel
logging.info("Processing prediction map in parallel (optimized)...")
start_time = time.time()

pred_results = Parallel(n_jobs=njobs)(
    delayed(batch_predict_chunk)(h_start, h_end, w_start, w_end)
    for h_start, h_end, w_start, w_end in pred_chunks
)

# Combine prediction results
for h_start, h_end, w_start, w_end, chunk_pred in pred_results:
    pred_map[h_start:h_end, w_start:w_end] = chunk_pred

end_time = time.time()
logging.info(f"Prediction map generation completed in {end_time - start_time:.2f} seconds")

# 2. Model Prediction Map
logging.info("Saving model prediction classification map...")
model_name = MODEL
plot_classification_map(
    pred_map, 
    f"{model_name} Classification Predictions", 
    cmap, 
    class_names, 
    f"prediction_map_{model_name.lower()}.png"
)

# Generate a composite map that shows the differences between prediction and ground truth
logging.info("Generating prediction difference map...")
diff_map = np.zeros_like(labels)

# Calculate difference map
for h in range(H):
    for w in range(W):
        if labels[h, w] in valid_classes and pred_map[h, w] > 0:
            if field_ids[h, w] in train_fids:
                # Training pixels - should match ground truth (but mark differently)
                diff_map[h, w] = 1  # Training pixel
            else:
                # Test/Val pixels - check if prediction matches ground truth
                diff_map[h, w] = 2 if pred_map[h, w] == labels[h, w] else 3  # 2=correct, 3=incorrect

diff_cmap = ListedColormap(['white', 'blue', 'lightgray', 'red'])  # White=background, blue=training, gray=correct, red=incorrect
diff_names = ["Background", "Training", "Correct Prediction", "Incorrect Prediction"]

plot_classification_map(
    diff_map, 
    f"{model_name} Prediction Accuracy", 
    diff_cmap, 
    diff_names, 
    f"prediction_difference_map_{model_name.lower()}.png"
)

# Calculate and log the accuracy statistics
test_pixels = 0
test_correct = 0
val_pixels = 0
val_correct = 0

for h in range(H):
    for w in range(W):
        if labels[h, w] in valid_classes:
            fid = field_ids[h, w]
            if fid in test_fids:
                test_pixels += 1
                if pred_map[h, w] == labels[h, w]:
                    test_correct += 1
            elif fid in val_fids:
                val_pixels += 1
                if pred_map[h, w] == labels[h, w]:
                    val_correct += 1

test_accuracy = test_correct / test_pixels * 100 if test_pixels > 0 else 0
val_accuracy = val_correct / val_pixels * 100 if val_pixels > 0 else 0

logging.info(f"\nTest set prediction statistics:")
logging.info(f"Test pixels: {test_pixels}, Correct: {test_correct}, Accuracy: {test_accuracy:.2f}%")
logging.info(f"Validation pixels: {val_pixels}, Correct: {val_correct}, Accuracy: {val_accuracy:.2f}%")

# Per-class accuracy statistics
class_accuracies = {}
for cls in sorted(valid_classes):
    # Count test pixels for this class
    cls_pixels = 0
    cls_correct = 0
    
    for h in range(H):
        for w in range(W):
            if labels[h, w] == cls and field_ids[h, w] in test_fids:
                cls_pixels += 1
                if pred_map[h, w] == cls:
                    cls_correct += 1
    
    if cls_pixels > 0:
        cls_accuracy = cls_correct / cls_pixels * 100
        class_accuracies[cls] = cls_accuracy
        class_name = class_names[cls-1] if cls-1 < len(class_names) else f"Class {cls}"
        logging.info(f"Class {cls} ({class_name}) accuracy: {cls_accuracy:.2f}% ({cls_correct}/{cls_pixels} pixels)")

# Generate a class accuracy bar plot
plt.figure(figsize=(14, 8))

# Set font to a common font available on all systems
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12
})

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
plt.yticks(range(len(sorted_classes)), sorted_labels, fontsize=12)
plt.xlabel('Accuracy (%)', fontsize=14)
plt.title(f'{model_name} - Per-Class Accuracy', fontsize=16)
plt.axvline(test_accuracy, color='red', linestyle='--', label=f'Overall Accuracy: {test_accuracy:.1f}%')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'class_accuracy_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
plt.close()

logging.info("\nAnalysis completed!")
print("Process finished. Logs saved to:", log_file)
print("Classification maps saved as PNG files.")

# If using MLP model, save the model
# if MODEL == "MLP":
#     model_save_path = f'mlp_model.pt'
#     torch.save(mlp_model.state_dict(), model_save_path)
#     logging.info(f"MLP model saved to {model_save_path}")