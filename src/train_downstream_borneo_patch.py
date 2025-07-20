import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("unet_models")

# Set target normalization parameters (calculated based on non-NaN values)
target_mean = 33.573902
target_std = 18.321875

####################################
# 1. Define Dataset class and filter patches that are all NaN
####################################
class BorneoDataset(Dataset):
    def __init__(self, rep_files, target_files, transform=None):
        """
        :param rep_files: List of representation .npy files, shape (H,W,C)
        :param target_files: List of target .npy files, shape (H,W), where some values may be NaN
        :param transform: Optional transform
        """
        # Filter out patches where the target is entirely NaN
        valid_rep_files = []
        valid_target_files = []
        for rep_file, target_file in zip(rep_files, target_files):
            target = np.load(target_file)
            if np.all(np.isnan(target)):
                # print(f"Skipping {target_file} as all values are NaN.")
                continue
            valid_rep_files.append(rep_file)
            valid_target_files.append(target_file)
        
        self.rep_files = valid_rep_files
        self.target_files = valid_target_files
        self.transform = transform
        
    def __len__(self):
        return len(self.rep_files)
    
    def __getitem__(self, idx):
        # Load representation and target
        rep = np.load(self.rep_files[idx])
        target = np.load(self.target_files[idx])
        
        # --- THIS IS THE CORRECTED LINE ---
        # Replace any NaN values in the input representation with 0 to prevent NaN loss.
        rep = np.nan_to_num(rep, nan=0.0)
        
        # representation: (H, W, C) -> convert to (C, H, W)
        rep = torch.from_numpy(rep).float().permute(2, 0, 1)
        
        # target: (H, W) -> convert to (1, H, W)
        target = torch.from_numpy(target).float().unsqueeze(0)
        # Normalize the target, NaN values remain unchanged
        target = (target - target_mean) / target_std
        
        if self.transform:
            rep = self.transform(rep)
            target = self.transform(target)
            
        return rep, target

####################################
# 2. Original UNet Model Definition
####################################
class DoubleConv(nn.Module):
    """Two 3x3 convolutions + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=128, out_channels=1, features=[128, 256, 512]):
        """
        :param in_channels: Input channels (here 128)
        :param out_channels: Output channels (set to 1 for regression task)
        :param features: Number of features in each layer of the encoder
        """
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Build encoder (Downsampling)
        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in_channels, feature))
            curr_in_channels = feature
        
        # Bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Build decoder (Upsampling)
        for feature in reversed(features):
            # Use transposed convolution for upsampling
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            # Channel number after concatenation is 2*feature, then reduced back to feature through double convolution
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Final 1x1 convolution, mapping channel number to 1
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        # Encoder part: save features of each stage for skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse to correspond with the decoder
        
        # Decoder part: upsampling, concatenation, double convolution
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            # If dimensions do not match, adjust using interpolation
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
            
        return self.final_conv(x)

####################################
# 3. UNet Model based on Depthwise Separable Convolution
####################################
class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution module"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            padding=padding, stride=stride, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ChannelAttention(nn.Module):
    """Channel Attention mechanism"""
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure reduction_ratio does not result in zero channels
        reduction = max(1, channels // reduction_ratio)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """Spatial Attention mechanism"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Ensure kernel_size is odd
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class DepthwiseUNetBlock(nn.Module):
    """Basic UNet block using Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, 
                 use_channel_attention=False, use_spatial_attention=False,
                 channel_reduction=16, use_residual=False):
        super(DepthwiseUNetBlock, self).__init__()
        
        self.use_residual = use_residual
        self.residual_conv = None
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # First depthwise separable convolution
        self.depthwise_conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second depthwise separable convolution
        self.depthwise_conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Attention mechanism
        self.channel_attention = ChannelAttention(out_channels, channel_reduction) if use_channel_attention else None
        self.spatial_attention = SpatialAttention() if use_spatial_attention else None
        
    def forward(self, x):
        identity = x
        
        # First convolution
        x = self.depthwise_conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second convolution
        x = self.depthwise_conv2(x)
        x = self.bn2(x)
        
        # Channel attention
        if self.channel_attention is not None:
            x = self.channel_attention(x)
            
        # Spatial attention
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
        
        # Residual connection
        if self.use_residual:
            if self.residual_conv is not None:
                identity = self.residual_conv(identity)
            x = x + identity
            
        x = self.relu2(x)
        return x

class DepthwiseUNet(nn.Module):
    """UNet using Depthwise Separable Convolution"""
    def __init__(self, in_channels=128, out_channels=1,
                 features=[64, 128, 256, 512], 
                 use_channel_attention=True,
                 use_spatial_attention=False,
                 channel_reduction=16,
                 use_residual=True,
                 dropout_rate=0.1,
                 use_deep_supervision=False,
                 use_bilinear_upsample=False):
        """
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param features: List of feature numbers for each layer
        :param use_channel_attention: Whether to use channel attention
        :param use_spatial_attention: Whether to use spatial attention
        :param channel_reduction: Channel reduction ratio in channel attention
        :param use_residual: Whether to use residual connections
        :param dropout_rate: Dropout rate
        :param use_deep_supervision: Whether to use deep supervision
        :param use_bilinear_upsample: Whether to use bilinear upsampling instead of transposed convolution
        """
        super(DepthwiseUNet, self).__init__()
        
        self.use_deep_supervision = use_deep_supervision
        self.use_bilinear_upsample = use_bilinear_upsample
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        # Encoder part
        in_channels_temp = in_channels
        for feature in features:
            self.downs.append(
                DepthwiseUNetBlock(
                    in_channels_temp, feature,
                    use_channel_attention=use_channel_attention,
                    use_spatial_attention=use_spatial_attention,
                    channel_reduction=channel_reduction,
                    use_residual=use_residual
                )
            )
            in_channels_temp = feature
        
        # Bottleneck
        self.bottleneck = DepthwiseUNetBlock(
            features[-1], features[-1]*2,
            use_channel_attention=use_channel_attention,
            use_spatial_attention=use_spatial_attention,
            channel_reduction=channel_reduction,
            use_residual=use_residual
        )
        
        # Decoder part
        self.deep_outputs = nn.ModuleList() if use_deep_supervision else None
        
        for idx, feature in enumerate(reversed(features)):
            # Upsampling - optional bilinear or transposed convolution
            if use_bilinear_upsample:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(
                            features[-idx-1]*2 if idx == 0 else features[-idx], 
                            feature, 
                            kernel_size=1, 
                            bias=False
                        ),
                        nn.BatchNorm2d(feature),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.ups.append(
                    nn.ConvTranspose2d(
                        features[-idx-1]*2 if idx == 0 else features[-idx],
                        feature, 
                        kernel_size=2, 
                        stride=2
                    )
                )
            
            # Convolution block
            self.ups.append(
                DepthwiseUNetBlock(
                    feature*2, feature,
                    use_channel_attention=use_channel_attention,
                    use_spatial_attention=use_spatial_attention,
                    channel_reduction=channel_reduction,
                    use_residual=use_residual
                )
            )
            
            # Deep supervision output
            if use_deep_supervision and idx > 0:  # Skip the deepest layer
                self.deep_outputs.append(
                    nn.Conv2d(feature, out_channels, kernel_size=1)
                )
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            if self.dropout is not None:
                x = self.dropout(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Prepare deep supervision outputs
        deep_outputs = [] if self.use_deep_supervision else None
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse the list of skip connections
        
        for idx in range(0, len(self.ups), 2):
            # Upsampling
            x = self.ups[idx](x)
            
            # Skip connection
            skip = skip_connections[idx // 2]
            
            # Handle dimension mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            # Concatenate
            x = torch.cat((skip, x), dim=1)
            
            # Convolution block
            x = self.ups[idx + 1](x)
            
            # Deep supervision
            if self.use_deep_supervision and idx < len(self.ups) - 2:  # Not including the last layer
                deep_out = self.deep_outputs[idx // 2](x)
                deep_out = F.interpolate(deep_out, size=skip_connections[-1].shape[2:], 
                                         mode='bilinear', align_corners=True)
                deep_outputs.append(deep_out)
            
            if self.dropout is not None and idx < len(self.ups) - 2:  # Do not use dropout in the last layer
                x = self.dropout(x)
        
        # Final output
        final_output = self.final_conv(x)
        
        if self.use_deep_supervision:
            # Return outputs of all scales
            return final_output, deep_outputs
        
        return final_output

####################################
# 4. Dataset Splitting and DataLoader
####################################
def get_file_lists(root_dir):
    rep_dir = os.path.join(root_dir, "representation")
    target_dir = os.path.join(root_dir, "target")
    
    rep_files = sorted(glob.glob(os.path.join(rep_dir, "*.npy")))
    target_files = sorted(glob.glob(os.path.join(target_dir, "*.npy")))
    
    return rep_files, target_files

def split_data(rep_files, target_files, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8):
    total = len(rep_files)
    indices = np.arange(total)
    np.random.shuffle(indices)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_rep = [rep_files[i] for i in train_idx]
    train_target = [target_files[i] for i in train_idx]
    
    val_rep = [rep_files[i] for i in val_idx]
    val_target = [target_files[i] for i in val_idx]
    
    test_rep = [rep_files[i] for i in test_idx]
    test_target = [target_files[i] for i in test_idx]
    
    return (train_rep, train_target), (val_rep, val_target), (test_rep, test_target)

####################################
# 5. Define Metric Calculation and Masked Loss
####################################
def masked_mse_loss(pred, target):
    """
    Calculate MSE Loss only for non-NaN parts of the target;
    If all are NaN, return a zero loss associated with the computation graph.
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        # Return a zero loss associated with the pred computation graph
        return (pred * 0).sum()
    loss = torch.mean((pred[mask] - target[mask]) ** 2)
    return loss

def compute_metrics(pred, target):
    """
    Calculate MAE, RMSE, R2 metrics, excluding NaN values in target
    Denormalize first, then calculate metrics
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    
    # Denormalize
    pred_denorm = pred * target_std + target_mean
    target_denorm = target * target_std + target_mean

    # Calculate only for valid pixels
    pred_valid = pred_denorm[mask]
    target_valid = target_denorm[mask]
    
    mae = torch.mean(torch.abs(pred_valid - target_valid))
    rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))
    
    # Calculate R2: first flatten valid pixels
    pred_flat = pred_valid.view(-1)
    target_flat = target_valid.view(-1)
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else torch.tensor(0.0)
    
    return mae.item(), rmse.item(), r2.item()

####################################
# New: Improved prediction visualization function, selecting samples with high valid value proportion
####################################
def visualize_predictions(model, val_loader, device, epoch, save_dir, num_candidates=20):
    """
    Visualize ground truth and prediction results for the 4 samples with the highest proportion of valid values in the validation set
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Execution device
        epoch: Current training epoch
        save_dir: Directory to save visualization results
        num_candidates: Number of candidate samples to consider
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect candidate samples
    candidates = []
    with torch.no_grad():
        for rep, target in val_loader:
            # Collect only the specified number of candidate samples
            if len(candidates) < num_candidates:
                rep = rep.to(device)
                target = target.to(device)
                
                # Get model prediction
                output = model(rep)
                
                # Handle deep supervision output (if it exists)
                if isinstance(output, tuple):
                    output, _ = output
                
                # Process each sample in the batch
                for i in range(min(rep.size(0), num_candidates - len(candidates))):
                    # Calculate proportion of valid values
                    valid_proportion = (~torch.isnan(target[i])).float().mean().item()
                    
                    # Denormalize target and output
                    target_denorm = target[i].cpu() * target_std + target_mean
                    output_denorm = output[i].cpu() * target_std + target_mean
                    
                    candidates.append((target_denorm, output_denorm, valid_proportion))
                
                # If enough candidate samples have been collected, stop
                if len(candidates) == num_candidates:
                    break
            else:
                break
    
    # Sort by proportion of valid values (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Select the 4 samples with the highest proportion of valid values
    top_samples = candidates[:4]
    
    # Create visualization figure
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    
    # Plot ground truth (top row) and predictions (bottom row)
    for i, (target, pred, valid_prop) in enumerate(top_samples):
        # Extract data
        target_data = target.squeeze().numpy()
        pred_data = pred.squeeze().numpy()
        
        # Create a mask for NaN values in target
        mask = ~np.isnan(target_data)
        
        # Calculate min/max values for normalization (excluding NaN)
        if np.any(mask):
            vmin = min(np.nanmin(target_data), np.nanmin(pred_data))
            vmax = max(np.nanmax(target_data), np.nanmax(pred_data))
        else:
            vmin, vmax = 0, 100 # If all are NaN, use default values
        
        # Create normalizer to maintain consistent color mapping
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Plot ground truth
        im1 = axes[0, i].imshow(target_data, cmap='gray', norm=norm)
        # axes[0, i].set_title(f"GT (Valid: {valid_prop:.2f})")
        axes[0, i].axis('off')
        
        # Plot prediction results
        im2 = axes[1, i].imshow(pred_data, cmap='gray', norm=norm)
        # axes[1, i].set_title(f"Pred")
        axes[1, i].axis('off')
    
    # Add colorbar
    # cbar = plt.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.7)
    # cbar.set_label('Canopy Height (m)')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"val_vis_epoch_{epoch}.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved visualization results for epoch {epoch} to {save_dir}, selected samples with the highest proportion of valid values")

####################################
# 6. Training, Validation, and Testing Process (including checkpoint saving and loading)
####################################
def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3, model_name="unet"):
    # Use AdamW optimizer for better weight decay handling
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler - Cosine Annealing
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer)
    
    # Gradient clipping value
    grad_clip_value = 1.0
    
    model.to(device)
    
    # For saving the best validation loss
    best_val_loss = float('inf')
    best_val_r2 = float('-inf')  # Also track the best R²
    patience_counter = 0
    patience = 30  # Patience value for early stopping
    
    checkpoint_dir = os.path.join("checkpoints", "downstream")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_borneo_patch_{model_name}_ckpt.pth")
    
    # Create log and visualization directories
    log_dir = "/mnt/e/Codes/btfm4rs/data/downstream/borneo_patch/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        # Training phase
        for batch_idx, (rep, target) in enumerate(progress_bar):
            rep = rep.to(device)      # (B, 128, 64, 64)
            target = target.to(device) # (B, 1, 64, 64) where some pixels might be NaN
            
            optimizer.zero_grad()
            
            # Handle output with deep supervision
            output = model(rep)      # Output should be (B, 1, 64, 64) or (final_output, deep_outputs)
            
            # If using deep supervision
            if isinstance(output, tuple):
                final_output, deep_outputs = output
                
                # Main loss
                main_loss = masked_mse_loss(final_output, target)
                
                # Deep supervision loss
                deep_loss = 0
                for deep_out in deep_outputs:
                    deep_loss += masked_mse_loss(deep_out, target)
                
                # Total loss = Main loss + 0.5 * Deep supervision loss
                loss = main_loss + 0.5 * (deep_loss / len(deep_outputs))
                
                # Evaluate metrics only for the main output
                mae, rmse, r2 = compute_metrics(final_output, target)
            else:
                # Standard output
                loss = masked_mse_loss(output, target)
                mae, rmse, r2 = compute_metrics(output, target)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Update progress bar with metrics in real-time
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MAE": f"{mae:.4f}",
                "RMSE": f"{rmse:.4f}",
                "R2": f"{r2:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Validation set evaluation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0
        val_r2 = 0.0
        with torch.no_grad():
            for rep, target in val_loader:
                rep = rep.to(device)
                target = target.to(device)
                
                output = model(rep)
                
                # Handle deep supervision output
                if isinstance(output, tuple):
                    final_output, _ = output
                    loss = masked_mse_loss(final_output, target)
                    mae, rmse, r2 = compute_metrics(final_output, target)
                else:
                    loss = masked_mse_loss(output, target)
                    mae, rmse, r2 = compute_metrics(output, target)
                
                val_loss += loss.item()
                val_mae += mae
                val_rmse += rmse
                val_r2 += r2
        
        num_val = len(val_loader)
        avg_val_loss = val_loss / num_val
        avg_val_r2 = val_r2 / num_val
        
        # Print training information
        print(f"Epoch {epoch}/{epochs} - Train Loss: {np.mean(train_losses):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae/num_val:.4f} | "
              f"Val RMSE: {val_rmse/num_val:.4f} | Val R2: {avg_val_r2:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Visualize predictions every 10 epochs or on the first epoch
        if epoch % 10 == 0 or epoch == 1:
            visualize_predictions(model, val_loader, device, epoch, log_dir)
        
        # Update learning rate
        scheduler.step()
        
        # Save the best checkpoint based on validation loss
        improved = False
        
        # Use R² as the primary metric, loss as secondary
        if avg_val_r2 > best_val_r2:
            best_val_r2 = avg_val_r2
            best_val_loss = avg_val_loss
            improved = True
        elif avg_val_r2 == best_val_r2 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True
            
        if improved and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_r2': avg_val_r2
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved new best checkpoint at epoch {epoch} with Val Loss: {avg_val_loss:.4f}, Val R2: {avg_val_r2:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping check
        # if patience_counter >= patience:
        #     print(f"Early stopping triggered after {patience} epochs without improvement")
        #     break
            
def test_model(model, test_loader, device):
    """Independent test function to evaluate model performance on the test dataset"""
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_rmse = 0.0
    test_r2 = 0.0
    
    with torch.no_grad():
        for rep, target in tqdm(test_loader, desc="Testing"):
            rep = rep.to(device)
            target = target.to(device)
            
            output = model(rep)
            
            # Handle deep supervision output
            if isinstance(output, tuple):
                final_output, _ = output
                loss = masked_mse_loss(final_output, target)
                mae, rmse, r2 = compute_metrics(final_output, target)
            else:
                loss = masked_mse_loss(output, target)
                mae, rmse, r2 = compute_metrics(output, target)
                
            test_loss += loss.item()
            test_mae += mae
            test_rmse += rmse
            test_r2 += r2
            
    num_test = len(test_loader)
    print(f"Test Loss: {test_loss/num_test:.4f} | "
          f"Test MAE: {test_mae/num_test:.4f} | Test RMSE: {test_rmse/num_test:.4f} | "
          f"Test R2: {test_r2/num_test:.4f}")
    
    return test_loss/num_test, test_mae/num_test, test_rmse/num_test, test_r2/num_test
            
####################################
# 7. Main function
####################################
def main():
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Root directory for data
    root_dir = "/mnt/e/Codes/btfm4rs/data/downstream/borneo_patch"
    total_epochs = 200
    lr = 1e-3
    
    # Select model type
    model_type = "unet"  # Options: "unet", "depthwise_unet"
    
    # Configuration for Depthwise Separable Convolution UNet
    depthwise_unet_config = {
        "in_channels": 128,
        # "in_channels": 64,
        "out_channels": 1,
        "features": [64, 128, 256, 512],
        # "features": [16, 32, 64, 128],
        "use_channel_attention": True,      # Whether to use channel attention
        "use_spatial_attention": False,     # Whether to use spatial attention
        "channel_reduction": 16,            # Reduction ratio for channel attention
        "use_residual": True,               # Whether to use residual connections
        "dropout_rate": 0.1,                # Dropout probability
        "use_deep_supervision": False,      # Whether to use deep supervision (multi-scale output)
        "use_bilinear_upsample": False      # Whether to use bilinear upsampling instead of transposed convolution
    }
    
    # Display current PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Get data
    rep_files, target_files = get_file_lists(root_dir)
    
    (train_rep, train_target), (val_rep, val_target), (test_rep, test_target) = split_data(
        rep_files, target_files, train_ratio=0.3, val_ratio=0.1, test_ratio=0.6)
    
    # Create datasets and DataLoaders
    train_dataset = BorneoDataset(train_rep, train_target)
    val_dataset = BorneoDataset(val_rep, val_target)
    test_dataset = BorneoDataset(test_rep, test_target)
    
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model and device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model based on selected type
    if model_type == "unet":
        # The input data has 128 channels, so in_channels must be 128.
        model = UNet(in_channels=128, out_channels=1, features=[128, 256, 512])
    elif model_type == "depthwise_unet":
        model = DepthwiseUNet(**depthwise_unet_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Print the number of model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters for model {model_type}: {num_params}")
    
    # Train the model
    train_model(model, train_loader, val_loader, device, epochs=total_epochs, lr=lr, model_name=model_type)
    
    # Load the best checkpoint for testing
    checkpoint_path = os.path.join("checkpoints", "downstream", f"best_borneo_patch_{model_type}_ckpt.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        if 'val_r2' in checkpoint:
            print(f"Validation R2: {checkpoint['val_r2']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    else:
        print("No checkpoint found. Using current model for testing.")
    
    # Evaluate the model using the independent test function
    test_loss, test_mae, test_rmse, test_r2 = test_model(model, test_loader, device)
    

if __name__ == "__main__":
    main()