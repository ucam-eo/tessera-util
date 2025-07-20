import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import logging
from datetime import datetime
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# (Optional) FocalLoss - adapted for segmentation
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # input: (B, C, H, W), target: (B, H, W)
        logpt = F.log_softmax(input, dim=1)
        ce_loss = F.nll_loss(logpt, target, weight=self.weight, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------------------------------------------------------------------
# NEW Dataset class for Pixel-wise Segmentation
# -----------------------------------------------------------------------------
class PatchSegmentationDataset(Dataset):
    def __init__(self, patch_rep_files, patch_label_files, label_map, ignore_label=0):
        """
        Args:
            patch_rep_files (list): List of file paths to representation patches (P, P, C)
            patch_label_files (list): List of file paths to label patches (P, P)
            label_map (dict): Dictionary to map original label values to 0-N indices
            ignore_label (int): Label value to ignore in loss calculation (default: 0 for background)
        """
        self.patch_rep_files = []
        self.patch_label_files = []
        self.label_map = label_map
        self.ignore_label = ignore_label
        self.ignore_index = -100  # PyTorch standard ignore index

        logging.info(f"Processing {len(patch_rep_files)} patch files for segmentation...")
        
        for rep_file, label_file in tqdm(zip(patch_rep_files, patch_label_files), total=len(patch_rep_files)):
            try:
                label_patch_original = np.load(label_file)  # Shape (P, P)
                
                # Check if patch has any valid labels (non-zero)
                valid_labels_in_patch = label_patch_original[label_patch_original != ignore_label]
                
                if valid_labels_in_patch.size == 0:
                    continue  # Skip patches with no valid labels
                
                self.patch_rep_files.append(rep_file)
                self.patch_label_files.append(label_file)
                
            except Exception as e:
                logging.error(f"Error processing patch {rep_file} or {label_file}: {e}")
                continue
                
        logging.info(f"Kept {len(self.patch_rep_files)} patches with valid labels for segmentation.")

    def __len__(self):
        return len(self.patch_rep_files)

    def __getitem__(self, idx):
        rep_patch = np.load(self.patch_rep_files[idx])  # Shape (P, P, C)
        label_patch = np.load(self.patch_label_files[idx])  # Shape (P, P)
        
        # Convert representation to (C, P, P) for PyTorch CNNs
        rep_patch_tensor = torch.tensor(rep_patch, dtype=torch.float32).permute(2, 0, 1)
        
        # Map labels and handle ignore_label
        label_patch_mapped = np.full_like(label_patch, self.ignore_index, dtype=np.int64)
        
        for original_label, mapped_label in self.label_map.items():
            mask = (label_patch == original_label)
            label_patch_mapped[mask] = mapped_label
        
        # Background (ignore_label) remains as ignore_index for loss calculation
        label_patch_tensor = torch.tensor(label_patch_mapped, dtype=torch.long)
        
        return rep_patch_tensor, label_patch_tensor

# -----------------------------------------------------------------------------
# UNet Models adapted for Segmentation (remove Global Average Pooling)
# -----------------------------------------------------------------------------
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
        :param in_channels: Input channels
        :param out_channels: Output channels (num_classes for segmentation)
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
        
        # Final 1x1 convolution for segmentation
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
            
        # Return segmentation map: (B, num_classes, H, W)
        return self.final_conv(x)

# Depthwise Separable Convolution Components (same as before)
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
    """UNet using Depthwise Separable Convolution for Segmentation"""
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
        :param out_channels: Output channels (num_classes for segmentation)
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
                # For segmentation, upsample to original size
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

# -----------------------------------------------------------------------------
# Simple CNN adapted for Segmentation
# -----------------------------------------------------------------------------
class PatchSegmenter(nn.Module):
    def __init__(self, input_channels, patch_size, num_classes):
        super(PatchSegmenter, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        
        # Final classification layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):  # x shape: (B, C, H, W)
        # Encoder
        x1 = self.relu1(self.conv1(x))
        x2 = self.pool1(x1)
        
        x3 = self.relu2(self.conv2(x2))
        x4 = self.pool2(x3)
        
        x5 = self.relu3(self.conv3(x4))
        
        # Decoder
        x6 = self.upconv2(x5)
        x7 = self.relu4(self.conv4(x6))
        
        x8 = self.upconv1(x7)
        x9 = self.relu5(self.conv5(x8))
        
        # Final segmentation output
        output = self.final_conv(x9)  # (B, num_classes, H, W)
        
        return output

# -----------------------------------------------------------------------------
# Pixel-wise metrics calculation
# -----------------------------------------------------------------------------
def compute_pixel_metrics(pred, target, ignore_index=-100):
    """
    Compute pixel-wise accuracy and F1 score
    Args:
        pred: (B, C, H, W) - predictions
        target: (B, H, W) - ground truth labels
        ignore_index: index to ignore in calculation
    """
    pred_labels = pred.argmax(dim=1)  # (B, H, W)
    
    # Create mask for valid pixels (not ignore_index)
    valid_mask = (target != ignore_index)
    
    if valid_mask.sum() == 0:
        return 0.0, 0.0
    
    pred_valid = pred_labels[valid_mask]
    target_valid = target[valid_mask]
    
    # Convert to numpy for sklearn metrics
    pred_np = pred_valid.cpu().numpy()
    target_np = target_valid.cpu().numpy()
    
    accuracy = accuracy_score(target_np, pred_np)
    f1 = f1_score(target_np, pred_np, average='weighted', zero_division=0)
    
    return accuracy, f1

# -----------------------------------------------------------------------------
# Main function for Pixel-wise Segmentation
# -----------------------------------------------------------------------------
def main_patch_segmentation():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # -------------------------
    # Model Selection Parameter
    # -------------------------
    model_type = "unet"  # Options: "unet", "depthwise_unet", "simple_cnn"
    
    # Model configurations
    unet_config = {
        "features": [128, 256, 512]
    }
    
    depthwise_unet_config = {
        "features": [64, 128, 256, 512],
        "use_channel_attention": True,
        "use_spatial_attention": False,
        "channel_reduction": 16,
        "use_residual": True,
        "dropout_rate": 0.1,
        "use_deep_supervision": False,
        "use_bilinear_upsample": False
    }

    # -------------------------
    # Define PATCH file paths
    # -------------------------
    base_patch_dir = "data/downstream/austrian_crop_patch_split"  # ADJUST THIS
    train_rep_files = sorted([os.path.join(base_patch_dir, "train", f) for f in os.listdir(os.path.join(base_patch_dir, "train")) if f.startswith("patch_") and f.endswith(".npy")])
    train_label_files = sorted([os.path.join(base_patch_dir, "train", f) for f in os.listdir(os.path.join(base_patch_dir, "train")) if f.startswith("label_") and f.endswith(".npy")])

    test_rep_files = sorted([os.path.join(base_patch_dir, "test", f) for f in os.listdir(os.path.join(base_patch_dir, "test")) if f.startswith("patch_") and f.endswith(".npy")])
    test_label_files = sorted([os.path.join(base_patch_dir, "test", f) for f in os.listdir(os.path.join(base_patch_dir, "test")) if f.startswith("label_") and f.endswith(".npy")])

    # Validation split from test set
    if len(test_rep_files) > 100:
        val_split_idx = int(len(test_rep_files) * 1 / 7)
        val_rep_files = test_rep_files[:val_split_idx]
        val_label_files = test_label_files[:val_split_idx]
        test_rep_files = test_rep_files[val_split_idx:]
        test_label_files = test_label_files[val_split_idx:]
    else:
        logging.warning("Test set is too small for a validation split, using test set for validation metrics.")
        val_rep_files = test_rep_files
        val_label_files = test_label_files

    # -------------------------
    # Determine number of classes and label mapping
    # -------------------------
    all_original_labels_in_patches = set()
    sample_label_files_for_meta = train_label_files[:min(100, len(train_label_files))]
    for f in sample_label_files_for_meta:
        patch_l = np.load(f)
        all_original_labels_in_patches.update(np.unique(patch_l))

    unique_original_nonzero_labels = sorted([l for l in all_original_labels_in_patches if l != 0])
    if not unique_original_nonzero_labels:
        raise ValueError("No non-zero labels found in sample patches. Cannot determine classes.")

    # Create a mapping from original non-zero labels to 0-N indices
    label_map_for_dataset = {label: i for i, label in enumerate(unique_original_nonzero_labels)}
    num_classes = len(unique_original_nonzero_labels)

    logging.info(f"Original non-zero labels found: {unique_original_nonzero_labels}")
    logging.info(f"Label map for segmentation: {label_map_for_dataset}")
    logging.info(f"Number of classes for pixel-wise segmentation: {num_classes}")

    # -------------------------
    # Create Datasets and DataLoaders for Segmentation
    # -------------------------
    if not train_rep_files:
        raise ValueError("No training representation files found. Check paths and patch generation.")
    sample_rep_patch = np.load(train_rep_files[0])
    patch_p, _, patch_c = sample_rep_patch.shape  # Assuming (P,P,C)
    patch_size_p = patch_p  # P from (P,P,C)
    input_channels_c = patch_c  # C from (P,P,C)
    logging.info(f"Detected patch properties: Size P={patch_size_p}, Channels C={input_channels_c}")

    train_dataset = PatchSegmentationDataset(train_rep_files, train_label_files, label_map_for_dataset, ignore_label=0)
    val_dataset = PatchSegmentationDataset(val_rep_files, val_label_files, label_map_for_dataset, ignore_label=0)
    test_dataset = PatchSegmentationDataset(test_rep_files, test_label_files, label_map_for_dataset, ignore_label=0)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check patch processing and label mapping.")

    batch_size = 16  # Smaller batch size for segmentation (higher memory usage)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # -------------------------
    # Model Selection for Segmentation
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    if model_type == "unet":
        model = UNet(
            in_channels=input_channels_c, 
            out_channels=num_classes, 
            **unet_config
        ).to(device)
        logging.info(f"Using UNet model for segmentation with {sum(p.numel() for p in model.parameters())} parameters")
        
    elif model_type == "depthwise_unet":
        model = DepthwiseUNet(
            in_channels=input_channels_c,
            out_channels=num_classes,
            **depthwise_unet_config
        ).to(device)
        logging.info(f"Using DepthwiseUNet model for segmentation with {sum(p.numel() for p in model.parameters())} parameters")
        
    elif model_type == "simple_cnn":
        model = PatchSegmenter(
            input_channels=input_channels_c, 
            patch_size=patch_size_p, 
            num_classes=num_classes
        ).to(device)
        logging.info(f"Using Simple CNN model for segmentation with {sum(p.numel() for p in model.parameters())} parameters")
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Use CrossEntropyLoss with ignore_index for background pixels
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    
    # Adjust learning rate based on model complexity
    lr = 0.0001 if model_type in ["unet", "depthwise_unet"] else 0.001
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # -------------------------
    # Training Config & Checkpoints
    # -------------------------
    num_epochs = 200
    log_interval = max(1, len(train_loader) // 10)
    best_val_f1 = 0.0
    best_val_accuracy = 0.0
    best_epoch = 0
    checkpoint_save_folder = f"checkpoints/patch_segmenter_{model_type}_{timestamp}/"
    os.makedirs(checkpoint_save_folder, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_save_folder, "best_model.pt")

    logging.info(f"Starting pixel-wise segmentation training for {num_epochs} epochs with {model_type} model.")
    
    # -------------------------
    # Training and Validation Loop for Segmentation
    # -------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_pixel_accuracy = 0.0
        epoch_pixel_f1 = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)
        for step, (rep_patches, label_patches) in enumerate(progress_bar):
            rep_patches = rep_patches.to(device)  # (B, C, H, W)
            label_patches = label_patches.to(device)  # (B, H, W)

            optimizer.zero_grad()
            outputs = model(rep_patches)  # (B, num_classes, H, W)
            
            # Handle deep supervision output if exists
            if isinstance(outputs, tuple):
                final_output, deep_outputs = outputs
                # Main loss
                main_loss = criterion(final_output, label_patches)
                # Deep supervision loss
                deep_loss = 0
                for deep_out in deep_outputs:
                    deep_loss += criterion(deep_out, label_patches)
                # Total loss
                loss = main_loss + 0.5 * (deep_loss / len(deep_outputs))
                outputs = final_output  # Use final output for metrics
            else:
                loss = criterion(outputs, label_patches)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            
            # Compute pixel-wise metrics
            acc, f1 = compute_pixel_metrics(outputs, label_patches)
            epoch_pixel_accuracy += acc
            epoch_pixel_f1 += f1
            num_batches += 1

            if (step + 1) % log_interval == 0 or (step + 1) == len(train_loader):
                current_loss = running_loss / (step % log_interval + 1)
                current_acc = epoch_pixel_accuracy / num_batches
                current_f1 = epoch_pixel_f1 / num_batches
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}', 
                    'acc': f'{current_acc:.4f}',
                    'f1': f'{current_f1:.4f}',
                    'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
                })

        avg_epoch_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = epoch_pixel_accuracy / num_batches
        avg_train_f1 = epoch_pixel_f1 / num_batches
        logging.info(f"Epoch {epoch+1} Train: Loss={avg_epoch_train_loss:.4f}, Pixel_Acc={avg_train_accuracy:.4f}, Pixel_F1={avg_train_f1:.4f}")

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_pixel_accuracy = 0.0
        val_pixel_f1 = 0.0
        val_num_batches = 0
        
        with torch.no_grad():
            for rep_patches, label_patches in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False):
                rep_patches = rep_patches.to(device)
                label_patches = label_patches.to(device)
                outputs = model(rep_patches)
                
                # Handle deep supervision output if exists
                if isinstance(outputs, tuple):
                    final_output, _ = outputs
                    loss = criterion(final_output, label_patches)
                    outputs = final_output
                else:
                    loss = criterion(outputs, label_patches)
                    
                val_running_loss += loss.item()
                
                # Compute pixel-wise metrics
                acc, f1 = compute_pixel_metrics(outputs, label_patches)
                val_pixel_accuracy += acc
                val_pixel_f1 += f1
                val_num_batches += 1

        avg_epoch_val_loss = val_running_loss / len(val_loader)
        avg_val_accuracy = val_pixel_accuracy / val_num_batches
        avg_val_f1 = val_pixel_f1 / val_num_batches
        logging.info(f"Epoch {epoch+1} VAL: Loss={avg_epoch_val_loss:.4f}, Pixel_Acc={avg_val_accuracy:.4f}, Pixel_F1={avg_val_f1:.4f}")

        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_val_accuracy = avg_val_accuracy
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_accuracy': best_val_accuracy,
                'label_map': label_map_for_dataset,
                'patch_size_p': patch_size_p,
                'input_channels_c': input_channels_c,
                'num_classes': num_classes,
                'model_type': model_type,
                'model_config': unet_config if model_type == "unet" else depthwise_unet_config if model_type == "depthwise_unet" else {}
            }, best_checkpoint_path)
            logging.info(f"🎉 Best Checkpoint Saved! Epoch {best_epoch}, Val Pixel_F1: {best_val_f1:.4f}, Val Pixel_Acc: {best_val_accuracy:.4f}")

    logging.info(f"Training completed. Best Val Pixel_F1: {best_val_f1:.4f}, Best Val Pixel_Acc: {best_val_accuracy:.4f} at Epoch {best_epoch}")

    # -------------------------
    # Final Test Evaluation for Segmentation
    # -------------------------
    if os.path.exists(best_checkpoint_path):
        logging.info(f"Loading best checkpoint from: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Best model from epoch {checkpoint.get('epoch', 'N/A')} loaded.")
    else:
        logging.warning("No best checkpoint found to load for final testing.")

    model.eval()
    test_running_loss = 0.0
    test_pixel_accuracy = 0.0
    test_pixel_f1 = 0.0
    test_num_batches = 0
    
    # Collect all predictions and targets for detailed classification report
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for rep_patches, label_patches in tqdm(test_loader, desc="Final Test Evaluation", leave=False):
            rep_patches = rep_patches.to(device)
            label_patches = label_patches.to(device)
            outputs = model(rep_patches)
            
            # Handle deep supervision output if exists
            if isinstance(outputs, tuple):
                final_output, _ = outputs
                loss = criterion(final_output, label_patches)
                outputs = final_output
            else:
                loss = criterion(outputs, label_patches)
                
            test_running_loss += loss.item()
            
            # Compute pixel-wise metrics
            acc, f1 = compute_pixel_metrics(outputs, label_patches)
            test_pixel_accuracy += acc
            test_pixel_f1 += f1
            test_num_batches += 1
            
            # Collect predictions and targets for classification report
            pred_labels = outputs.argmax(dim=1)  # (B, H, W)
            valid_mask = (label_patches != -100)
            
            if valid_mask.sum() > 0:
                all_test_preds.extend(pred_labels[valid_mask].cpu().numpy())
                all_test_targets.extend(label_patches[valid_mask].cpu().numpy())

    avg_test_loss = test_running_loss / len(test_loader) if len(test_loader) > 0 else 0
    avg_test_accuracy = test_pixel_accuracy / test_num_batches
    avg_test_f1 = test_pixel_f1 / test_num_batches

    total_pixels = len(all_test_targets)
    logging.info(f"FINAL TEST: Loss={avg_test_loss:.4f}, Pixel_Acc={avg_test_accuracy:.4f}, Pixel_F1={avg_test_f1:.4f}")
    logging.info(f"Total pixels evaluated: {total_pixels}")

    # Generate and log classification report for test set (pixel-wise)
    idx_to_original_label = {v: k for k, v in label_map_for_dataset.items()}
    target_names_for_report = [str(idx_to_original_label.get(i, f"Unknown_{i}")) for i in range(num_classes)]

    test_class_report = classification_report(all_test_targets, all_test_preds,
                                              target_names=target_names_for_report,
                                              digits=4, zero_division=0)
    logging.info(f"\nFinal Test Classification Report (Pixel-wise):\n{test_class_report}")
    print(f"\nFinal Test Classification Report (Pixel-wise):\n{test_class_report}")
    print(f"Total pixels in test set: {total_pixels}")

if __name__ == "__main__":
    # Call the segmentation main function
    main_patch_segmentation()