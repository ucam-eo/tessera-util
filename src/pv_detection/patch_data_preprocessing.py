import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from typing import Tuple, List, Optional, Dict
import os
import pickle
from pathlib import Path

from data_preprocessing import load_and_dequantize_representation, identify_valid_pixels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatchExtractor:
    """
    Extract 3x3 patches from representation data for CNN training.
    """

    def __init__(self, patch_size: int = 3, padding_mode: str = 'reflect'):
        """
        Initialize patch extractor.

        Args:
            patch_size: size of patches to extract (default: 3 for 3x3)
            padding_mode: padding mode for edge pixels ('reflect', 'constant', 'edge')
        """
        self.patch_size = patch_size
        self.padding_mode = padding_mode
        self.pad_width = patch_size // 2  # For 3x3 patches, pad_width = 1

    def extract_patches(self, representation: np.ndarray, labels: np.ndarray,
                       valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract patches from representation data.

        Args:
            representation: representation array (H, W, C)
            labels: labels array (H, W)
            valid_mask: valid pixels mask (H, W)

        Returns:
            patches: extracted patches (N, C, patch_size, patch_size)
            patch_labels: labels for patches (N,)
            patch_coords: coordinates of patch centers (N, 2)
        """
        logger.info(f"Extracting {self.patch_size}x{self.patch_size} patches...")

        height, width, channels = representation.shape

        # Pad the representation and valid_mask
        padded_representation = np.pad(
            representation,
            ((self.pad_width, self.pad_width), (self.pad_width, self.pad_width), (0, 0)),
            mode=self.padding_mode
        )

        padded_valid_mask = np.pad(
            valid_mask,
            ((self.pad_width, self.pad_width), (self.pad_width, self.pad_width)),
            mode='constant',
            constant_values=False
        )

        # Find valid center pixels (excluding padded regions)
        center_valid_mask = valid_mask.copy()

        # Get coordinates of valid center pixels
        valid_rows, valid_cols = np.where(center_valid_mask)
        n_valid_pixels = len(valid_rows)

        logger.info(f"Found {n_valid_pixels:,} valid center pixels")

        # Extract patches
        patches = np.zeros((n_valid_pixels, channels, self.patch_size, self.patch_size), dtype=np.float32)
        patch_labels = np.zeros(n_valid_pixels, dtype=np.int64)
        patch_coords = np.zeros((n_valid_pixels, 2), dtype=np.int32)

        for i, (row, col) in enumerate(zip(valid_rows, valid_cols)):
            # Adjust coordinates for padded array
            padded_row = row + self.pad_width
            padded_col = col + self.pad_width

            # Extract patch from padded representation
            patch = padded_representation[
                padded_row - self.pad_width:padded_row + self.pad_width + 1,
                padded_col - self.pad_width:padded_col + self.pad_width + 1,
                :
            ]

            # Check if the entire patch is valid (optional - you can remove this check)
            patch_valid_mask = padded_valid_mask[
                padded_row - self.pad_width:padded_row + self.pad_width + 1,
                padded_col - self.pad_width:padded_col + self.pad_width + 1
            ]

            # Only keep patches where center and most neighbors are valid
            if np.sum(patch_valid_mask) >= (self.patch_size * self.patch_size) // 2:
                # Convert to PyTorch format: (C, H, W)
                patches[i] = patch.transpose(2, 0, 1)
                patch_labels[i] = labels[row, col]
                patch_coords[i] = [row, col]
            else:
                # Mark invalid patches with label -1 (to be filtered out later)
                patch_labels[i] = -1

        # Filter out invalid patches
        valid_patch_mask = patch_labels != -1
        patches = patches[valid_patch_mask]
        patch_labels = patch_labels[valid_patch_mask]
        patch_coords = patch_coords[valid_patch_mask]

        logger.info(f"Extracted {len(patches):,} valid patches")

        # Print class distribution
        unique, counts = np.unique(patch_labels, return_counts=True)
        for val, count in zip(unique, counts):
            logger.info(f"  Class {val}: {count:,} patches ({100 * count / len(patch_labels):.2f}%)")

        return patches, patch_labels, patch_coords

def load_patch_data(base_dir: str, patch_size: int = 3, years: list = None, 
                   use_cache: bool = False, cache_dir: str = None, 
                   pos_neg_ratio: float = 0.33) -> dict:
    """
    Load and process all data for patch-based CNN training from multiple years.

    Args:
        base_dir: base directory containing the data files
        patch_size: size of patches to extract
        years: list of years to load data from (default: [2017])
        use_cache: if True, use cached coordinates and fast loading via memmap
        cache_dir: directory to store cache files (default: base_dir/cache)
        pos_neg_ratio: ratio of positive to negative samples for balanced sampling

    Returns:
        Dictionary containing processed patch data from all years
    """
    if years is None:
        years = [2017]  # Default to single year for backward compatibility
    
    if cache_dir is None:
        cache_dir = os.path.join(base_dir, "cache")
    
    logger.info(f"Loading patch data from years: {years}")
    logger.info(f"Use cache: {use_cache}")
    
    if use_cache:
        return _load_patch_data_with_cache(base_dir, cache_dir, years, patch_size, pos_neg_ratio)
    else:
        return _load_patch_data_full(base_dir, years, patch_size)
def _load_patch_data_full(base_dir: str, years: list, patch_size: int) -> dict:
    """
    Load patch data from multiple years (full loading mode).
    """
    # Load labels (shared across all years)
    labels_path = os.path.join(base_dir, "roi_1_clipped_gt_10m.npy")
    logger.info(f"Loading labels from {labels_path}")
    labels = np.load(labels_path)
    logger.info(f"Labels shape: {labels.shape}")

    # Initialize lists to store data from all years
    all_patches = []
    all_labels = []
    all_coords = []
    all_year_indices = []

    for year_idx, year in enumerate(years):
        logger.info(f"Processing year {year}...")
        
        # File paths for this year
        representation_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_128bands.npy")
        scales_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_scales.npy")
        
        # Check if files exist
        if not os.path.exists(representation_path):
            logger.warning(f"Representation file not found for year {year}: {representation_path}")
            continue
        if not os.path.exists(scales_path):
            logger.warning(f"Scales file not found for year {year}: {scales_path}")
            continue

        # Load representation and dequantize
        representation = load_and_dequantize_representation(representation_path, scales_path)

        # Identify valid pixels
        valid_mask = identify_valid_pixels(representation)

        # Extract patches
        patch_extractor = PatchExtractor(patch_size=patch_size)
        patches, patch_labels, patch_coords = patch_extractor.extract_patches(
            representation, labels, valid_mask
        )

        # Add year index
        year_indices = np.full(len(patches), year_idx, dtype=np.int32)

        # Append to lists
        all_patches.append(patches)
        all_labels.append(patch_labels)
        all_coords.append(patch_coords)
        all_year_indices.append(year_indices)

        logger.info(f"Year {year}: extracted {len(patches):,} patches")

    if not all_patches:
        raise ValueError("No valid data found for any year")

    # Concatenate all years
    logger.info("Concatenating data from all years...")
    combined_patches = np.concatenate(all_patches, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_coords = np.concatenate(all_coords, axis=0)
    combined_year_indices = np.concatenate(all_year_indices, axis=0)

    logger.info(f"Total patches from all years: {len(combined_patches):,}")

    # Print class distribution
    unique, counts = np.unique(combined_labels, return_counts=True)
    for val, count in zip(unique, counts):
        logger.info(f"  Class {val}: {count:,} patches ({100 * count / len(combined_labels):.2f}%)")

    return {
        'patches': combined_patches,
        'labels': combined_labels,
        'coords': combined_coords,
        'year_indices': combined_year_indices,
        'original_shape': representation.shape[:2],
        'valid_mask': valid_mask,
        'years': years
    }

def train_val_split_patches(patches: np.ndarray, labels: np.ndarray, coords: np.ndarray,
                           val_ratio: float = 0.1, random_seed: int = 42) -> Tuple:
    """
    Split patch data into training and validation sets.

    Args:
        patches: patch data (N, C, H, W)
        labels: patch labels (N,)
        coords: patch coordinates (N, 2)
        val_ratio: validation ratio
        random_seed: random seed

    Returns:
        train_patches, train_labels, train_coords, val_patches, val_labels, val_coords
    """
    np.random.seed(random_seed)

    n_samples = len(patches)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_patches = patches[train_indices]
    train_labels = labels[train_indices]
    train_coords = coords[train_indices]

    val_patches = patches[val_indices]
    val_labels = labels[val_indices]
    val_coords = coords[val_indices]

    logger.info(f"Training patches: {len(train_patches):,}")
    logger.info(f"Validation patches: {len(val_patches):,}")

    # Print class distribution for both sets
    for name, lbls in [("Training", train_labels), ("Validation", val_labels)]:
        unique, counts = np.unique(lbls, return_counts=True)
        logger.info(f"{name} class distribution:")
        for val, count in zip(unique, counts):
            logger.info(f"  Class {val}: {count:,} patches ({100 * count / len(lbls):.2f}%)")

    return train_patches, train_labels, train_coords, val_patches, val_labels, val_coords

class PVPatchDataset(Dataset):
    """
    PyTorch Dataset for PV detection patches.
    """

    def __init__(self, patches: np.ndarray, labels: np.ndarray,
                 transform=None, normalize: bool = True):
        """
        Initialize dataset.

        Args:
            patches: patch data (N, C, H, W)
            labels: patch labels (N,)
            transform: optional transforms
            normalize: whether to normalize patches
        """
        self.patches = torch.from_numpy(patches).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

        # Normalize patches if requested
        if normalize:
            self._normalize_patches()

    def _normalize_patches(self):
        """Normalize patches using global statistics."""
        # Calculate global mean and std across all patches
        # Reshape to (N*H*W, C) for calculation
        n_patches, c, h, w = self.patches.shape
        patches_flat = self.patches.view(-1, c)

        # Calculate per-channel statistics
        self.mean = patches_flat.mean(dim=0, keepdim=True)  # (1, C)
        self.std = patches_flat.std(dim=0, keepdim=True) + 1e-8  # (1, C), add small epsilon

        # Reshape for broadcasting: (1, C, 1, 1)
        self.mean = self.mean.view(1, c, 1, 1)
        self.std = self.std.view(1, c, 1, 1)

        # Normalize
        self.patches = (self.patches - self.mean) / self.std

        logger.info("Patches normalized using global statistics")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]

        if self.transform:
            patch = self.transform(patch)

        return patch, label

def create_balanced_dataset(patches: np.ndarray, labels: np.ndarray, coords: np.ndarray,
                           max_neg_pos_ratio: float = 9.0, random_seed: int = 42) -> Tuple:
    """
    Create a balanced dataset with controlled negative to positive ratio.

    Args:
        patches: patch data (N, C, H, W)
        labels: patch labels (N,)
        coords: patch coordinates (N, 2)
        max_neg_pos_ratio: maximum ratio of negative to positive samples
        random_seed: random seed

    Returns:
        balanced_patches, balanced_labels, balanced_coords
    """
    np.random.seed(random_seed)

    # Separate positive and negative samples
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_patches = patches[pos_mask]
    pos_labels = labels[pos_mask]
    pos_coords = coords[pos_mask]

    neg_patches = patches[neg_mask]
    neg_labels = labels[neg_mask]
    neg_coords = coords[neg_mask]

    n_pos = len(pos_patches)
    n_neg = len(neg_patches)

    logger.info(f"Original dataset:")
    logger.info(f"  Positive samples: {n_pos:,}")
    logger.info(f"  Negative samples: {n_neg:,}")
    logger.info(f"  Negative:Positive ratio: {n_neg/n_pos:.1f}:1")

    # Calculate target number of negative samples
    target_neg = min(int(n_pos * max_neg_pos_ratio), n_neg)

    # Sample negative examples
    if target_neg < n_neg:
        neg_indices = np.random.choice(n_neg, target_neg, replace=False)
        selected_neg_patches = neg_patches[neg_indices]
        selected_neg_labels = neg_labels[neg_indices]
        selected_neg_coords = neg_coords[neg_indices]
    else:
        selected_neg_patches = neg_patches
        selected_neg_labels = neg_labels
        selected_neg_coords = neg_coords

    # Combine positive and selected negative samples
    balanced_patches = np.concatenate([pos_patches, selected_neg_patches], axis=0)
    balanced_labels = np.concatenate([pos_labels, selected_neg_labels], axis=0)
    balanced_coords = np.concatenate([pos_coords, selected_neg_coords], axis=0)

    # Shuffle the combined dataset
    indices = np.arange(len(balanced_patches))
    np.random.shuffle(indices)

    balanced_patches = balanced_patches[indices]
    balanced_labels = balanced_labels[indices]
    balanced_coords = balanced_coords[indices]

    # Print final statistics
    final_pos = np.sum(balanced_labels == 1)
    final_neg = np.sum(balanced_labels == 0)

    logger.info(f"Balanced dataset:")
    logger.info(f"  Positive samples: {final_pos:,}")
    logger.info(f"  Negative samples: {final_neg:,}")
    logger.info(f"  Negative:Positive ratio: {final_neg/final_pos:.1f}:1")
    logger.info(f"  Total samples: {len(balanced_patches):,}")

    return balanced_patches, balanced_labels, balanced_coords

def _get_patch_cache_paths(cache_dir: str, base_dir: str) -> Dict[str, Path]:
    """
    Get cache file paths for patch data.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(base_dir).name
    
    return {
        'pos_coords': cache_dir / f"{base_name}_pos_coords.npy",
        'neg_coords_sample': cache_dir / f"{base_name}_neg_coords_sample.npy", 
        'valid_coords': cache_dir / f"{base_name}_valid_coords.npy",
        'cache_info': cache_dir / f"{base_name}_cache_info.pkl"
    }

def _create_patch_coordinate_cache(base_dir: str, cache_dir: str, years: list, 
                                 patch_size: int, pos_neg_ratio: float):
    """
    Create coordinate cache for patch data.
    """
    logger.info("Creating patch coordinate cache...")
    
    cache_paths = _get_patch_cache_paths(cache_dir, base_dir)
    
    # Load labels (shared across all years)
    labels_path = os.path.join(base_dir, "roi_1_clipped_gt_10m.npy")
    labels = np.load(labels_path)
    
    # Get valid coordinates from the first available year
    representation_path = None
    scales_path = None
    for year in years:
        rep_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_128bands.npy")
        sc_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_scales.npy")
        if os.path.exists(rep_path) and os.path.exists(sc_path):
            representation_path = rep_path
            scales_path = sc_path
            break
    
    if representation_path is None:
        raise ValueError("No valid representation files found for any year")
    
    # Load representation to get valid mask (using memmap for memory efficiency)
    representation_int8 = np.load(representation_path, mmap_mode='r')
    scales = np.load(scales_path)
    
    # Identify valid pixels without loading full representation
    valid_mask = np.any(representation_int8 != 0, axis=2)
    
    # Get valid coordinates for patch extraction
    pad_width = patch_size // 2
    h, w = valid_mask.shape
    
    # Find coordinates where we can extract valid patches
    valid_patch_coords = []
    for row in range(pad_width, h - pad_width):
        for col in range(pad_width, w - pad_width):
            if valid_mask[row, col]:  # Center pixel is valid
                # Check if patch area has enough valid pixels
                patch_valid = valid_mask[row-pad_width:row+pad_width+1, col-pad_width:col+pad_width+1]
                if np.sum(patch_valid) >= (patch_size * patch_size) // 2:
                    valid_patch_coords.append([row, col])
    
    valid_patch_coords = np.array(valid_patch_coords)
    logger.info(f"Found {len(valid_patch_coords):,} valid patch coordinates")
    
    # Separate positive and negative coordinates
    pos_coords = []
    neg_coords = []
    
    for coord in valid_patch_coords:
        row, col = coord
        if labels[row, col] == 1:
            pos_coords.append(coord)
        else:
            neg_coords.append(coord)
    
    pos_coords = np.array(pos_coords) if pos_coords else np.empty((0, 2), dtype=np.int32)
    neg_coords = np.array(neg_coords) if neg_coords else np.empty((0, 2), dtype=np.int32)
    
    logger.info(f"Positive patch coordinates: {len(pos_coords):,}")
    logger.info(f"Negative patch coordinates: {len(neg_coords):,}")
    
    # Pre-sample negative coordinates based on pos_neg_ratio
    n_pos = len(pos_coords)
    if n_pos > 0 and len(neg_coords) > 0:
        target_neg = int(n_pos / pos_neg_ratio)
        target_neg = min(target_neg, len(neg_coords))
        
        np.random.seed(42)  # Fixed seed for reproducible sampling
        neg_indices = np.random.choice(len(neg_coords), target_neg, replace=False)
        neg_coords_sample = neg_coords[neg_indices]
    else:
        neg_coords_sample = neg_coords
    
    logger.info(f"Pre-sampled negative coordinates: {len(neg_coords_sample):,}")
    
    # Save coordinate cache
    np.save(cache_paths['pos_coords'], pos_coords)
    np.save(cache_paths['neg_coords_sample'], neg_coords_sample)
    np.save(cache_paths['valid_coords'], valid_patch_coords)
    
    # Save cache info
    cache_info = {
        'years': years,
        'patch_size': patch_size,
        'pos_neg_ratio': pos_neg_ratio,
        'n_pos': len(pos_coords),
        'n_neg_sample': len(neg_coords_sample),
        'n_valid': len(valid_patch_coords)
    }
    
    with open(cache_paths['cache_info'], 'wb') as f:
        pickle.dump(cache_info, f)
    
    logger.info("Patch coordinate cache created successfully!")
    logger.info(f"  - Cached {len(pos_coords):,} positive coordinates")
    logger.info(f"  - Cached {len(neg_coords_sample):,} negative coordinates")

def _load_patch_data_with_cache(base_dir: str, cache_dir: str, years: list, 
                               patch_size: int, pos_neg_ratio: float) -> dict:
    """
    Load patch data using cached coordinates and memmap for memory efficiency.
    """
    cache_paths = _get_patch_cache_paths(cache_dir, base_dir)
    
    # Check if cache exists and is valid
    if not all(path.exists() for path in cache_paths.values()):
        logger.info("Patch cache not found, creating cache...")
        _create_patch_coordinate_cache(base_dir, cache_dir, years, patch_size, pos_neg_ratio)
    else:
        # Load cache info and validate
        with open(cache_paths['cache_info'], 'rb') as f:
            cache_info = pickle.load(f)
        
        if (cache_info['years'] != years or 
            cache_info['patch_size'] != patch_size or 
            cache_info['pos_neg_ratio'] != pos_neg_ratio):
            logger.info("Patch cache is outdated, recreating...")
            _create_patch_coordinate_cache(base_dir, cache_dir, years, patch_size, pos_neg_ratio)
    
    # Load cached coordinates
    logger.info("Loading patch data with coordinate cache...")
    
    pos_coords = np.load(cache_paths['pos_coords'])
    neg_coords_sample = np.load(cache_paths['neg_coords_sample'])
    
    logger.info(f"Loaded {len(pos_coords):,} positive coordinates")
    logger.info(f"Loaded {len(neg_coords_sample):,} negative coordinates")
    
    # Combine coordinates and create labels
    all_coords = np.concatenate([pos_coords, neg_coords_sample], axis=0)
    all_labels = np.concatenate([
        np.ones(len(pos_coords), dtype=np.int64),
        np.zeros(len(neg_coords_sample), dtype=np.int64)
    ])
    
    # Load labels (shared across all years)
    labels_path = os.path.join(base_dir, "roi_1_clipped_gt_10m.npy")
    labels = np.load(labels_path)
    
    # Extract patches from all years using memmap
    logger.info("Extracting patches from all years using memmap...")
    all_patches = []
    all_year_indices = []
    
    for year_idx, year in enumerate(years):
        logger.info(f"Processing patches for year {year}...")
        
        # File paths for this year
        representation_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_128bands.npy")
        scales_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_scales.npy")
        
        if not os.path.exists(representation_path) or not os.path.exists(scales_path):
            logger.warning(f"Files not found for year {year}, skipping...")
            continue
        
        # Load representation using memmap for memory efficiency
        representation_int8 = np.load(representation_path, mmap_mode='r')
        scales = np.load(scales_path)
        
        # Extract patches for this year
        year_patches = []
        pad_width = patch_size // 2
        
        for coord in all_coords:
            row, col = coord
            
            # Extract patch from memmap array
            patch_int8 = representation_int8[
                max(0, row - pad_width):min(representation_int8.shape[0], row + pad_width + 1),
                max(0, col - pad_width):min(representation_int8.shape[1], col + pad_width + 1),
                :
            ]
            
            # Extract corresponding scales patch
            scales_patch = scales[
                max(0, row - pad_width):min(scales.shape[0], row + pad_width + 1),
                max(0, col - pad_width):min(scales.shape[1], col + pad_width + 1)
            ]
            
            # Handle edge cases by padding if necessary
            if patch_int8.shape[0] != patch_size or patch_int8.shape[1] != patch_size:
                padded_patch = np.zeros((patch_size, patch_size, patch_int8.shape[2]), dtype=patch_int8.dtype)
                padded_scales = np.zeros((patch_size, patch_size), dtype=scales.dtype)
                
                # Calculate padding offsets
                start_row = max(0, pad_width - row)
                start_col = max(0, pad_width - col)
                end_row = start_row + patch_int8.shape[0]
                end_col = start_col + patch_int8.shape[1]
                
                padded_patch[start_row:end_row, start_col:end_col, :] = patch_int8
                padded_scales[start_row:end_row, start_col:end_col] = scales_patch
                
                patch_int8 = padded_patch
                scales_patch = padded_scales
            
            # Dequantize patch using corresponding scales
            # Expand scales to match patch dimensions: (H, W) -> (H, W, 1)
            scales_expanded = scales_patch[..., np.newaxis]
            patch_float = patch_int8.astype(np.float32) * scales_expanded
            
            # Convert to PyTorch format: (C, H, W)
            patch = patch_float.transpose(2, 0, 1)
            year_patches.append(patch)
        
        year_patches = np.array(year_patches, dtype=np.float32)
        all_patches.append(year_patches)
        
        # Add year indices
        year_indices = np.full(len(all_coords), year_idx, dtype=np.int32)
        all_year_indices.append(year_indices)
        
        logger.info(f"Year {year}: extracted {len(year_patches):,} patches")
    
    if not all_patches:
        raise ValueError("No valid data found for any year")
    
    # Concatenate all years
    logger.info("Concatenating patches from all years...")
    combined_patches = np.concatenate(all_patches, axis=0)
    combined_year_indices = np.concatenate(all_year_indices, axis=0)
    
    # Replicate labels and coords for each year
    n_years = len(all_patches)
    final_labels = np.tile(all_labels, n_years)
    final_coords = np.tile(all_coords, (n_years, 1))
    
    logger.info(f"Total patches from all years: {len(combined_patches):,}")
    logger.info(f"Positive samples: {np.sum(final_labels):,}")
    logger.info(f"Negative samples: {len(final_labels) - np.sum(final_labels):,}")
    
    # Get original shape from first available year for compatibility
    representation_path = None
    for year in years:
        rep_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_128bands.npy")
        if os.path.exists(rep_path):
            representation_path = rep_path
            break
    
    if representation_path:
        # Load just to get shape info (minimal memory usage)
        temp_repr = np.load(representation_path, mmap_mode='r')
        original_shape = temp_repr.shape[:2]
        del temp_repr
    else:
        original_shape = (5643, 5565)  # Default fallback
    
    return {
        'patches': combined_patches,
        'labels': final_labels,
        'coords': final_coords,
        'year_indices': combined_year_indices,
        'original_shape': original_shape,
        'years': years
    }

if __name__ == "__main__":
    # Test patch extraction
    base_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1"

    logger.info("Testing patch extraction...")
    data = load_patch_data(base_dir, patch_size=3)

    print(f"Patches shape: {data['patches'].shape}")
    print(f"Labels shape: {data['labels'].shape}")
    print(f"Coordinates shape: {data['coords'].shape}")

    # Test train/val split
    train_patches, train_labels, train_coords, val_patches, val_labels, val_coords = train_val_split_patches(
        data['patches'], data['labels'], data['coords']
    )

    # Test dataset creation
    train_dataset = PVPatchDataset(train_patches, train_labels)
    val_dataset = PVPatchDataset(val_patches, val_labels)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Test sample
    sample_patch, sample_label = train_dataset[0]
    print(f"Sample patch shape: {sample_patch.shape}")
    print(f"Sample label: {sample_label}")

    logger.info("Patch extraction test completed successfully!")