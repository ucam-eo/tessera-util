import numpy as np
import os
from typing import Tuple, Dict
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_dequantize_representation(representation_file_path: str, scales_file_path: str) -> np.ndarray:
    """
    Load and dequantize int8 representations back to float32.

    Args:
        representation_file_path: Path to the int8 representation file (H,W,C)
        scales_file_path: Path to the float32 scales file (H,W)

    Returns:
        representation_f32: float32 ndarray of shape (H,W,C)
    """
    logger.info(f"Loading representation from {representation_file_path}")
    logger.info(f"Loading scales from {scales_file_path}")

    # Load the files
    representation_int8 = np.load(representation_file_path)  # (H, W, C), dtype=int8
    scales = np.load(scales_file_path)  # (H, W), dtype=float32

    logger.info(f"Representation shape: {representation_int8.shape}")
    logger.info(f"Scales shape: {scales.shape}")

    # Convert int8 to float32 for computation
    representation_f32 = representation_int8.astype(np.float32)

    # Expand scales to match representation shape
    # scales shape: (H, W) -> (H, W, 1)
    scales_expanded = scales[..., np.newaxis]

    # Dequantize by multiplying with scales
    representation_f32 = representation_f32 * scales_expanded

    return representation_f32

def identify_valid_pixels(representation: np.ndarray) -> np.ndarray:
    """
    Identify valid pixels (non-zero embeddings).

    Args:
        representation: numpy array of shape (H, W, C)

    Returns:
        valid_mask: boolean array of shape (H, W) where True indicates valid pixels
    """
    # Check if all channels are zero for each pixel
    valid_mask = ~np.all(representation == 0, axis=2)

    num_valid = np.sum(valid_mask)
    total_pixels = representation.shape[0] * representation.shape[1]

    logger.info(f"Valid pixels: {num_valid:,} / {total_pixels:,} ({100 * num_valid / total_pixels:.2f}%)")

    return valid_mask

def extract_valid_data(representation: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract valid pixels from representation and labels.

    Args:
        representation: numpy array of shape (H, W, C)
        labels: numpy array of shape (H, W)
        valid_mask: boolean array of shape (H, W)

    Returns:
        valid_features: array of shape (N_valid, C)
        valid_labels: array of shape (N_valid,)
        valid_coords: array of shape (N_valid, 2) containing (row, col) coordinates
    """
    # Get coordinates of valid pixels
    valid_coords = np.column_stack(np.where(valid_mask))

    # Extract features and labels for valid pixels
    valid_features = representation[valid_mask]  # Shape: (N_valid, C)
    valid_labels = labels[valid_mask]  # Shape: (N_valid,)

    logger.info(f"Extracted features shape: {valid_features.shape}")
    logger.info(f"Extracted labels shape: {valid_labels.shape}")

    # Print class distribution
    unique, counts = np.unique(valid_labels, return_counts=True)
    for val, count in zip(unique, counts):
        logger.info(f"Class {val}: {count:,} samples ({100 * count / len(valid_labels):.2f}%)")

    return valid_features, valid_labels, valid_coords

def train_val_split(features: np.ndarray, labels: np.ndarray, coords: np.ndarray, year_indices: np.ndarray = None,
                   val_ratio: float = 0.1, random_seed: int = 42, max_positive_samples: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split multi-year data into training and validation sets.
    If max_positive_samples is specified and the number of positive samples exceeds it,
    only max_positive_samples positive samples will be used for training, and the rest
    will be moved to the validation set.

    Args:
        features: array of shape (N, C)
        labels: array of shape (N,)
        coords: array of shape (N, 2)
        year_indices: array of shape (N,) indicating which year each sample comes from
        val_ratio: fraction of data to use for validation (ignored if max_positive_samples is used)
        random_seed: random seed for reproducibility
        max_positive_samples: maximum number of positive samples to use for training

    Returns:
        train_features, train_labels, train_coords, val_features, val_labels, val_coords, train_years, val_years
    """
    np.random.seed(random_seed)

    n_samples = len(features)
    
    # If year_indices is None, create dummy indices (for backward compatibility)
    if year_indices is None:
        year_indices = np.zeros(n_samples, dtype=int)
    
    # Check if we need to apply positive sample limit
    if max_positive_samples is not None:
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        positive_indices = np.where(positive_mask)[0]
        negative_indices = np.where(negative_mask)[0]
        
        n_positive = len(positive_indices)
        n_negative = len(negative_indices)
        
        logger.info(f"Total positive samples: {n_positive:,}")
        logger.info(f"Total negative samples: {n_negative:,}")
        
        # Log year distribution for positive samples
        if year_indices is not None and len(np.unique(year_indices)) > 1:
            positive_years = year_indices[positive_indices]
            unique_years, year_counts = np.unique(positive_years, return_counts=True)
            logger.info("Positive samples by year:")
            for year, count in zip(unique_years, year_counts):
                logger.info(f"  Year {year}: {count:,} positive samples")
        
        if n_positive > max_positive_samples:
            logger.info(f"Limiting positive samples to {max_positive_samples:,} for training")
            
            # Randomly shuffle positive indices
            np.random.shuffle(positive_indices)
            
            # Split positive samples
            train_positive_indices = positive_indices[:max_positive_samples]
            val_positive_indices = positive_indices[max_positive_samples:]
            
            # For negative samples, use normal train/val split
            np.random.shuffle(negative_indices)
            n_val_negative = int(n_negative * val_ratio)
            val_negative_indices = negative_indices[:n_val_negative]
            train_negative_indices = negative_indices[n_val_negative:]
            
            # Combine indices
            train_indices = np.concatenate([train_positive_indices, train_negative_indices])
            val_indices = np.concatenate([val_positive_indices, val_negative_indices])
            
        else:
            logger.info(f"Positive samples ({n_positive:,}) within limit, using normal split")
            # Use normal split
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            n_val = int(n_samples * val_ratio)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
    else:
        # Normal split without positive sample limit
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        n_val = int(n_samples * val_ratio)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    train_coords = coords[train_indices]
    train_years = year_indices[train_indices]

    val_features = features[val_indices]
    val_labels = labels[val_indices]
    val_coords = coords[val_indices]
    val_years = year_indices[val_indices]

    logger.info(f"Training set: {len(train_features):,} samples")
    logger.info(f"Validation set: {len(val_features):,} samples")

    # Print class distribution for both sets
    for name, lbls in [("Training", train_labels), ("Validation", val_labels)]:
        unique, counts = np.unique(lbls, return_counts=True)
        logger.info(f"{name} class distribution:")
        for val, count in zip(unique, counts):
            logger.info(f"  Class {val}: {count:,} samples ({100 * count / len(lbls):.2f}%)")
    
    # Print year distribution for both sets if multi-year data
    if len(np.unique(year_indices)) > 1:
        for name, years in [("Training", train_years), ("Validation", val_years)]:
            unique_years, year_counts = np.unique(years, return_counts=True)
            logger.info(f"{name} year distribution:")
            for year, count in zip(unique_years, year_counts):
                logger.info(f"  Year {year}: {count:,} samples ({100 * count / len(years):.2f}%)")

    return train_features, train_labels, train_coords, val_features, val_labels, val_coords, train_years, val_years

def load_processed_data(base_dir: str, max_positive_samples: int = None, years: list = None, 
                       use_cache: bool = False, cache_dir: str = None, pos_neg_ratio: float = 0.33) -> Dict[str, np.ndarray]:
    """
    Load and process all data for PV detection from multiple years.

    Args:
        base_dir: base directory containing the data files
        max_positive_samples: maximum number of positive samples to use for training
        years: list of years to load data from (default: 2017-2024)
        use_cache: if True, use cached coordinates and fast loading via memmap
        cache_dir: directory to store cache files (default: base_dir/cache)
        pos_neg_ratio: ratio of positive to negative samples for balanced sampling

    Returns:
        Dictionary containing processed data from all years
    """
    if years is None:
        years = list(range(2017, 2025))  # 2017-2024
    
    if cache_dir is None:
        cache_dir = os.path.join(base_dir, "cache")
    
    logger.info(f"Loading data from years: {years}")
    logger.info(f"Use cache: {use_cache}")
    
    if use_cache:
        return _load_data_with_cache(base_dir, cache_dir, years, max_positive_samples, pos_neg_ratio)
    else:
        return _load_data_full(base_dir, years, max_positive_samples)
    
def _get_cache_paths(cache_dir: str, base_dir: str) -> Dict[str, str]:
    """
    Get cache file paths.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a hash of the base_dir to make cache unique to this dataset
    import hashlib
    base_dir_hash = hashlib.md5(base_dir.encode()).hexdigest()[:8]
    
    return {
        'valid_coords': cache_dir / f"valid_coords_{base_dir_hash}.npy",
        'valid_mask': cache_dir / f"valid_mask_{base_dir_hash}.npy", 
        'pos_coords': cache_dir / f"pos_coords_{base_dir_hash}.npy",
        'neg_coords_sample': cache_dir / f"neg_coords_sample_{base_dir_hash}.npy",
        'cache_info': cache_dir / f"cache_info_{base_dir_hash}.pkl"
    }


def _create_coordinate_cache(base_dir: str, cache_dir: str, years: list, pos_neg_ratio: float = 0.33):
    """
    Create cache files for coordinates and sampling indices.
    """
    logger.info("Creating coordinate cache...")
    
    cache_paths = _get_cache_paths(cache_dir, base_dir)
    
    # Load labels (shared across all years)
    labels_path = os.path.join(base_dir, "roi_1_clipped_gt_10m.npy")
    labels = np.load(labels_path)
    
    # Use first year to determine valid mask (should be same for all years)
    first_year = years[0]
    representation_path = os.path.join(base_dir, f"{first_year}_roi_1_map_10m_utm30n_128bands.npy")
    scales_path = os.path.join(base_dir, f"{first_year}_roi_1_map_10m_utm30n_scales.npy")
    
    # Load and get valid mask
    representation = load_and_dequantize_representation(representation_path, scales_path)
    valid_mask = identify_valid_pixels(representation)
    
    # Get all valid coordinates
    valid_coords = np.column_stack(np.where(valid_mask))
    valid_labels = labels[valid_mask]
    
    # Separate positive and negative coordinates
    pos_indices = np.where(valid_labels == 1)[0]
    neg_indices = np.where(valid_labels == 0)[0]
    
    pos_coords = valid_coords[pos_indices]
    neg_coords = valid_coords[neg_indices]
    
    logger.info(f"Found {len(pos_coords)} positive coordinates")
    logger.info(f"Found {len(neg_coords)} negative coordinates")
    
    # Pre-sample negative coordinates for balanced sampling
    # Calculate how many negative samples we might need
    max_pos_samples = len(pos_coords) * len(years)  # Maximum possible positive samples
    max_neg_samples_needed = int(max_pos_samples / pos_neg_ratio)
    
    # Sample more than needed to ensure we have enough
    neg_sample_size = min(len(neg_coords), max_neg_samples_needed * 2)
    np.random.seed(42)  # For reproducibility
    neg_sample_indices = np.random.choice(len(neg_coords), size=neg_sample_size, replace=False)
    neg_coords_sample = neg_coords[neg_sample_indices]
    
    logger.info(f"Pre-sampled {len(neg_coords_sample)} negative coordinates for balanced sampling")
    
    # Save cache files
    np.save(cache_paths['valid_coords'], valid_coords)
    np.save(cache_paths['valid_mask'], valid_mask)
    np.save(cache_paths['pos_coords'], pos_coords)
    np.save(cache_paths['neg_coords_sample'], neg_coords_sample)
    
    # Save cache info
    cache_info = {
        'years': years,
        'labels_shape': labels.shape,
        'pos_neg_ratio': pos_neg_ratio,
        'total_valid_pixels': len(valid_coords),
        'positive_pixels': len(pos_coords),
        'negative_pixels': len(neg_coords),
        'neg_sample_size': len(neg_coords_sample)
    }
    
    with open(cache_paths['cache_info'], 'wb') as f:
        pickle.dump(cache_info, f)
    
    logger.info("Coordinate cache created successfully")
    return cache_info


def _load_data_with_cache(base_dir: str, cache_dir: str, years: list, max_positive_samples: int = None, pos_neg_ratio: float = 0.33) -> Dict[str, np.ndarray]:
    """
    Load data using cached coordinates and memmap for fast access.
    """
    cache_paths = _get_cache_paths(cache_dir, base_dir)
    
    # Check if cache exists and is valid
    if not all(path.exists() for path in cache_paths.values()):
        logger.info("Cache not found, creating cache...")
        _create_coordinate_cache(base_dir, cache_dir, years, pos_neg_ratio)
    else:
        # Load cache info and validate
        with open(cache_paths['cache_info'], 'rb') as f:
            cache_info = pickle.load(f)
        
        if cache_info['years'] != years or cache_info['pos_neg_ratio'] != pos_neg_ratio:
            logger.info("Cache is outdated, recreating...")
            _create_coordinate_cache(base_dir, cache_dir, years, pos_neg_ratio)
    
    # Load cached coordinates
    logger.info("Loading cached coordinates...")
    pos_coords = np.load(cache_paths['pos_coords'])
    neg_coords_sample = np.load(cache_paths['neg_coords_sample'])
    
    # Load labels
    labels_path = os.path.join(base_dir, "roi_1_clipped_gt_10m.npy")
    labels = np.load(labels_path)
    
    logger.info(f"Using cached coordinates: {len(pos_coords)} positive, {len(neg_coords_sample)} negative (pre-sampled)")
    
    # Collect features from all years using memmap
    all_features = []
    all_labels = []
    all_coords = []
    all_year_indices = []
    
    for year in years:
        logger.info(f"Loading features for year {year} using memmap...")
        
        # File paths for this year
        representation_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_128bands.npy")
        scales_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_scales.npy")
        
        if not os.path.exists(representation_path) or not os.path.exists(scales_path):
            logger.warning(f"Data files for year {year} not found, skipping...")
            continue
        
        # Load using memmap for memory efficiency
        representation_mmap = np.load(representation_path, mmap_mode='r')
        scales_mmap = np.load(scales_path, mmap_mode='r')
        
        # Extract positive samples
        pos_features = []
        for coord in pos_coords:
            y, x = coord
            feature = representation_mmap[y, x] * scales_mmap[y, x]
            pos_features.append(feature)
        pos_features = np.array(pos_features)
        pos_labels = np.ones(len(pos_features))
        pos_year_indices = np.full(len(pos_features), year)
        
        # Extract negative samples (use subset for balanced sampling)
        neg_sample_size = int(len(pos_features) / pos_neg_ratio) - len(pos_features)
        neg_sample_size = min(neg_sample_size, len(neg_coords_sample))
        
        if neg_sample_size > 0:
            # Use different subset for each year to add variety
            np.random.seed(year)  # Different seed for each year
            neg_indices = np.random.choice(len(neg_coords_sample), size=neg_sample_size, replace=False)
            selected_neg_coords = neg_coords_sample[neg_indices]
            
            neg_features = []
            for coord in selected_neg_coords:
                y, x = coord
                feature = representation_mmap[y, x] * scales_mmap[y, x]
                neg_features.append(feature)
            neg_features = np.array(neg_features)
            neg_labels = np.zeros(len(neg_features))
            neg_year_indices = np.full(len(neg_features), year)
        else:
            neg_features = np.array([]).reshape(0, pos_features.shape[1])
            neg_labels = np.array([])
            neg_year_indices = np.array([])
            selected_neg_coords = np.array([]).reshape(0, 2)
        
        # Combine positive and negative samples for this year
        year_features = np.concatenate([pos_features, neg_features], axis=0)
        year_labels = np.concatenate([pos_labels, neg_labels], axis=0)
        year_coords = np.concatenate([pos_coords, selected_neg_coords], axis=0)
        year_indices = np.concatenate([pos_year_indices, neg_year_indices], axis=0)
        
        all_features.append(year_features)
        all_labels.append(year_labels)
        all_coords.append(year_coords)
        all_year_indices.append(year_indices)
        
        logger.info(f"Year {year}: {len(pos_features)} positive, {len(neg_features)} negative samples")
    
    # Concatenate all years
    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_coords = np.concatenate(all_coords, axis=0)
    combined_year_indices = np.concatenate(all_year_indices, axis=0)
    
    logger.info(f"Total samples: {len(combined_features)} ({np.sum(combined_labels == 1)} positive, {np.sum(combined_labels == 0)} negative)")
    
    # Apply max_positive_samples limit if specified
    if max_positive_samples is not None and np.sum(combined_labels == 1) > max_positive_samples:
        logger.info(f"Limiting positive samples to {max_positive_samples}")
        
        pos_indices = np.where(combined_labels == 1)[0]
        neg_indices = np.where(combined_labels == 0)[0]
        
        # Randomly select positive samples
        np.random.seed(42)
        selected_pos_indices = np.random.choice(pos_indices, size=max_positive_samples, replace=False)
        
        # Keep all negative samples
        selected_indices = np.concatenate([selected_pos_indices, neg_indices])
        
        combined_features = combined_features[selected_indices]
        combined_labels = combined_labels[selected_indices]
        combined_coords = combined_coords[selected_indices]
        combined_year_indices = combined_year_indices[selected_indices]
    
    # Simple train/val split (80/20)
    n_samples = len(combined_features)
    n_train = int(0.8 * n_samples)
    
    # Shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_features = combined_features[train_indices]
    train_labels = combined_labels[train_indices]
    train_coords = combined_coords[train_indices]
    train_years = combined_year_indices[train_indices]
    
    val_features = combined_features[val_indices]
    val_labels = combined_labels[val_indices]
    val_coords = combined_coords[val_indices]
    val_years = combined_year_indices[val_indices]
    
    logger.info(f"Train set: {len(train_features)} samples ({np.sum(train_labels == 1)} positive)")
    logger.info(f"Val set: {len(val_features)} samples ({np.sum(val_labels == 1)} positive)")
    
    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'train_coords': train_coords,
        'train_years': train_years,
        'val_features': val_features,
        'val_labels': val_labels,
        'val_coords': val_coords,
        'val_years': val_years,
        'years': years,
        'labels': labels,
        'use_cache': True  # Flag to indicate cache was used
    }
def _load_data_full(base_dir: str, years: list, max_positive_samples: int = None) -> Dict[str, np.ndarray]:
    """
    Original full data loading method (without cache).
    """
    # Load labels (shared across all years)
    labels_path = os.path.join(base_dir, "roi_1_clipped_gt_10m.npy")
    logger.info(f"Loading labels from {labels_path}")
    labels = np.load(labels_path)
    logger.info(f"Labels shape: {labels.shape}")
    
    # Initialize lists to store data from all years
    all_valid_features = []
    all_valid_labels = []
    all_valid_coords = []
    all_year_indices = []  # Track which year each sample comes from
    
    for year in years:
        logger.info(f"Processing data for year {year}")
        
        # File paths for this year
        representation_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_128bands.npy")
        scales_path = os.path.join(base_dir, f"{year}_roi_1_map_10m_utm30n_scales.npy")
        
        # Check if files exist
        if not os.path.exists(representation_path) or not os.path.exists(scales_path):
            logger.warning(f"Data files for year {year} not found, skipping...")
            continue
        
        # Load representation and dequantize
        representation = load_and_dequantize_representation(representation_path, scales_path)
        
        # Identify valid pixels
        valid_mask = identify_valid_pixels(representation)
        
        # Extract valid data
        valid_features, valid_labels, valid_coords = extract_valid_data(representation, labels, valid_mask)
        
        # Store data from this year
        all_valid_features.append(valid_features)
        all_valid_labels.append(valid_labels)
        all_valid_coords.append(valid_coords)
        
        # Create year indices for this year's data
        year_indices = np.full(len(valid_features), year)
        all_year_indices.append(year_indices)
        
        logger.info(f"Year {year}: {len(valid_features)} valid samples")
    
    # Concatenate all years' data
    logger.info("Concatenating data from all years...")
    combined_features = np.concatenate(all_valid_features, axis=0)
    combined_labels = np.concatenate(all_valid_labels, axis=0)
    combined_coords = np.concatenate(all_valid_coords, axis=0)
    combined_year_indices = np.concatenate(all_year_indices, axis=0)
    
    logger.info(f"Total samples across all years: {len(combined_features)}")
    logger.info(f"Total positive samples: {np.sum(combined_labels == 1)}")
    logger.info(f"Total negative samples: {np.sum(combined_labels == 0)}")
    
    # Split into train/val with multi-year data
    train_features, train_labels, train_coords, val_features, val_labels, val_coords, train_years, val_years = train_val_split(
        combined_features, combined_labels, combined_coords, combined_year_indices, max_positive_samples=max_positive_samples
    )

    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'train_coords': train_coords,
        'train_years': train_years,
        'val_features': val_features,
        'val_labels': val_labels,
        'val_coords': val_coords,
        'val_years': val_years,
        'years': years,
        'labels': labels  # Original labels for reference
    }

if __name__ == "__main__":
    # Test the data loading
    base_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1"
    data = load_processed_data(base_dir)
    print("Data loading completed successfully!")