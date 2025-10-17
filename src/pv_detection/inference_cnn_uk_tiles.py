import numpy as np
import torch
import torch.nn.functional as F
import os
import time
import logging
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, Optional, List, Dict
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import json
import glob
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

from data_preprocessing import load_and_dequantize_representation, identify_valid_pixels
from patch_data_preprocessing import PatchExtractor
from cnn_models import create_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""_summary_

Raises:
    FileNotFoundError: _description_

Returns:
    _type_: _description_

cp -r /maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_-5.35_57.15 /maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data/2017/
cp -r /maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_-3.75_56.15 /maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data/2017/
cp -r /maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_-0.65_52.75 /maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data/2017/
cp -r /maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_-0.95_53.95 /maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data/2017/
cp -r /maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_-2.65_53.65 /maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data/2017/
cp -r /maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_-3.95_50.35 /maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data/2017/

"""

class PVDetectionCNNInference:
    """
    Inference class for PV detection using trained CNN models.
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize CNN inference engine.

        Args:
            model_path: path to saved CNN model (.pth file)
            device: device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_path = model_path
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.config = None
        self.optimal_threshold = 0.999
        self.patch_extractor = None

        logger.info(f"Initializing PV Detection CNN Inference")
        logger.info(f"Model path: {model_path}")

        self._load_model()

    def _load_model(self) -> None:
        """Load the trained CNN model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract configuration and threshold
        self.config = checkpoint['config']
        # self.optimal_threshold = checkpoint.get('optimal_threshold', 0.999)
        self.optimal_threshold = 0.999
        
        logger.info(f"Model configuration:")
        for key, value in self.config.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Optimal threshold: {self.optimal_threshold:.3f}")

        # Create model
        self.model = create_model(
            model_type=self.config['model_type'],
            input_channels=self.config['input_channels'],
            num_classes=self.config['num_classes'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Initialize patch extractor
        self.patch_extractor = PatchExtractor(
            patch_size=self.config['patch_size'],
            padding_mode='reflect'
        )

        logger.info(f"Loaded CNN model successfully")

    def load_inference_data(self, representation_path: str, scales_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess data for inference.

        Args:
            representation_path: path to representation file
            scales_path: path to scales file

        Returns:
            representation: dequantized representation array (H, W, C)
            valid_mask: boolean mask of valid pixels (H, W)
            valid_features: features for valid pixels (N_valid, C)
        """
        logger.info("Loading inference data...")

        # Load and dequantize representation
        representation = load_and_dequantize_representation(representation_path, scales_path)

        # Identify valid pixels
        valid_mask = identify_valid_pixels(representation)

        # Extract valid features (for compatibility, though not used in CNN)
        valid_features = representation[valid_mask]

        logger.info(f"Loaded representation shape: {representation.shape}")
        logger.info(f"Valid pixels: {np.sum(valid_mask):,} / {valid_mask.size:,}")

        return representation, valid_mask, valid_features

    def extract_patches_for_inference(self, representation: np.ndarray, 
                                    valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract patches for CNN inference.

        Args:
            representation: representation array (H, W, C)
            valid_mask: valid pixels mask (H, W)

        Returns:
            patches: extracted patches (N, C, patch_size, patch_size)
            patch_coords: coordinates of patch centers (N, 2)
            patch_valid_mask: mask indicating which patches are valid (N,)
        """
        logger.info("Extracting patches for inference...")

        height, width, channels = representation.shape
        patch_size = self.config['patch_size']
        pad_width = patch_size // 2

        # Pad the representation and valid_mask
        padded_representation = np.pad(
            representation,
            ((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
            mode='reflect'
        )

        padded_valid_mask = np.pad(
            valid_mask,
            ((pad_width, pad_width), (pad_width, pad_width)),
            mode='constant',
            constant_values=False
        )

        # Get all pixel coordinates (including invalid ones for full coverage)
        all_rows, all_cols = np.meshgrid(range(height), range(width), indexing='ij')
        all_rows = all_rows.flatten()
        all_cols = all_cols.flatten()
        n_pixels = len(all_rows)

        logger.info(f"Processing {n_pixels:,} pixels")

        # Extract patches for all pixels
        patches = np.zeros((n_pixels, channels, patch_size, patch_size), dtype=np.float32)
        patch_coords = np.zeros((n_pixels, 2), dtype=np.int32)
        patch_valid_mask = np.zeros(n_pixels, dtype=bool)

        for i, (row, col) in enumerate(zip(all_rows, all_cols)):
            # Adjust coordinates for padded array
            padded_row = row + pad_width
            padded_col = col + pad_width

            # Extract patch from padded representation
            patch = padded_representation[
                padded_row - pad_width:padded_row + pad_width + 1,
                padded_col - pad_width:padded_col + pad_width + 1,
                :
            ]

            # Check if center pixel is valid
            center_valid = valid_mask[row, col]

            # Convert to PyTorch format: (C, H, W)
            patches[i] = patch.transpose(2, 0, 1)
            patch_coords[i] = [row, col]
            patch_valid_mask[i] = center_valid

        logger.info(f"Extracted {n_pixels:,} patches, {np.sum(patch_valid_mask):,} with valid centers")

        return patches, patch_coords, patch_valid_mask

    def predict_patches(self, patches: np.ndarray, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on patches using CNN.

        Args:
            patches: input patches array (N, C, H, W)
            batch_size: batch size for inference

        Returns:
            predictions: binary predictions (N,)
            probabilities: prediction probabilities for positive class (N,)
        """
        logger.info(f"Making CNN predictions on {len(patches):,} patches...")

        start_time = time.time()
        n_patches = len(patches)
        
        all_probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_patches, batch_size):
                end_idx = min(i + batch_size, n_patches)
                batch_patches = patches[i:end_idx]
                
                # Convert to tensor and move to device
                batch_tensor = torch.from_numpy(batch_patches).to(self.device)
                
                # Forward pass
                outputs = self.model(batch_tensor)
                
                # Get probabilities for positive class
                probabilities = F.softmax(outputs, dim=1)[:, 1]  # Class 1 probabilities
                
                all_probabilities.append(probabilities.cpu().numpy())

        # Concatenate all probabilities
        probabilities = np.concatenate(all_probabilities)
        
        # Apply optimal threshold
        predictions = (probabilities >= self.optimal_threshold).astype(int)

        inference_time = time.time() - start_time
        logger.info(f"CNN inference completed in {inference_time:.2f} seconds")

        # Log prediction statistics
        n_positive = np.sum(predictions)
        n_total = len(predictions)
        logger.info(f"Predicted PV pixels: {n_positive:,} / {n_total:,} ({100*n_positive/n_total:.2f}%)")

        return predictions, probabilities

    def create_prediction_map(self, predictions: np.ndarray, patch_coords: np.ndarray,
                            patch_valid_mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create full prediction map from patch predictions.

        Args:
            predictions: predictions for patches (N,)
            patch_coords: coordinates of patch centers (N, 2)
            patch_valid_mask: mask indicating which patches had valid centers (N,)
            original_shape: original shape (H, W)

        Returns:
            prediction_map: full prediction map (H, W)
        """
        prediction_map = np.zeros(original_shape, dtype=np.uint8)
        
        # Only use predictions for patches with valid centers
        valid_predictions = predictions[patch_valid_mask]
        valid_coords = patch_coords[patch_valid_mask]
        
        # Fill prediction map
        for pred, (row, col) in zip(valid_predictions, valid_coords):
            prediction_map[row, col] = pred

        return prediction_map

    def save_as_geotiff(self, prediction_map: np.ndarray, output_path: str,
                       reference_tiff_path: str = None, crs: str = 'EPSG:32630') -> None:
        """
        Save prediction map as GeoTIFF with proper georeferencing based on reference TIFF.

        Args:
            prediction_map: prediction map (H, W)
            output_path: path to save the GeoTIFF
            reference_tiff_path: path to reference TIFF for spatial information
            crs: coordinate reference system (default: EPSG:32630 - WGS 84 / UTM zone 30N)
        """
        height, width = prediction_map.shape

        if reference_tiff_path and os.path.exists(reference_tiff_path):
            # Read spatial information from reference TIFF
            with rasterio.open(reference_tiff_path) as ref:
                ref_bounds = ref.bounds
                ref_height, ref_width = ref.height, ref.width
                ref_crs = ref.crs

                logger.info(f"Reference TIFF info:")
                logger.info(f"  Size: {ref_width} x {ref_height}")
                logger.info(f"  Bounds: {ref_bounds}")
                logger.info(f"  CRS: {ref_crs}")

                # Calculate transform that maps our array to the reference bounds
                transform = rasterio.transform.from_bounds(
                    ref_bounds.left, ref_bounds.bottom,
                    ref_bounds.right, ref_bounds.top,
                    width, height
                )

                # Calculate actual pixel size for our output
                actual_pixel_size_x = (ref_bounds.right - ref_bounds.left) / width
                actual_pixel_size_y = (ref_bounds.top - ref_bounds.bottom) / height

                logger.info(f"Output array size: {width} x {height}")
                logger.info(f"Actual output pixel size: {actual_pixel_size_x:.6f} x {actual_pixel_size_y:.6f}")

                # Use reference CRS if available
                if ref_crs:
                    output_crs = ref_crs
                else:
                    output_crs = CRS.from_string(crs)
        else:
            # Fallback to default parameters if no reference
            logger.warning(f"Reference TIFF not found: {reference_tiff_path}")
            logger.info("Using default spatial parameters")
            pixel_size = 10.0
            upper_left_x = 0.0
            upper_left_y = height * pixel_size

            left = upper_left_x
            top = upper_left_y
            right = left + width * pixel_size
            bottom = top - height * pixel_size

            transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
            output_crs = CRS.from_string(crs)

        # Convert prediction map to appropriate data type for GeoTIFF
        # For TIFF: nodata for background, 1 for PV pixels
        geotiff_array = np.full(prediction_map.shape, 255, dtype=np.uint8)  # Use 255 as nodata

        # Set PV pixels to 1
        geotiff_array[prediction_map == 1] = 1

        # Keep nodata (255) for non-PV areas
        nodata_value = 255

        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            crs=output_crs,
            transform=transform,
            nodata=nodata_value,
            compress='lzw'
        ) as dst:
            dst.write(geotiff_array, 1)

        logger.info(f"Saved GeoTIFF to: {output_path}")

    def visualize_predictions(self, prediction_map: np.ndarray, valid_mask: np.ndarray,
                            output_path: str, dpi: int = 300) -> None:
        """
        Create and save visualization of predictions.

        Args:
            prediction_map: prediction map (H, W)
            valid_mask: valid pixels mask (H, W)
            output_path: path to save the visualization
            dpi: resolution for saved image
        """
        logger.info("Creating prediction visualization...")

        # Create visualization array
        # 0: Invalid pixels (gray)
        # 1: Valid non-PV pixels (blue)
        # 2: PV pixels (red)
        vis_array = np.zeros_like(prediction_map, dtype=np.uint8)
        vis_array[~valid_mask] = 0  # Invalid pixels
        vis_array[valid_mask & (prediction_map == 0)] = 1  # Valid non-PV
        vis_array[valid_mask & (prediction_map == 1)] = 2  # PV pixels

        # Create color map
        colors = ['gray', 'lightblue', 'red']
        n_colors = len(colors)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Display the visualization
        im = ax.imshow(vis_array, cmap=plt.cm.colors.ListedColormap(colors),
                      vmin=0, vmax=n_colors-1, interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['Invalid', 'Non-PV', 'PV'])

        # Set title and labels
        ax.set_title('PV Detection Results (CNN)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')

        # Add statistics
        n_total_valid = np.sum(valid_mask)
        n_pv = np.sum(valid_mask & (prediction_map == 1))
        pv_percentage = 100 * n_pv / n_total_valid if n_total_valid > 0 else 0

        stats_text = f'Valid pixels: {n_total_valid:,}\nPV pixels: {n_pv:,} ({pv_percentage:.2f}%)'
        ax.text(0.02, 0.9998, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization to: {output_path}")

    def run_inference(self, representation_path: str, scales_path: str,
                     output_path: str, reference_tiff_path: str = None,
                     visualize: bool = True, save_geotiff: bool = True,
                     batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Run complete CNN inference pipeline.

        Args:
            representation_path: path to representation file
            scales_path: path to scales file
            output_path: base path for output files (without extension)
            reference_tiff_path: path to reference TIFF for spatial information
            visualize: whether to create visualization
            save_geotiff: whether to save as GeoTIFF
            batch_size: batch size for CNN inference

        Returns:
            prediction_map: prediction map (H, W)
            probabilities_map: probabilities map (H, W)
            stats: statistics dictionary
        """
        logger.info("Starting CNN inference pipeline...")
        start_time = time.time()

        # Load data
        representation, valid_mask, _ = self.load_inference_data(representation_path, scales_path)

        # Extract patches
        patches, patch_coords, patch_valid_mask = self.extract_patches_for_inference(
            representation, valid_mask
        )

        # Make predictions
        predictions, probabilities = self.predict_patches(patches, batch_size=batch_size)

        # Create prediction map
        prediction_map = self.create_prediction_map(
            predictions, patch_coords, patch_valid_mask, representation.shape[:2]
        )

        # Create probabilities map (for valid pixels only)
        probabilities_map = np.zeros(representation.shape[:2], dtype=np.float32)
        valid_coords = patch_coords[patch_valid_mask]
        valid_probabilities = probabilities[patch_valid_mask]
        for prob, (row, col) in zip(valid_probabilities, valid_coords):
            probabilities_map[row, col] = prob

        # Calculate statistics
        total_pixels = representation.shape[0] * representation.shape[1]
        valid_pixels = np.sum(valid_mask)
        pv_pixels = np.sum(prediction_map == 1)
        pv_ratio_total = pv_pixels / total_pixels
        pv_ratio_valid = pv_pixels / valid_pixels if valid_pixels > 0 else 0
        
        stats = {
            'total_pixels': int(total_pixels),
            'valid_pixels': int(valid_pixels),
            'pv_pixels': int(pv_pixels),
            'pv_ratio_total': float(pv_ratio_total),
            'pv_ratio_valid': float(pv_ratio_valid),
            'inference_time': time.time() - start_time
        }

        # Save outputs
        if save_geotiff:
            geotiff_path = f"{output_path}.tiff"
            self.save_as_geotiff(prediction_map, geotiff_path, reference_tiff_path)

        if visualize:
            png_path = f"{output_path}.png"
            self.visualize_predictions(prediction_map, valid_mask, png_path)

        total_time = time.time() - start_time
        logger.info(f"CNN inference pipeline completed in {total_time:.2f} seconds")

        return prediction_map, probabilities_map, stats

def find_grid_directories(base_dir: str, years: List[str] = None) -> List[Dict]:
    """
    Find all grid directories in the UK data structure.
    
    Args:
        base_dir: base directory path (e.g., '/maps/zf281/btfm4rs/data/downstream/pv_detection/uk')
        years: list of years to process (e.g., ['2024']). If None, process all available years.
    
    Returns:
        List of dictionaries containing grid information
    """
    grid_info = []
    
    # If no years specified, find all available years
    if years is None:
        year_pattern = os.path.join(base_dir, "*")
        year_dirs = [d for d in glob.glob(year_pattern) if os.path.isdir(d)]
        years = [os.path.basename(d) for d in year_dirs]
        logger.info(f"Found years: {years}")
    
    for year in years:
        year_dir = os.path.join(base_dir, year)
        if not os.path.exists(year_dir):
            logger.warning(f"Year directory not found: {year_dir}")
            continue
            
        # Find all grid directories in this year
        grid_pattern = os.path.join(year_dir, "grid_*")
        grid_dirs = glob.glob(grid_pattern)
        
        for grid_dir in grid_dirs:
            if not os.path.isdir(grid_dir):
                continue
                
            grid_name = os.path.basename(grid_dir)
            
            # Check if required files exist
            npy_file = os.path.join(grid_dir, f"{grid_name}.npy")
            scales_file = os.path.join(grid_dir, f"{grid_name}_scales.npy")
            tiff_file = os.path.join(grid_dir, f"{grid_name}.tiff")
            
            if all(os.path.exists(f) for f in [npy_file, scales_file, tiff_file]):
                grid_info.append({
                    'year': year,
                    'grid_name': grid_name,
                    'grid_dir': grid_dir,
                    'npy_file': npy_file,
                    'scales_file': scales_file,
                    'tiff_file': tiff_file
                })
            else:
                missing_files = [f for f in [npy_file, scales_file, tiff_file] if not os.path.exists(f)]
                logger.warning(f"Missing files in {grid_dir}: {missing_files}")
    
    logger.info(f"Found {len(grid_info)} valid grid directories")
    return grid_info


def save_inference_log(output_path: str, grid_info: Dict, stats: Dict, model_path: str) -> None:
    """
    Save inference log file with statistics and metadata.
    
    Args:
        output_path: path for log file (without extension)
        grid_info: grid information dictionary
        stats: statistics dictionary from inference
        model_path: path to the model used
    """
    log_path = f"{output_path}.log"
    
    with open(log_path, 'w') as f:
        f.write("PV Detection CNN Inference Log\n")
        f.write("=" * 50 + "\n\n")
        
        # Metadata
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Year: {grid_info['year']}\n")
        f.write(f"Grid: {grid_info['grid_name']}\n")
        f.write(f"Grid Directory: {grid_info['grid_dir']}\n\n")
        
        # Statistics
        f.write("Inference Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total pixels: {stats['total_pixels']:,}\n")
        f.write(f"Valid pixels: {stats['valid_pixels']:,}\n")
        f.write(f"PV pixels detected: {stats['pv_pixels']:,}\n")
        f.write(f"PV ratio (total): {stats['pv_ratio_total']:.6f} ({stats['pv_ratio_total']*100:.4f}%)\n")
        f.write(f"PV ratio (valid): {stats['pv_ratio_valid']:.6f} ({stats['pv_ratio_valid']*100:.4f}%)\n")
        f.write(f"Inference time: {stats['inference_time']:.2f} seconds\n\n")
        
        # File paths
        f.write("Input Files:\n")
        f.write("-" * 12 + "\n")
        f.write(f"Representation: {grid_info['npy_file']}\n")
        f.write(f"Scales: {grid_info['scales_file']}\n")
        f.write(f"Reference TIFF: {grid_info['tiff_file']}\n\n")
        
        f.write("Output Files:\n")
        f.write("-" * 12 + "\n")
        f.write(f"Prediction PNG: {output_path}.png\n")
        f.write(f"Prediction TIFF: {output_path}.tiff\n")
        f.write(f"Log file: {log_path}\n")


def check_grid_processed(output_base_dir: str, year: str, grid_name: str) -> bool:
    """
    Check if a grid has already been processed.
    
    Args:
        output_base_dir: base output directory
        year: year string
        grid_name: grid name
        
    Returns:
        True if all output files exist, False otherwise
    """
    year_output_dir = os.path.join(output_base_dir, year)
    output_path = os.path.join(year_output_dir, f"{grid_name}_prediction")
    
    required_files = [
        f"{output_path}.png",
        f"{output_path}.tiff", 
        f"{output_path}.log"
    ]
    
    return all(os.path.exists(f) for f in required_files)


def process_single_grid(args_tuple) -> Dict:
    """
    Process a single grid for inference. This function is designed to be called by multiprocessing.
    
    Args:
        args_tuple: tuple containing (grid_info, model_path, output_base_dir, batch_size, device, overwrite)
        
    Returns:
        Dictionary with processing results
    """
    grid_info, model_path, output_base_dir, batch_size, device, overwrite = args_tuple
    
    try:
        # Check if already processed
        if not overwrite and check_grid_processed(output_base_dir, grid_info['year'], grid_info['grid_name']):
            return {
                'status': 'skipped',
                'grid_name': grid_info['grid_name'],
                'year': grid_info['year'],
                'message': 'Already processed'
            }
        
        # Initialize inference engine for this process
        inference_engine = PVDetectionCNNInference(
            model_path=model_path,
            device=device
        )
        
        # Create output directory
        year_output_dir = os.path.join(output_base_dir, grid_info['year'])
        os.makedirs(year_output_dir, exist_ok=True)
        
        # Define output path
        output_path = os.path.join(year_output_dir, f"{grid_info['grid_name']}_prediction")
        
        # Run inference
        prediction_map, probabilities_map, stats = inference_engine.run_inference(
            representation_path=grid_info['npy_file'],
            scales_path=grid_info['scales_file'],
            output_path=output_path,
            reference_tiff_path=grid_info['tiff_file'],
            visualize=True,
            save_geotiff=True,
            batch_size=batch_size
        )
        
        # Save log file
        save_inference_log(output_path, grid_info, stats, model_path)
        
        return {
            'status': 'success',
            'grid_name': grid_info['grid_name'],
            'year': grid_info['year'],
            'stats': stats,
            'message': f"{stats['pv_pixels']} PV pixels ({stats['pv_ratio_valid']*100:.4f}% of valid pixels)"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'grid_name': grid_info['grid_name'],
            'year': grid_info['year'],
            'message': str(e)
        }


def run_parallel_inference(model_path: str, base_data_dir: str, output_base_dir: str,
                          years: List[str] = None, batch_size: int = 1024, device: str = 'auto',
                          num_workers: int = 4, overwrite: bool = False) -> None:
    """
    Run CNN inference on multiple grids in parallel across multiple years.
    
    Args:
        model_path: path to trained CNN model
        base_data_dir: base directory containing UK data
        output_base_dir: base directory for output files
        years: list of years to process. If None, process all available years.
        batch_size: batch size for CNN inference
        device: device to use for inference
        num_workers: number of parallel workers
        overwrite: whether to overwrite existing results
    """
    logger.info("Starting parallel CNN inference...")
    logger.info(f"Model: {model_path}")
    logger.info(f"Data directory: {base_data_dir}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"Years: {years if years else 'All available'}")
    logger.info(f"Number of workers: {num_workers}")
    logger.info(f"Overwrite existing: {overwrite}")
    
    # Find all grid directories
    grid_list = find_grid_directories(base_data_dir, years)
    
    if not grid_list:
        logger.error("No valid grid directories found!")
        return
    
    # Filter out already processed grids if not overwriting
    if not overwrite:
        original_count = len(grid_list)
        grid_list = [grid for grid in grid_list 
                    if not check_grid_processed(output_base_dir, grid['year'], grid['grid_name'])]
        skipped_count = original_count - len(grid_list)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already processed grids")
    
    if not grid_list:
        logger.info("All grids have been processed!")
        return
    
    # Prepare arguments for parallel processing
    args_list = [(grid_info, model_path, output_base_dir, batch_size, device, overwrite) 
                 for grid_info in grid_list]
    
    # Process grids in parallel
    total_grids = len(grid_list)
    successful_inferences = 0
    failed_inferences = 0
    skipped_inferences = 0
    
    logger.info(f"Processing {total_grids} grids with {num_workers} workers...")
    
    # Use a thread-safe progress bar
    pbar = tqdm(total=total_grids, desc="Processing grids", unit="grid")
    pbar_lock = threading.Lock()
    
    def update_progress(result):
        with pbar_lock:
            if result['status'] == 'success':
                pbar.set_description(f"✓ {result['year']}/{result['grid_name']}")
            elif result['status'] == 'skipped':
                pbar.set_description(f"⏭ {result['year']}/{result['grid_name']}")
            else:
                pbar.set_description(f"✗ {result['year']}/{result['grid_name']}")
            pbar.update(1)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_grid = {executor.submit(process_single_grid, args): args[0] 
                         for args in args_list}
        
        # Process completed tasks
        for future in as_completed(future_to_grid):
            grid_info = future_to_grid[future]
            try:
                result = future.result()
                
                if result['status'] == 'success':
                    successful_inferences += 1
                    logger.info(f"✓ {result['year']}/{result['grid_name']}: {result['message']}")
                elif result['status'] == 'skipped':
                    skipped_inferences += 1
                    logger.info(f"⏭ {result['year']}/{result['grid_name']}: {result['message']}")
                else:
                    failed_inferences += 1
                    logger.error(f"✗ {result['year']}/{result['grid_name']}: {result['message']}")
                
                update_progress(result)
                
            except Exception as e:
                failed_inferences += 1
                logger.error(f"✗ {grid_info['year']}/{grid_info['grid_name']}: Unexpected error: {str(e)}")
                update_progress({
                    'status': 'error',
                    'year': grid_info['year'],
                    'grid_name': grid_info['grid_name']
                })
    
    pbar.close()
    
    # Summary
    logger.info("Parallel inference completed!")
    logger.info(f"Total grids: {total_grids}")
    logger.info(f"Successful: {successful_inferences}")
    logger.info(f"Failed: {failed_inferences}")
    logger.info(f"Skipped: {skipped_inferences}")


def run_batch_inference(model_path: str, base_data_dir: str, output_base_dir: str,
                       years: List[str] = None, batch_size: int = 1024, device: str = 'auto') -> None:
    """
    Run CNN inference on multiple grids across multiple years.
    
    Args:
        model_path: path to trained CNN model
        base_data_dir: base directory containing UK data
        output_base_dir: base directory for output files
        years: list of years to process. If None, process all available years.
        batch_size: batch size for CNN inference
        device: device to use for inference
    """
    logger.info("Starting batch CNN inference...")
    logger.info(f"Model: {model_path}")
    logger.info(f"Data directory: {base_data_dir}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"Years: {years if years else 'All available'}")
    
    # Find all grid directories
    grid_list = find_grid_directories(base_data_dir, years)
    
    if not grid_list:
        logger.error("No valid grid directories found!")
        return
    
    # Initialize inference engine
    logger.info("Initializing CNN inference engine...")
    inference_engine = PVDetectionCNNInference(
        model_path=model_path,
        device=device
    )
    
    # Process each grid with progress bar
    total_grids = len(grid_list)
    successful_inferences = 0
    failed_inferences = 0
    
    with tqdm(total=total_grids, desc="Processing grids", unit="grid") as pbar:
        for i, grid_info in enumerate(grid_list):
            try:
                # Update progress bar description
                pbar.set_description(f"Processing {grid_info['year']}/{grid_info['grid_name']}")
                
                # Create output directory
                year_output_dir = os.path.join(output_base_dir, grid_info['year'])
                os.makedirs(year_output_dir, exist_ok=True)
                
                # Define output path
                output_path = os.path.join(year_output_dir, f"{grid_info['grid_name']}_prediction")
                
                # Run inference
                logger.info(f"Processing grid {i+1}/{total_grids}: {grid_info['year']}/{grid_info['grid_name']}")
                
                prediction_map, probabilities_map, stats = inference_engine.run_inference(
                    representation_path=grid_info['npy_file'],
                    scales_path=grid_info['scales_file'],
                    output_path=output_path,
                    reference_tiff_path=grid_info['tiff_file'],
                    visualize=True,
                    save_geotiff=True,
                    batch_size=batch_size
                )
                
                # Save log file
                save_inference_log(output_path, grid_info, stats, model_path)
                
                successful_inferences += 1
                logger.info(f"Successfully processed {grid_info['grid_name']}: "
                          f"{stats['pv_pixels']} PV pixels ({stats['pv_ratio_valid']*100:.4f}% of valid pixels)")
                
            except Exception as e:
                failed_inferences += 1
                logger.error(f"Failed to process {grid_info['year']}/{grid_info['grid_name']}: {str(e)}")
                
            finally:
                pbar.update(1)
    
    # Summary
    logger.info("Batch inference completed!")
    logger.info(f"Total grids processed: {total_grids}")
    logger.info(f"Successful inferences: {successful_inferences}")
    logger.info(f"Failed inferences: {failed_inferences}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='CNN inference for PV detection')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'parallel'], default='single',
                       help='Inference mode: single grid, batch processing, or parallel processing')

    # Model and data paths
    parser.add_argument('--model_path', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection/cnn_models/cnn_model.pth',
                       help='Path to trained CNN model')
    parser.add_argument('--base_data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/uk',
                    #    default='/maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data',
                       help='Base directory containing UK data')
    parser.add_argument('--output_base_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/uk_prediction',
                    #    default='/maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data_prediction',
                       help='Base directory for output files')

    # Year selection
    parser.add_argument('--years', type=str, nargs='*', default=None,
                       help='Years to process (e.g., 2024 2023). If not specified, process all available years')

    # Single grid mode parameters (for backward compatibility)
    parser.add_argument('--representation_path', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_0.35_52.25/grid_0.35_52.25.npy',
                       help='Path to representation file (single mode only)')
    parser.add_argument('--scales_path', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_0.35_52.25/grid_0.35_52.25_scales.npy',
                       help='Path to scales file (single mode only)')
    parser.add_argument('--output_path', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection/cnn_inference_results',
                       help='Base path for output files (single mode only)')
    parser.add_argument('--reference_tiff_path', type=str, 
                       default="/maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_0.35_52.25/grid_0.35_52.25.tiff",
                       help='Path to reference TIFF for spatial information (single mode only)')

    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for CNN inference')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'], help='Device to use for inference')

    # Parallel processing parameters
    parser.add_argument('--num_workers', type=int, default=64,
                       help='Number of parallel workers (for parallel mode)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results. If not set, skip already processed grids.')

    # Output options
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip creating visualization (single mode only)')
    parser.add_argument('--no_geotiff', action='store_true',
                       help='Skip saving GeoTIFF (single mode only)')

    args = parser.parse_args()

    if args.mode == 'batch':
        # Batch processing mode
        logger.info("Running in batch processing mode")
        
        # Create output base directory
        os.makedirs(args.output_base_dir, exist_ok=True)
        
        # Run batch inference
        run_batch_inference(
            model_path=args.model_path,
            base_data_dir=args.base_data_dir,
            output_base_dir=args.output_base_dir,
            years=args.years,
            batch_size=args.batch_size,
            device=args.device
        )
        
    elif args.mode == 'parallel':
        # Parallel processing mode
        logger.info("Running in parallel processing mode")
        
        # Create output base directory
        os.makedirs(args.output_base_dir, exist_ok=True)
        
        # Run parallel inference
        run_parallel_inference(
            model_path=args.model_path,
            base_data_dir=args.base_data_dir,
            output_base_dir=args.output_base_dir,
            years=args.years,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            overwrite=args.overwrite
        )
        
    else:
        # Single grid processing mode (backward compatibility)
        logger.info("Running in single grid processing mode")
        
        # Create output directory
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize inference engine
        inference_engine = PVDetectionCNNInference(
            model_path=args.model_path,
            device=args.device
        )

        # Run inference
        prediction_map, probabilities_map, stats = inference_engine.run_inference(
            representation_path=args.representation_path,
            scales_path=args.scales_path,
            output_path=args.output_path,
            reference_tiff_path=args.reference_tiff_path,
            visualize=not args.no_visualization,
            save_geotiff=not args.no_geotiff,
            batch_size=args.batch_size
        )

        # Save log file for single mode
        grid_info = {
            'year': 'unknown',
            'grid_name': os.path.basename(args.representation_path).replace('.npy', ''),
            'grid_dir': os.path.dirname(args.representation_path),
            'npy_file': args.representation_path,
            'scales_file': args.scales_path,
            'tiff_file': args.reference_tiff_path
        }
        save_inference_log(args.output_path, grid_info, stats, args.model_path)

    logger.info("CNN inference completed successfully!")

if __name__ == "__main__":
    main()