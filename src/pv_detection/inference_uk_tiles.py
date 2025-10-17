import numpy as np
import xgboost as xgb
import pickle
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PVDetectionXGBoostInference:
    """
    Inference class for PV detection using trained XGBoost models.
    """

    def __init__(self, model_path: str):
        """
        Initialize XGBoost inference engine.

        Args:
            model_path: path to saved XGBoost model (.pkl file)
        """
        self.model_path = model_path
        self.model = None
        self.optimal_threshold = 0.995

        logger.info(f"Initializing PV Detection XGBoost Inference")
        logger.info(f"Model path: {model_path}")

        self._load_model()

    def _load_model(self) -> None:
        """Load the trained XGBoost model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Load model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Loaded XGBoost model successfully")
        logger.info(f"Optimal threshold: {self.optimal_threshold:.3f}")

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

        # Extract valid features for XGBoost inference
        valid_features = representation[valid_mask]

        logger.info(f"Loaded representation shape: {representation.shape}")
        logger.info(f"Valid pixels: {np.sum(valid_mask):,} / {valid_mask.size:,}")
        logger.info(f"Valid features shape: {valid_features.shape}")

        return representation, valid_mask, valid_features

    def predict_pixels(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on pixel features using XGBoost.

        Args:
            features: input features array (N, C)

        Returns:
            predictions: binary predictions (N,)
            probabilities: prediction probabilities for positive class (N,)
        """
        logger.info(f"Making XGBoost predictions on {len(features):,} pixels...")

        start_time = time.time()
        
        # Convert to DMatrix for XGBoost
        dtest = xgb.DMatrix(features)
        
        # Get probabilities
        probabilities = self.model.predict(dtest)
        
        # Apply optimal threshold
        predictions = (probabilities >= self.optimal_threshold).astype(int)

        inference_time = time.time() - start_time
        logger.info(f"XGBoost inference completed in {inference_time:.2f} seconds")

        # Log prediction statistics
        n_positive = np.sum(predictions)
        n_total = len(predictions)
        logger.info(f"Predicted PV pixels: {n_positive:,} / {n_total:,} ({100*n_positive/n_total:.2f}%)")

        return predictions, probabilities

    def create_prediction_map(self, predictions: np.ndarray, probabilities: np.ndarray,
                            valid_mask: np.ndarray, original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create full prediction map from pixel predictions.

        Args:
            predictions: predictions for valid pixels (N_valid,)
            probabilities: probabilities for valid pixels (N_valid,)
            valid_mask: mask indicating valid pixels (H, W)
            original_shape: original shape (H, W)

        Returns:
            prediction_map: full prediction map (H, W)
            probabilities_map: full probabilities map (H, W)
        """
        prediction_map = np.zeros(original_shape, dtype=np.uint8)
        probabilities_map = np.zeros(original_shape, dtype=np.float32)
        
        # Fill prediction map for valid pixels only
        prediction_map[valid_mask] = predictions
        probabilities_map[valid_mask] = probabilities

        return prediction_map, probabilities_map

    def save_as_geotiff(self, prediction_map: np.ndarray, output_path: str,
                       reference_tiff_path: str = None, crs: str = 'EPSG:32630') -> None:
        """
        Save prediction map as GeoTIFF with proper georeferencing based on reference TIFF.

        Args:
            prediction_map: prediction map to save (H, W)
            output_path: path to save the GeoTIFF
            reference_tiff_path: path to reference TIFF for spatial information
            crs: coordinate reference system (default: UTM Zone 30N)
        """
        logger.info(f"Saving prediction map as GeoTIFF: {output_path}")

        if reference_tiff_path and os.path.exists(reference_tiff_path):
            # Use reference TIFF for spatial information
            with rasterio.open(reference_tiff_path) as ref:
                transform = ref.transform
                crs = ref.crs
                logger.info(f"Using spatial reference from: {reference_tiff_path}")
        else:
            # Create default transform (this should be adjusted based on your data)
            logger.warning("No reference TIFF provided, using default transform")
            # Default transform for UK data (adjust as needed)
            transform = from_bounds(
                west=0, south=50, east=2, north=60,
                width=prediction_map.shape[1], height=prediction_map.shape[0]
            )

        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=prediction_map.shape[0],
            width=prediction_map.shape[1],
            count=1,
            dtype=prediction_map.dtype,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(prediction_map, 1)

        logger.info(f"Saved GeoTIFF: {output_path}")

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
        logger.info(f"Creating prediction visualization: {output_path}")

        # Create visualization array
        # 0: Invalid pixels (gray)
        # 1: Valid non-PV pixels (light blue)
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
        ax.set_title('PV Detection Results (XGBoost)', fontsize=14, fontweight='bold')
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
                     visualize: bool = True, save_geotiff: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Run complete XGBoost inference pipeline.

        Args:
            representation_path: path to representation file
            scales_path: path to scales file
            output_path: base path for output files (without extension)
            reference_tiff_path: path to reference TIFF for spatial information
            visualize: whether to create visualization
            save_geotiff: whether to save as GeoTIFF

        Returns:
            prediction_map: prediction map (H, W)
            probabilities_map: probabilities map (H, W)
            stats: statistics dictionary
        """
        logger.info("Starting XGBoost inference pipeline...")
        start_time = time.time()

        # Load data
        representation, valid_mask, valid_features = self.load_inference_data(representation_path, scales_path)

        # Make predictions on valid pixels only
        predictions, probabilities = self.predict_pixels(valid_features)

        # Create prediction maps
        prediction_map, probabilities_map = self.create_prediction_map(
            predictions, probabilities, valid_mask, representation.shape[:2]
        )

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
        logger.info(f"XGBoost inference pipeline completed in {total_time:.2f} seconds")

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
            
            # Check for required files
            npy_file = os.path.join(grid_dir, f"{grid_name}.npy")
            scales_file = os.path.join(grid_dir, f"{grid_name}_scales.npy")
            tiff_file = os.path.join(grid_dir, f"{grid_name}.tiff")
            
            if os.path.exists(npy_file) and os.path.exists(scales_file):
                grid_info.append({
                    'year': year,
                    'grid_name': grid_name,
                    'grid_dir': grid_dir,
                    'npy_file': npy_file,
                    'scales_file': scales_file,
                    'tiff_file': tiff_file if os.path.exists(tiff_file) else None
                })
            else:
                logger.warning(f"Missing required files in {grid_dir}")
    
    logger.info(f"Found {len(grid_info)} valid grid directories")
    return grid_info

def save_inference_log(output_path: str, grid_info: Dict, stats: Dict, model_path: str) -> None:
    """
    Save inference log with metadata and statistics.
    
    Args:
        output_path: base output path
        grid_info: grid information dictionary
        stats: inference statistics
        model_path: path to the model used
    """
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'model_type': 'xgboost',
        'grid_info': grid_info,
        'statistics': stats
    }
    
    log_path = f"{output_path}_log.json"
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    logger.info(f"Saved inference log: {log_path}")

def process_single_grid(args_tuple) -> Tuple[bool, str, Dict]:
    """
    Process a single grid for parallel inference.
    
    Args:
        args_tuple: tuple containing (grid_info, model_path, output_base_dir, overwrite)
    
    Returns:
        Tuple of (success, grid_name, stats)
    """
    grid_info, model_path, output_base_dir, overwrite = args_tuple
    
    try:
        # Create output directory
        output_dir = os.path.join(output_base_dir, grid_info['year'])
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output path
        output_path = os.path.join(output_dir, f"{grid_info['grid_name']}_xgboost_prediction")
        
        # Check if already processed
        if not overwrite and os.path.exists(f"{output_path}.tiff"):
            logger.info(f"Skipping {grid_info['grid_name']} (already processed)")
            return True, grid_info['grid_name'], {}
        
        # Initialize inference engine
        inference_engine = PVDetectionXGBoostInference(model_path=model_path)
        
        # Run inference
        prediction_map, probabilities_map, stats = inference_engine.run_inference(
            representation_path=grid_info['npy_file'],
            scales_path=grid_info['scales_file'],
            output_path=output_path,
            reference_tiff_path=grid_info['tiff_file'],
            visualize=True,
            save_geotiff=True
        )
        
        # Save log
        save_inference_log(output_path, grid_info, stats, model_path)
        
        logger.info(f"Successfully processed {grid_info['grid_name']}")
        return True, grid_info['grid_name'], stats
        
    except Exception as e:
        logger.error(f"Error processing {grid_info['grid_name']}: {e}")
        return False, grid_info['grid_name'], {}

def run_batch_inference(model_path: str, base_data_dir: str, output_base_dir: str,
                       years: List[str] = None) -> None:
    """
    Run batch inference on multiple grids sequentially.
    
    Args:
        model_path: path to trained XGBoost model
        base_data_dir: base directory containing UK data
        output_base_dir: base directory for output files
        years: list of years to process
    """
    logger.info("Starting batch XGBoost inference...")
    
    # Find all grid directories
    grid_info_list = find_grid_directories(base_data_dir, years)
    
    if not grid_info_list:
        logger.error("No valid grid directories found!")
        return
    
    # Initialize inference engine once
    inference_engine = PVDetectionXGBoostInference(model_path=model_path)
    
    # Process each grid
    successful_inferences = 0
    failed_inferences = 0
    
    with tqdm(total=len(grid_info_list), desc="Processing grids") as pbar:
        for grid_info in grid_info_list:
            try:
                # Create output directory
                output_dir = os.path.join(output_base_dir, grid_info['year'])
                os.makedirs(output_dir, exist_ok=True)
                
                # Define output path
                output_path = os.path.join(output_dir, f"{grid_info['grid_name']}_xgboost_prediction")
                
                # Run inference
                prediction_map, probabilities_map, stats = inference_engine.run_inference(
                    representation_path=grid_info['npy_file'],
                    scales_path=grid_info['scales_file'],
                    output_path=output_path,
                    reference_tiff_path=grid_info['tiff_file'],
                    visualize=True,
                    save_geotiff=True
                )
                
                # Save log
                save_inference_log(output_path, grid_info, stats, model_path)
                
                successful_inferences += 1
                logger.info(f"Successfully processed {grid_info['grid_name']}")
                
            except Exception as e:
                failed_inferences += 1
                logger.error(f"Error processing {grid_info['grid_name']}: {e}")
            
            pbar.update(1)
    
    # Summary
    logger.info("Batch inference completed!")
    logger.info(f"Total grids processed: {len(grid_info_list)}")
    logger.info(f"Successful inferences: {successful_inferences}")
    logger.info(f"Failed inferences: {failed_inferences}")

def run_parallel_inference(model_path: str, base_data_dir: str, output_base_dir: str,
                          years: List[str] = None, num_workers: int = 64, overwrite: bool = False) -> None:
    """
    Run parallel inference on multiple grids.
    
    Args:
        model_path: path to trained XGBoost model
        base_data_dir: base directory containing UK data
        output_base_dir: base directory for output files
        years: list of years to process
        num_workers: number of parallel workers
        overwrite: whether to overwrite existing results
    """
    logger.info(f"Starting parallel XGBoost inference with {num_workers} workers...")
    
    # Find all grid directories
    grid_info_list = find_grid_directories(base_data_dir, years)
    
    if not grid_info_list:
        logger.error("No valid grid directories found!")
        return
    
    # Prepare arguments for parallel processing
    args_list = [(grid_info, model_path, output_base_dir, overwrite) for grid_info in grid_info_list]
    
    # Process grids in parallel
    successful_inferences = 0
    failed_inferences = 0
    total_grids = len(grid_info_list)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_grid = {executor.submit(process_single_grid, args): args[0]['grid_name'] 
                         for args in args_list}
        
        # Process completed tasks with progress bar
        with tqdm(total=total_grids, desc="Processing grids") as pbar:
            for future in as_completed(future_to_grid):
                grid_name = future_to_grid[future]
                try:
                    success, processed_grid_name, stats = future.result()
                    if success:
                        successful_inferences += 1
                    else:
                        failed_inferences += 1
                except Exception as e:
                    failed_inferences += 1
                    logger.error(f"Error in parallel processing of {grid_name}: {e}")
                
                pbar.update(1)
    
    # Summary
    logger.info("Parallel inference completed!")
    logger.info(f"Total grids processed: {total_grids}")
    logger.info(f"Successful inferences: {successful_inferences}")
    logger.info(f"Failed inferences: {failed_inferences}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='XGBoost inference for PV detection')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'parallel'], default='parallel',
                       help='Inference mode: single grid, batch processing, or parallel processing')

    # Model and data paths
    parser.add_argument('--model_path', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection/models/xgboost_model.pkl',
                       help='Path to trained XGBoost model')
    parser.add_argument('--base_data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/uk',
                       help='Base directory containing UK data')
    parser.add_argument('--output_base_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/uk_prediction_xgboost',
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
                       default='/maps/zf281/btfm4rs/src/pv_detection/xgboost_inference_results',
                       help='Base path for output files (single mode only)')
    parser.add_argument('--reference_tiff_path', type=str, 
                       default="/maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2017/grid_0.35_52.25/grid_0.35_52.25.tiff",
                       help='Path to reference TIFF for spatial information (single mode only)')

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
            years=args.years
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
        inference_engine = PVDetectionXGBoostInference(model_path=args.model_path)

        # Run inference
        prediction_map, probabilities_map, stats = inference_engine.run_inference(
            representation_path=args.representation_path,
            scales_path=args.scales_path,
            output_path=args.output_path,
            reference_tiff_path=args.reference_tiff_path,
            visualize=not args.no_visualization,
            save_geotiff=not args.no_geotiff
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

    logger.info("XGBoost inference completed successfully!")

if __name__ == "__main__":
    main()