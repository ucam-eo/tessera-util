import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
import time
import logging
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, Optional
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from data_preprocessing import load_and_dequantize_representation, identify_valid_pixels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PVDetectionInference:
    """
    Inference class for PV detection using trained models.
    """

    def __init__(self, model_dir: str, model_type: str = 'xgboost'):
        """
        Initialize inference engine.

        Args:
            model_dir: directory containing saved models
            model_type: type of model to use ('xgboost' or 'lightgbm')
        """
        self.model_dir = model_dir
        self.model_type = model_type.lower()
        self.model = None
        self.results = None
        self.optimal_threshold = 0.99

        logger.info(f"Initializing PV Detection Inference")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Model type: {model_type}")

        self._load_model()
        self._load_results()

    def _load_model(self) -> None:
        """Load the trained model."""
        if self.model_type == 'xgboost':
            model_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded XGBoost model from {model_path}")
            else:
                raise FileNotFoundError(f"XGBoost model not found at {model_path}")

        elif self.model_type == 'lightgbm':
            model_path = os.path.join(self.model_dir, 'lightgbm_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded LightGBM model from {model_path}")
            else:
                raise FileNotFoundError(f"LightGBM model not found at {model_path}")

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_results(self) -> None:
        """Load training results to get optimal threshold."""
        results_path = os.path.join(self.model_dir, 'results.pkl')
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                self.results = pickle.load(f)

            # Get optimal threshold for the selected model
            if self.model_type in self.results:
                # self.optimal_threshold = self.results[self.model_type].get('best_threshold', 0.99)
                self.optimal_threshold = 0.999
                logger.info(f"Using optimal threshold: {self.optimal_threshold:.3f}")
            else:
                logger.warning(f"No results found for {self.model_type}, using default threshold 0.99")
        else:
            logger.warning(f"Results file not found at {results_path}, using default threshold 0.99")

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

        # Extract valid features
        valid_features = representation[valid_mask]

        logger.info(f"Loaded representation shape: {representation.shape}")
        logger.info(f"Valid pixels: {np.sum(valid_mask):,} / {valid_mask.size:,}")

        return representation, valid_mask, valid_features

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on features.

        Args:
            features: input features array (N, C)

        Returns:
            predictions: binary predictions (N,)
            probabilities: prediction probabilities (N,)
        """
        logger.info(f"Making predictions on {len(features):,} samples...")

        start_time = time.time()

        if self.model_type == 'xgboost':
            # Convert to DMatrix for XGBoost
            dtest = xgb.DMatrix(features)
            probabilities = self.model.predict(dtest)
        elif self.model_type == 'lightgbm':
            probabilities = self.model.predict(features)

        # Apply optimal threshold
        predictions = (probabilities >= self.optimal_threshold).astype(int)

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")

        # Log prediction statistics
        n_positive = np.sum(predictions)
        n_total = len(predictions)
        logger.info(f"Predicted PV pixels: {n_positive:,} / {n_total:,} ({100*n_positive/n_total:.2f}%)")

        return predictions, probabilities

    def create_prediction_map(self, predictions: np.ndarray, valid_mask: np.ndarray,
                            original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create full prediction map from valid pixel predictions.

        Args:
            predictions: predictions for valid pixels (N_valid,)
            valid_mask: boolean mask of valid pixels (H, W)
            original_shape: original shape (H, W)

        Returns:
            prediction_map: full prediction map (H, W)
        """
        prediction_map = np.zeros(original_shape, dtype=np.uint8)
        prediction_map[valid_mask] = predictions
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

                # Calculate pixel size from reference
                pixel_size_x = (ref_bounds.right - ref_bounds.left) / ref_width
                pixel_size_y = (ref_bounds.top - ref_bounds.bottom) / ref_height

                logger.info(f"  Reference pixel size: {pixel_size_x:.6f} x {pixel_size_y:.6f}")

                # Use reference bounds and adjust transform for our array size
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
            dtype=geotiff_array.dtype,
            crs=output_crs,
            transform=transform,
            nodata=nodata_value,
            compress='lzw'  # Add compression to reduce file size
        ) as dst:
            dst.write(geotiff_array, 1)

        logger.info(f"GeoTIFF saved to: {output_path}")
        logger.info(f"  CRS: {output_crs}")
        logger.info(f"  Array shape: {height} x {width}")
        logger.info(f"  NoData value: {nodata_value}")

    def visualize_predictions(self, prediction_map: np.ndarray, valid_mask: np.ndarray,
                            output_path: str, dpi: int = 300) -> None:
        """
        Create and save visualization of predictions.

        Args:
            prediction_map: prediction map (H, W)
            valid_mask: valid pixel mask (H, W)
            output_path: path to save the visualization
            dpi: DPI for saved image
        """
        logger.info(f"Creating visualization...")

        # Create RGB image
        # White background (255, 255, 255)
        # Red for PV pixels (255, 0, 0)
        # Gray for invalid pixels (128, 128, 128)

        height, width = prediction_map.shape
        rgb_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        # Set invalid pixels to gray
        rgb_image[~valid_mask] = [128, 128, 128]

        # Set PV pixels to red
        pv_pixels = (prediction_map == 1) & valid_mask
        rgb_image[pv_pixels] = [255, 0, 0]

        # Save as PNG
        pil_image = Image.fromarray(rgb_image)
        pil_image.save(output_path, dpi=(dpi, dpi))

        logger.info(f"Visualization saved to: {output_path}")

        # Print statistics
        total_pixels = height * width
        valid_pixels = np.sum(valid_mask)
        pv_pixels_count = np.sum(pv_pixels)

        logger.info(f"Image statistics:")
        logger.info(f"  Total pixels: {total_pixels:,}")
        logger.info(f"  Valid pixels: {valid_pixels:,} ({100*valid_pixels/total_pixels:.1f}%)")
        logger.info(f"  PV pixels: {pv_pixels_count:,} ({100*pv_pixels_count/valid_pixels:.1f}% of valid pixels)")

    def run_inference(self, representation_path: str, scales_path: str,
                     output_path: str, reference_tiff_path: str = None,
                     visualize: bool = True, save_geotiff: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run complete inference pipeline.

        Args:
            representation_path: path to representation file
            scales_path: path to scales file
            output_path: path to save results (base path, extensions will be added)
            reference_tiff_path: path to reference TIFF for spatial information
            visualize: whether to create PNG visualization
            save_geotiff: whether to save as GeoTIFF

        Returns:
            prediction_map: full prediction map (H, W)
            probabilities_map: full probabilities map (H, W)
        """
        # Load data
        representation, valid_mask, valid_features = self.load_inference_data(
            representation_path, scales_path
        )

        # Make predictions
        predictions, probabilities = self.predict(valid_features)

        # Create full prediction maps
        prediction_map = self.create_prediction_map(predictions, valid_mask, representation.shape[:2])

        # Create probabilities map
        probabilities_map = np.zeros(representation.shape[:2], dtype=np.float32)
        probabilities_map[valid_mask] = probabilities

        # Save numpy arrays
        base_path = output_path.replace('.png', '').replace('.tiff', '').replace('.tif', '')
        prediction_output_path = f"{base_path}_predictions.npy"
        probabilities_output_path = f"{base_path}_probabilities.npy"

        np.save(prediction_output_path, prediction_map)
        np.save(probabilities_output_path, probabilities_map)

        logger.info(f"Predictions saved to: {prediction_output_path}")
        logger.info(f"Probabilities saved to: {probabilities_output_path}")

        # Create PNG visualization
        if visualize:
            png_path = f"{base_path}_visualization.png"
            self.visualize_predictions(prediction_map, valid_mask, png_path)

        # Save as GeoTIFF
        if save_geotiff:
            tiff_path = f"{base_path}_predictions.tiff"
            self.save_as_geotiff(prediction_map, tiff_path, reference_tiff_path)

        return prediction_map, probabilities_map

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run PV detection inference')
    parser.add_argument('--model_dir', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection/models',
                       help='Directory containing trained models')
    parser.add_argument('--model_type', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm'],
                       help='Type of model to use')
    # parser.add_argument('--data_dir', type=str,
    #                    default='/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile',
    #                    help='Directory containing inference data')
    # parser.add_argument('--representation_file', type=str,
    #                    default='2023_change_detection_test_tile_map_10m_utm31n_128bands.npy',
    #                    help='Representation file name')
    # parser.add_argument('--scales_file', type=str,
    #                    default='2023_change_detection_test_tile_map_10m_utm31n_scales.npy',
    #                    help='Scales file name')
    # parser.add_argument('--reference_tiff', type=str,
    #                    default='/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile/2023_change_detection_test_tile_map_10m_utm31n_128bands.tiff',
    #                    help='Reference TIFF for spatial information')
    # parser.add_argument('--output_path', type=str,
    #                    default='/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile',
    #                    help='Output base path (extensions will be added automatically)')
    parser.add_argument('--data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2023/grid_0.35_52.25/',
                       help='Directory containing inference data')
    parser.add_argument('--representation_file', type=str,
                       default='grid_0.35_52.25.npy',
                       help='Path to representation file (single mode only)')
    parser.add_argument('--scales_file', type=str,
                       default='grid_0.35_52.25_scales.npy',
                       help='Path to scales file (single mode only)')
    parser.add_argument('--output_path', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection/cnn_inference_results',
                       help='Base path for output files (single mode only)')
    parser.add_argument('--reference_tiff', type=str, 
                       default="/maps/zf281/btfm4rs/data/downstream/pv_detection/uk/2023/grid_0.35_52.25/grid_0.35_52.25.tiff",
                       help='Path to reference TIFF for spatial information (single mode only)')
    
    parser.add_argument('--crs', type=str, default='EPSG:32630',
                       help='Coordinate reference system for output TIFF')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved PNG image')
    parser.add_argument('--no_png', action='store_true',
                       help='Skip PNG visualization')
    parser.add_argument('--no_tiff', action='store_true',
                       help='Skip GeoTIFF output')

    args = parser.parse_args()

    # Initialize inference engine
    inference_engine = PVDetectionInference(args.model_dir, args.model_type)

    # Construct file paths
    representation_path = os.path.join(args.data_dir, args.representation_file)
    scales_path = os.path.join(args.data_dir, args.scales_file)

    # Check if files exist
    if not os.path.exists(representation_path):
        raise FileNotFoundError(f"Representation file not found: {representation_path}")
    if not os.path.exists(scales_path):
        raise FileNotFoundError(f"Scales file not found: {scales_path}")

    # Check reference TIFF
    if args.reference_tiff and not os.path.exists(args.reference_tiff):
        logger.warning(f"Reference TIFF not found: {args.reference_tiff}")
        args.reference_tiff = None

    # Run inference
    logger.info("Starting inference...")
    prediction_map, probabilities_map = inference_engine.run_inference(
        representation_path,
        scales_path,
        args.output_path,
        reference_tiff_path=args.reference_tiff,
        visualize=not args.no_png,
        save_geotiff=not args.no_tiff
    )

    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    main()