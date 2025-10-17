import numpy as np
import torch
import torch.nn.functional as F
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
import json

from data_preprocessing import load_and_dequantize_representation, identify_valid_pixels
from patch_data_preprocessing import PatchExtractor
from cnn_models import create_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.optimal_threshold = 0.5
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
        self.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
        
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
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization to: {output_path}")

    def run_inference(self, representation_path: str, scales_path: str,
                     output_path: str, reference_tiff_path: str = None,
                     visualize: bool = True, save_geotiff: bool = True,
                     batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
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

        # Save outputs
        if save_geotiff:
            geotiff_path = f"{output_path}.tiff"
            self.save_as_geotiff(prediction_map, geotiff_path, reference_tiff_path)

        if visualize:
            png_path = f"{output_path}.png"
            self.visualize_predictions(prediction_map, valid_mask, png_path)

        total_time = time.time() - start_time
        logger.info(f"CNN inference pipeline completed in {total_time:.2f} seconds")

        return prediction_map, probabilities_map

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='CNN inference for PV detection')

    # Input/output paths
    parser.add_argument('--model_path', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection/cnn_models/cnn_model.pth',
                       help='Path to trained CNN model')
    parser.add_argument('--representation_path', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile/2024_change_detection_test_tile_map_10m_utm31n_128bands.npy',
                       help='Path to representation file')
    parser.add_argument('--scales_path', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile/2024_change_detection_test_tile_map_10m_utm31n_scales.npy',
                       help='Path to scales file')
    parser.add_argument('--output_path', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection/cnn_inference_results',
                       help='Base path for output files (without extension)')
    parser.add_argument('--reference_tiff_path', type=str, default="/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile/2024_change_detection_test_tile_map_10m_utm31n_128bands.tiff",
                       help='Path to reference TIFF for spatial information')

    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for CNN inference')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'], help='Device to use for inference')

    # Output options
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip creating visualization')
    parser.add_argument('--no_geotiff', action='store_true',
                       help='Skip saving GeoTIFF')

    args = parser.parse_args()

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
    prediction_map, probabilities_map = inference_engine.run_inference(
        representation_path=args.representation_path,
        scales_path=args.scales_path,
        output_path=args.output_path,
        reference_tiff_path=args.reference_tiff_path,
        visualize=not args.no_visualization,
        save_geotiff=not args.no_geotiff,
        batch_size=args.batch_size
    )

    logger.info("CNN inference completed successfully!")

if __name__ == "__main__":
    main()