#!/usr/bin/env python3
"""
UMAP Visualization for Solar Panel Detection
============================================

This script creates UMAP visualizations of 2024 training data embeddings for solar panel detection.
It performs the following steps:
1. Load and dequantize 2024 embeddings
2. Sample solar panels (all) vs others (10x random sampling)
3. Apply UMAP dimensionality reduction with optimized parameters
4. Create Nature journal-standard visualization with transparency and density

Author: Generated for PV Detection Project
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import umap
import os
import logging
import time
from typing import Tuple, Dict, Any
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from data_preprocessing import load_and_dequantize_representation, identify_valid_pixels, extract_valid_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('umap_visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UMAPVisualizer:
    """
    UMAP Visualizer for Solar Panel Detection Data
    """
    
    def __init__(self, data_dir: str, output_dir: str, year: int = 2024, random_seed: int = 42):
        """
        Initialize UMAP Visualizer
        
        Args:
            data_dir: Directory containing the data files
            output_dir: Directory to save visualization outputs
            year: Year of data to visualize (default: 2024)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.year = year
        self.random_seed = random_seed
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.embeddings = None
        self.labels = None
        self.coords = None
        self.sampled_embeddings = None
        self.sampled_labels = None
        self.umap_embeddings = None
        
        logger.info(f"Initialized UMAP Visualizer")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Target year: {year}")
        logger.info(f"Random seed: {random_seed}")
        
        # Set random seeds
        np.random.seed(random_seed)
        
    def load_data(self) -> None:
        """Load and dequantize embeddings data for the specified year"""
        logger.info(f"Loading data for year {self.year}...")
        
        # File paths
        representation_path = os.path.join(self.data_dir, f"{self.year}_change_detection_test_tile_map_10m_utm31n_128bands.npy")
        scales_path = os.path.join(self.data_dir, f"{self.year}_change_detection_test_tile_map_10m_utm31n_scales.npy")
        labels_path = os.path.join(self.data_dir, "2024_change_detection_test_tile_labels.npy")
        
        # Check if files exist
        for path, name in [(representation_path, "representation"), (scales_path, "scales"), (labels_path, "labels")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name.capitalize()} file not found: {path}")
        
        logger.info("Loading and dequantizing representation...")
        start_time = time.time()
        
        # Load and dequantize representation
        representation = load_and_dequantize_representation(representation_path, scales_path)
        logger.info(f"Representation loaded in {time.time() - start_time:.2f}s")
        logger.info(f"Representation shape: {representation.shape}")
        
        # Load labels
        logger.info("Loading labels...")
        labels = np.load(labels_path)
        logger.info(f"Labels shape: {labels.shape}")
        
        # Identify valid pixels (non-zero embeddings)
        logger.info("Identifying valid pixels...")
        valid_mask = identify_valid_pixels(representation)
        
        # Extract valid data
        logger.info("Extracting valid data...")
        valid_features, valid_labels, valid_coords = extract_valid_data(representation, labels, valid_mask)
        
        self.embeddings = valid_features
        self.labels = valid_labels
        self.coords = valid_coords
        
        logger.info(f"Data loading completed:")
        logger.info(f"  Total valid samples: {len(self.embeddings):,}")
        logger.info(f"  Solar panel samples: {np.sum(self.labels == 1):,}")
        logger.info(f"  Other samples: {np.sum(self.labels == 0):,}")
        logger.info(f"  Embedding dimension: {self.embeddings.shape[1]}")
        
    def sample_data(self, others_multiplier: int = 10, max_solar_samples: int = None) -> None:
        """
        Sample data for visualization: solar panels + random others
        
        Args:
            others_multiplier: Multiplier for others samples relative to solar panels
            max_solar_samples: Maximum number of solar panel samples to use (default: None, use all)
        """
        logger.info(f"Sampling data for visualization...")
        
        # Get indices for each class
        solar_indices = np.where(self.labels == 1)[0]
        other_indices = np.where(self.labels == 0)[0]
        
        n_solar_available = len(solar_indices)
        n_others_available = len(other_indices)
        
        # Determine number of solar samples to use
        if max_solar_samples is None:
            n_solar = n_solar_available
            selected_solar_indices = solar_indices
            logger.info(f"Solar panel samples: {n_solar:,} (using all)")
        else:
            n_solar = min(max_solar_samples, n_solar_available)
            selected_solar_indices = np.random.choice(solar_indices, size=n_solar, replace=False)
            logger.info(f"Solar panel samples: {n_solar:,} (limited from {n_solar_available:,})")
        
        # Calculate target number of others samples
        n_others_target = n_solar * others_multiplier
        
        logger.info(f"Other samples available: {n_others_available:,}")
        logger.info(f"Other samples target: {n_others_target:,}")
        
        # Sample others randomly
        if n_others_target > n_others_available:
            logger.warning(f"Not enough other samples available. Using all {n_others_available:,} samples.")
            selected_other_indices = other_indices
        else:
            selected_other_indices = np.random.choice(other_indices, size=n_others_target, replace=False)
        
        # Combine indices
        all_selected_indices = np.concatenate([selected_solar_indices, selected_other_indices])
        
        # Shuffle to mix classes
        np.random.shuffle(all_selected_indices)
        
        # Extract sampled data
        self.sampled_embeddings = self.embeddings[all_selected_indices]
        self.sampled_labels = self.labels[all_selected_indices]
        
        logger.info(f"Sampling completed:")
        logger.info(f"  Total samples for UMAP: {len(self.sampled_embeddings):,}")
        logger.info(f"  Solar panels: {np.sum(self.sampled_labels == 1):,}")
        logger.info(f"  Others: {np.sum(self.sampled_labels == 0):,}")
        
    def fit_umap(self, n_neighbors: int = 50, min_dist: float = 0.1, n_epochs: int = 200, 
                 metric: str = 'euclidean', n_jobs: int = 64) -> None:
        """
        Fit UMAP model with optimized parameters for better clustering
        
        Args:
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            n_epochs: Number of training epochs
            metric: Distance metric
            n_jobs: Number of parallel jobs (utilize 64 CPU cores)
        """
        logger.info(f"Fitting UMAP model...")
        logger.info(f"Parameters:")
        logger.info(f"  n_neighbors: {n_neighbors}")
        logger.info(f"  min_dist: {min_dist}")
        logger.info(f"  n_epochs: {n_epochs}")
        logger.info(f"  metric: {metric}")
        logger.info(f"  n_jobs: {n_jobs}")
        
        start_time = time.time()
        
        # Initialize UMAP with optimized parameters for better clustering
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            n_epochs=n_epochs,
            metric=metric,
            random_state=self.random_seed,
            n_jobs=n_jobs,
            verbose=True
        )
        
        # Fit and transform
        logger.info("Starting UMAP transformation...")
        self.umap_embeddings = umap_model.fit_transform(self.sampled_embeddings)
        
        fit_time = time.time() - start_time
        logger.info(f"UMAP fitting completed in {fit_time:.2f}s")
        logger.info(f"UMAP embeddings shape: {self.umap_embeddings.shape}")
        
        # Save UMAP model and embeddings
        model_path = os.path.join(self.output_dir, f'umap_model_{self.year}.pkl')
        embeddings_path = os.path.join(self.output_dir, f'umap_embeddings_{self.year}.npy')
        
        with open(model_path, 'wb') as f:
            pickle.dump(umap_model, f)
        np.save(embeddings_path, self.umap_embeddings)
        
        logger.info(f"UMAP model saved to: {model_path}")
        logger.info(f"UMAP embeddings saved to: {embeddings_path}")
        
    def create_visualization(self, figsize: Tuple[int, int] = (12, 10), dpi: int = 300,
                           alpha_solar: float = 0.8, alpha_others: float = 0.3,
                           s_solar: float = 1.0, s_others: float = 0.5) -> None:
        """
        Create Nature journal-standard visualization
        
        Args:
            figsize: Figure size
            dpi: DPI for high-quality output
            alpha_solar: Transparency for solar panel points
            alpha_others: Transparency for other points
            s_solar: Size for solar panel points
            s_others: Size for other points
        """
        logger.info("Creating visualization...")
        
        # Set style for Nature journal standards
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Separate data by class
        solar_mask = self.sampled_labels == 1
        others_mask = self.sampled_labels == 0
        
        solar_points = self.umap_embeddings[solar_mask]
        others_points = self.umap_embeddings[others_mask]
        
        logger.info(f"Plotting {len(others_points):,} other samples...")
        logger.info(f"Plotting {len(solar_points):,} solar panel samples...")
        
        # Plot others first (background)
        scatter_others = ax.scatter(
            others_points[:, 0], others_points[:, 1],
            c='#1f77b4',  # Blue
            alpha=alpha_others,
            s=s_others,
            label=f'Others (n={len(others_points):,})',
            rasterized=True  # For better PDF rendering
        )
        
        # Plot solar panels on top (foreground)
        scatter_solar = ax.scatter(
            solar_points[:, 0], solar_points[:, 1],
            c='#ff7f0e',  # Orange
            alpha=alpha_solar,
            s=s_solar,
            label=f'Solar Panels (n={len(solar_points):,})',
            rasterized=True  # For better PDF rendering
        )
        
        # Remove axes (as requested)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add legend with Nature journal style
        legend = ax.legend(
            loc='upper right',
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=0.9,
            edgecolor='black',
            fontsize=12
        )
        legend.get_frame().set_linewidth(0.5)
        
        # Set title
        ax.set_title(
            f'UMAP Visualization of Solar Panel Detection Features ({self.year})',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Save high-quality figures
        output_formats = ['png', 'pdf', 'svg']
        for fmt in output_formats:
            output_path = os.path.join(self.output_dir, f'umap_visualization_{self.year}.{fmt}')
            plt.savefig(
                output_path,
                format=fmt,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            logger.info(f"Visualization saved: {output_path}")
        
        plt.show()
        
    def create_density_visualization(self, figsize: Tuple[int, int] = (12, 10), dpi: int = 300) -> None:
        """
        Create density-based visualization to better show clustering patterns
        
        Args:
            figsize: Figure size
            dpi: DPI for high-quality output
        """
        logger.info("Creating density visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]), dpi=dpi)
        
        # Separate data by class
        solar_mask = self.sampled_labels == 1
        others_mask = self.sampled_labels == 0
        
        solar_points = self.umap_embeddings[solar_mask]
        others_points = self.umap_embeddings[others_mask]
        
        # Density plot for others
        ax1.hexbin(others_points[:, 0], others_points[:, 1], 
                  gridsize=50, cmap='Blues', alpha=0.8, mincnt=1)
        ax1.set_title('Others Density Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Density plot for solar panels
        ax2.hexbin(solar_points[:, 0], solar_points[:, 1], 
                  gridsize=50, cmap='Oranges', alpha=0.8, mincnt=1)
        ax2.set_title('Solar Panels Density Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Remove spines
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        
        # Save density visualization
        output_formats = ['png', 'pdf']
        for fmt in output_formats:
            output_path = os.path.join(self.output_dir, f'umap_density_{self.year}.{fmt}')
            plt.savefig(
                output_path,
                format=fmt,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            logger.info(f"Density visualization saved: {output_path}")
        
        plt.show()
        
    def save_statistics(self) -> None:
        """Save visualization statistics and metadata"""
        logger.info("Saving statistics...")
        
        stats = {
            'year': self.year,
            'total_samples': len(self.sampled_embeddings),
            'solar_panels': int(np.sum(self.sampled_labels == 1)),
            'others': int(np.sum(self.sampled_labels == 0)),
            'embedding_dimension': self.embeddings.shape[1],
            'umap_dimension': 2,
            'random_seed': self.random_seed,
            'umap_range_x': [float(self.umap_embeddings[:, 0].min()), float(self.umap_embeddings[:, 0].max())],
            'umap_range_y': [float(self.umap_embeddings[:, 1].min()), float(self.umap_embeddings[:, 1].max())],
        }
        
        stats_path = os.path.join(self.output_dir, f'visualization_stats_{self.year}.json')
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved: {stats_path}")
        
    def run_full_pipeline(self, others_multiplier: int = 10, max_solar_samples: int = None,
                         n_neighbors: int = 50, min_dist: float = 0.1, n_epochs: int = 200) -> None:
        """
        Run the complete UMAP visualization pipeline
        
        Args:
            others_multiplier: Multiplier for others samples relative to solar panels
            max_solar_samples: Maximum number of solar panel samples to use (default: None, use all)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            n_epochs: UMAP n_epochs parameter
        """
        logger.info("Starting full UMAP visualization pipeline...")
        start_time = time.time()
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Sample data
        self.sample_data(others_multiplier=others_multiplier, max_solar_samples=max_solar_samples)
        
        # Step 3: Fit UMAP
        self.fit_umap(n_neighbors=n_neighbors, min_dist=min_dist, n_epochs=n_epochs)
        
        # Step 4: Create visualizations
        self.create_visualization()
        self.create_density_visualization()
        
        # Step 5: Save statistics
        self.save_statistics()
        
        total_time = time.time() - start_time
        logger.info(f"Full pipeline completed in {total_time:.2f}s")
        logger.info(f"All outputs saved to: {self.output_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create UMAP visualization for solar panel detection')
    parser.add_argument('--data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile',
                       help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection',
                       help='Directory to save visualization outputs')
    parser.add_argument('--year', type=int, default=2024,
                       help='Year of data to visualize')
    parser.add_argument('--others_multiplier', type=int, default=20,
                       help='Multiplier for others samples relative to solar panels')
    parser.add_argument('--max_solar_samples', type=int, default=500000,
                       help='Maximum number of solar panel samples to use (default: None, use all)')
    parser.add_argument('--n_neighbors', type=int, default=1000,
                       help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.01,
                       help='UMAP min_dist parameter')
    parser.add_argument('--n_epochs', type=int, default=1000,
                       help='UMAP n_epochs parameter')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = UMAPVisualizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        year=args.year,
        random_seed=args.random_seed
    )
    
    # Run full pipeline
    visualizer.run_full_pipeline(
        others_multiplier=args.others_multiplier,
        max_solar_samples=args.max_solar_samples,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_epochs=args.n_epochs
    )
    
    logger.info("UMAP visualization completed successfully!")


if __name__ == "__main__":
    main()