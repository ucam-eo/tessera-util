#!/usr/bin/env python3
"""
Enhanced PCA Visualization for Solar Panel Detection
===================================================

This enhanced script creates optimized PCA visualizations specifically designed 
to improve solar panel clustering. It includes:

1. PCA dimensionality reduction to 2D for visualization
2. Advanced preprocessing (standardization, outlier detection)
3. Parameter optimization for PCA settings
4. KMeans clustering algorithm for fast performance
5. Clustering quality evaluation metrics
6. Solar panel-specific optimization strategies

Author: Enhanced for PV Detection Project
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import logging
import time
from typing import Tuple, Dict, Any, List, Optional
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Advanced ML libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import ParameterGrid
import itertools

# Import local modules
from data_preprocessing import load_and_dequantize_representation, identify_valid_pixels, extract_valid_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pca_visualization_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedPCAVisualizer:
    """
    Enhanced PCA Visualizer with advanced clustering optimization for Solar Panel Detection
    """
    
    def __init__(self, data_dir: str, output_dir: str, year: int = 2024, random_seed: int = 42):
        """
        Initialize Enhanced PCA Visualizer
        
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
        self.preprocessed_embeddings = None
        self.pca_embeddings = None
        self.cluster_labels = None
        
        # Initialize models
        self.scaler = None
        self.pca_model = None
        self.best_params = None
        self.clustering_results = {}
        
        logger.info(f"Initialized Enhanced PCA Visualizer")
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
        
        logger.info("Loading representation data...")
        self.embeddings = load_and_dequantize_representation(representation_path, scales_path)
        logger.info(f"Loaded embeddings shape: {self.embeddings.shape}")
        
        logger.info("Loading labels...")
        self.labels = np.load(labels_path)
        logger.info(f"Loaded labels shape: {self.labels.shape}")
        
        # Identify valid pixels
        logger.info("Identifying valid pixels...")
        valid_mask = identify_valid_pixels(self.embeddings)
        logger.info(f"Valid pixels: {np.sum(valid_mask):,} / {len(valid_mask):,}")
        
        # Extract valid data
        self.embeddings, self.labels, self.coords = extract_valid_data(
            self.embeddings, self.labels, valid_mask
        )
        
        logger.info(f"Final data shapes:")
        logger.info(f"  Embeddings: {self.embeddings.shape}")
        logger.info(f"  Labels: {self.labels.shape}")
        logger.info(f"  Coordinates: {self.coords.shape}")
        
        # Data statistics
        solar_count = np.sum(self.labels == 1)
        others_count = np.sum(self.labels == 0)
        logger.info(f"Data distribution:")
        logger.info(f"  Solar panels: {solar_count:,} ({solar_count/len(self.labels)*100:.2f}%)")
        logger.info(f"  Others: {others_count:,} ({others_count/len(self.labels)*100:.2f}%)")
        
    def sample_data(self, others_multiplier: int = 10, max_solar_samples: int = None) -> None:
        """
        Sample data with enhanced strategy for better solar panel representation
        
        Args:
            others_multiplier: Multiplier for others samples relative to solar panels
            max_solar_samples: Maximum number of solar panel samples (None for all)
        """
        logger.info(f"Sampling data with enhanced strategy...")
        
        # Get indices for each class
        solar_indices = np.where(self.labels == 1)[0]
        others_indices = np.where(self.labels == 0)[0]
        
        logger.info(f"Available samples:")
        logger.info(f"  Solar panels: {len(solar_indices):,}")
        logger.info(f"  Others: {len(others_indices):,}")
        
        # Sample solar panels (use all or limit if specified)
        if max_solar_samples is not None and len(solar_indices) > max_solar_samples:
            sampled_solar_indices = np.random.choice(solar_indices, max_solar_samples, replace=False)
        else:
            sampled_solar_indices = solar_indices
        
        # Sample others (balanced sampling)
        n_others_samples = min(len(sampled_solar_indices) * others_multiplier, len(others_indices))
        sampled_others_indices = np.random.choice(others_indices, n_others_samples, replace=False)
        
        # Combine samples
        all_sampled_indices = np.concatenate([sampled_solar_indices, sampled_others_indices])
        
        # Extract sampled data
        self.sampled_embeddings = self.embeddings[all_sampled_indices]
        self.sampled_labels = self.labels[all_sampled_indices]
        
        logger.info(f"Sampling completed:")
        logger.info(f"  Total samples for analysis: {len(self.sampled_embeddings):,}")
        logger.info(f"  Solar panels: {np.sum(self.sampled_labels == 1):,}")
        logger.info(f"  Others: {np.sum(self.sampled_labels == 0):,}")
        
    def preprocess_data(self, use_robust_scaler: bool = True, remove_outliers: bool = False, 
                       outlier_contamination: float = 0.1) -> None:
        """
        Advanced data preprocessing for better clustering
        
        Args:
            use_robust_scaler: Use RobustScaler instead of StandardScaler
            remove_outliers: Remove outliers using LocalOutlierFactor
            outlier_contamination: Expected proportion of outliers
        """
        logger.info("Starting advanced data preprocessing...")
        
        # Step 1: Feature scaling
        if use_robust_scaler:
            self.scaler = RobustScaler()
            logger.info("Using RobustScaler for feature scaling...")
        else:
            self.scaler = StandardScaler()
            logger.info("Using StandardScaler for feature scaling...")
        
        scaled_embeddings = self.scaler.fit_transform(self.sampled_embeddings)
        
        # Step 2: Outlier detection and removal
        if remove_outliers:
            logger.info(f"Detecting outliers with contamination={outlier_contamination}...")
            lof = LocalOutlierFactor(contamination=outlier_contamination, n_jobs=-1)
            outlier_labels = lof.fit_predict(scaled_embeddings)
            
            # Keep only inliers (outlier_labels == 1)
            inlier_mask = outlier_labels == 1
            self.preprocessed_embeddings = scaled_embeddings[inlier_mask]
            self.sampled_labels = self.sampled_labels[inlier_mask]
            
            n_outliers = np.sum(~inlier_mask)
            logger.info(f"Removed {n_outliers:,} outliers ({n_outliers/len(scaled_embeddings)*100:.2f}%)")
        else:
            self.preprocessed_embeddings = scaled_embeddings
        
        logger.info(f"Preprocessed data shape: {self.preprocessed_embeddings.shape}")
        logger.info(f"Final solar panels: {np.sum(self.sampled_labels == 1):,}")
        logger.info(f"Final others: {np.sum(self.sampled_labels == 0):,}")
        
    def apply_pca(self, n_components: int = 2, explained_variance_threshold: float = 0.95) -> None:
        """
        Apply PCA for dimensionality reduction to 2D for visualization
        
        Args:
            n_components: Number of PCA components (default: 2 for visualization)
            explained_variance_threshold: Minimum explained variance ratio for reporting
        """
        logger.info("Applying PCA for dimensionality reduction to 2D...")
        
        try:
            # Ensure n_components is not larger than min(n_samples, n_features)
            max_components = min(self.preprocessed_embeddings.shape[0], self.preprocessed_embeddings.shape[1])
            n_components = min(n_components, max_components)
            
            logger.info(f"Using {n_components} PCA components")
            
            # Apply PCA
            self.pca_model = PCA(n_components=n_components, random_state=self.random_seed)
            self.pca_embeddings = self.pca_model.fit_transform(self.preprocessed_embeddings)
            
            # Calculate explained variance
            explained_variance_ratio = self.pca_model.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            logger.info(f"PCA Results:")
            logger.info(f"  Components: {n_components}")
            logger.info(f"  Explained variance per component: {explained_variance_ratio}")
            logger.info(f"  Total explained variance: {cumulative_variance[-1]:.4f}")
            
            # Report if we meet the threshold
            if cumulative_variance[-1] >= explained_variance_threshold:
                logger.info(f"✓ Explained variance threshold ({explained_variance_threshold}) met")
            else:
                logger.warning(f"⚠ Explained variance ({cumulative_variance[-1]:.4f}) below threshold ({explained_variance_threshold})")
            
            logger.info(f"PCA embeddings shape: {self.pca_embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error in PCA: {e}")
            raise
    
    def optimize_pca_parameters(self, max_components: int = 10) -> Dict:
        """
        Optimize PCA parameters by testing different numbers of components
        
        Args:
            max_components: Maximum number of components to test
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Optimizing PCA parameters...")
        
        # Limit max_components to data constraints
        data_max_components = min(self.preprocessed_embeddings.shape[0], self.preprocessed_embeddings.shape[1])
        max_components = min(max_components, data_max_components)
        
        results = {}
        component_range = range(2, max_components + 1)
        
        for n_comp in tqdm(component_range, desc="Testing PCA components"):
            try:
                # Fit PCA
                pca = PCA(n_components=n_comp, random_state=self.random_seed)
                embeddings_2d = pca.fit_transform(self.preprocessed_embeddings)
                
                # For visualization, we only use first 2 components
                embeddings_viz = embeddings_2d[:, :2]
                
                # Calculate clustering quality metrics using 2D representation
                silhouette = silhouette_score(embeddings_viz, self.sampled_labels)
                calinski = calinski_harabasz_score(embeddings_viz, self.sampled_labels)
                davies_bouldin = davies_bouldin_score(embeddings_viz, self.sampled_labels)
                
                results[n_comp] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'cumulative_variance': np.sum(pca.explained_variance_ratio_),
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'davies_bouldin_score': davies_bouldin
                }
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {n_comp} components: {e}")
                continue
        
        # Find best parameters based on silhouette score
        if results:
            best_n_comp = max(results.keys(), key=lambda k: results[k]['silhouette_score'])
            self.best_params = {'n_components': best_n_comp}
            
            logger.info(f"Best PCA parameters found:")
            logger.info(f"  Components: {best_n_comp}")
            logger.info(f"  Silhouette Score: {results[best_n_comp]['silhouette_score']:.4f}")
            logger.info(f"  Explained Variance: {results[best_n_comp]['cumulative_variance']:.4f}")
        
        return results
    
    def fit_pca(self, params: Dict = None) -> None:
        """
        Fit PCA with specified or optimized parameters
        
        Args:
            params: PCA parameters dictionary
        """
        if params is None:
            params = self.best_params if self.best_params else {'n_components': 2}
        
        logger.info(f"Fitting PCA with parameters: {params}")
        
        # Apply PCA with specified components, but extract only 2D for visualization
        n_components = params.get('n_components', 2)
        
        # Fit PCA with the specified number of components
        self.pca_model = PCA(n_components=n_components, random_state=self.random_seed)
        full_embeddings = self.pca_model.fit_transform(self.preprocessed_embeddings)
        
        # Extract first 2 components for visualization
        self.pca_embeddings = full_embeddings[:, :2]
        
        logger.info(f"PCA fitted successfully")
        logger.info(f"  Total components: {n_components}")
        logger.info(f"  Visualization components: 2")
        logger.info(f"  Explained variance (first 2): {np.sum(self.pca_model.explained_variance_ratio_[:2]):.4f}")
        logger.info(f"  Total explained variance: {np.sum(self.pca_model.explained_variance_ratio_):.4f}")
    
    def apply_clustering_algorithms(self, algorithms: List[str] = None) -> Dict:
        """
        Apply KMeans clustering to PCA embeddings
        
        Args:
            algorithms: List of algorithm names to apply (only 'kmeans' supported)
            
        Returns:
            Dictionary with clustering results
        """
        if algorithms is None:
            algorithms = ['kmeans']
        
        logger.info(f"Applying clustering algorithms: {algorithms}")
        
        results = {}
        
        for algorithm in algorithms:
            logger.info(f"Applying {algorithm.upper()} clustering...")
            
            try:
                if algorithm == 'kmeans':
                    results[algorithm] = self._apply_kmeans()
                else:
                    logger.warning(f"Unknown algorithm: {algorithm}. Only 'kmeans' is supported.")
                    continue
                    
                logger.info(f"{algorithm.upper()} completed successfully")
                
            except Exception as e:
                logger.error(f"Error in {algorithm}: {e}")
                results[algorithm] = {'error': str(e)}
        
        self.clustering_results = results
        return results
    
    def _apply_kmeans(self) -> Dict:
        """Apply KMeans clustering"""
        # Use a reasonable number of clusters based on data
        n_clusters = min(8, len(np.unique(self.sampled_labels)) * 2)
        
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_seed,
            n_init=10
        )
        cluster_labels = clusterer.fit_predict(self.pca_embeddings)
        
        return {
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': 0,
            'algorithm': 'KMeans'
        }
    
    def evaluate_clustering_quality(self) -> Dict:
        """
        Evaluate clustering quality using multiple metrics
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating clustering quality...")
        
        evaluation_results = {}
        
        for algorithm, result in self.clustering_results.items():
            if 'error' in result:
                continue
                
            cluster_labels = result['labels']
            
            try:
                # Calculate metrics
                silhouette = silhouette_score(self.pca_embeddings, cluster_labels)
                calinski = calinski_harabasz_score(self.pca_embeddings, cluster_labels)
                davies_bouldin = davies_bouldin_score(self.pca_embeddings, cluster_labels)
                
                # Calculate solar panel clustering effectiveness
                solar_mask = self.sampled_labels == 1
                if np.sum(solar_mask) > 0:
                    solar_clusters = cluster_labels[solar_mask]
                    unique_solar_clusters = len(np.unique(solar_clusters))
                    solar_cluster_purity = self._calculate_cluster_purity(cluster_labels, self.sampled_labels)
                else:
                    unique_solar_clusters = 0
                    solar_cluster_purity = 0
                
                evaluation_results[algorithm] = {
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'davies_bouldin_score': davies_bouldin,
                    'solar_clusters': unique_solar_clusters,
                    'cluster_purity': solar_cluster_purity,
                    'n_clusters': result['n_clusters'],
                    'n_noise': result.get('n_noise', 0)
                }
                
                logger.info(f"{algorithm.upper()} Quality Metrics:")
                logger.info(f"  Silhouette Score: {silhouette:.4f}")
                logger.info(f"  Calinski-Harabasz: {calinski:.2f}")
                logger.info(f"  Davies-Bouldin: {davies_bouldin:.4f}")
                logger.info(f"  Solar Panel Clusters: {unique_solar_clusters}")
                
            except Exception as e:
                logger.error(f"Error evaluating {algorithm}: {e}")
                evaluation_results[algorithm] = {'error': str(e)}
        
        return evaluation_results
    
    def _calculate_cluster_purity(self, cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate cluster purity score"""
        total_samples = len(cluster_labels)
        if total_samples == 0:
            return 0.0
        
        correct_assignments = 0
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) == 0:
                continue
            
            # Find the most common true label in this cluster
            cluster_true_labels = true_labels[cluster_mask]
            most_common_label = np.bincount(cluster_true_labels).argmax()
            
            # Count correct assignments
            correct_assignments += np.sum(cluster_true_labels == most_common_label)
        
        return correct_assignments / total_samples
    
    def create_simple_visualization(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Create simple 2D PCA visualization focused on class separation
        
        Args:
            figsize: Figure size for the plot
        """
        logger.info("Creating simple 2D PCA visualization...")
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Separate different classes
        solar_mask = self.sampled_labels == 1
        others_mask = self.sampled_labels == 0
        
        solar_points = self.pca_embeddings[solar_mask]
        others_points = self.pca_embeddings[others_mask]
        
        # Plot scatter points
        ax.scatter(others_points[:, 0], others_points[:, 1], 
                  c='lightblue', alpha=0.6, s=20, label=f'Others ({len(others_points):,})', 
                  edgecolors='none')
        
        ax.scatter(solar_points[:, 0], solar_points[:, 1], 
                  c='red', alpha=0.8, s=25, label=f'Solar Panels ({len(solar_points):,})', 
                  edgecolors='darkred', linewidths=0.5)
        
        # Set labels and title
        ax.set_xlabel('First Principal Component', fontsize=12)
        ax.set_ylabel('Second Principal Component', fontsize=12)
        ax.set_title(f'PCA Visualization - Solar Panel Detection ({self.year})', 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f'pca_visualization_simple_{self.year}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Simple visualization saved: {output_path}")
        
        plt.show()
    
    def _plot_pca_scatter(self, ax, labels, title, colors=None, labels_legend=None):
        """Create a scatter plot of PCA embeddings"""
        unique_labels = np.unique(labels)
        
        if colors is None:
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = colors[i] if i < len(colors) else colors[i % len(colors)]
            
            if labels_legend and i < len(labels_legend):
                legend_label = labels_legend[i]
            else:
                legend_label = f'Cluster {label}' if label != -1 else 'Noise'
            
            ax.scatter(self.pca_embeddings[mask, 0], self.pca_embeddings[mask, 1], 
                      c=[color], alpha=0.6, s=20, label=legend_label)
        
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_explained_variance(self, ax):
        """Plot explained variance ratio"""
        if self.pca_model is not None:
            explained_var = self.pca_model.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            x = range(1, len(explained_var) + 1)
            ax.bar(x, explained_var, alpha=0.7, label='Individual')
            ax.plot(x, cumulative_var, 'ro-', label='Cumulative')
            
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Explained Variance')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_solar_panel_focus(self, ax):
        """Create a focused plot highlighting solar panels"""
        solar_mask = self.sampled_labels == 1
        others_mask = self.sampled_labels == 0
        
        # Plot others in background
        ax.scatter(self.pca_embeddings[others_mask, 0], self.pca_embeddings[others_mask, 1], 
                  c='lightgray', alpha=0.3, s=10, label='Others')
        
        # Highlight solar panels
        ax.scatter(self.pca_embeddings[solar_mask, 0], self.pca_embeddings[solar_mask, 1], 
                  c='red', alpha=0.8, s=30, label='Solar Panels')
        
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_title('Solar Panel Focus')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_density_map(self, ax):
        """Create a density map of the PCA embeddings"""
        from scipy.stats import gaussian_kde
        
        # Create density estimation
        xy = self.pca_embeddings.T
        density = gaussian_kde(xy)
        
        # Create grid for density plot
        x_min, x_max = self.pca_embeddings[:, 0].min(), self.pca_embeddings[:, 0].max()
        y_min, y_max = self.pca_embeddings[:, 1].min(), self.pca_embeddings[:, 1].max()
        
        xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = density(positions).reshape(xx.shape)
        
        # Plot density
        ax.contourf(xx, yy, f, levels=20, alpha=0.6, cmap='viridis')
        ax.contour(xx, yy, f, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_title('Data Density Map')
    
    def _plot_quality_metrics(self, ax):
        """Plot clustering quality metrics comparison"""
        if not hasattr(self, 'evaluation_results'):
            self.evaluation_results = self.evaluate_clustering_quality()
        
        algorithms = []
        silhouette_scores = []
        
        for algorithm, metrics in self.evaluation_results.items():
            if 'error' not in metrics:
                algorithms.append(algorithm.upper())
                silhouette_scores.append(metrics['silhouette_score'])
        
        if algorithms:
            bars = ax.bar(algorithms, silhouette_scores, alpha=0.7)
            ax.set_ylabel('Silhouette Score')
            ax.set_title('Clustering Quality Comparison')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, silhouette_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_component_analysis(self, ax):
        """Plot component loadings analysis"""
        if self.pca_model is not None and hasattr(self.pca_model, 'components_'):
            components = self.pca_model.components_[:2]  # First 2 components
            
            # Create heatmap of component loadings
            im = ax.imshow(components, cmap='RdBu_r', aspect='auto')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Principal Component')
            ax.set_title('PCA Component Loadings')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['PC1', 'PC2'])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_clustering_comparison(self, ax):
        """Plot clustering results comparison"""
        n_algorithms = len(self.clustering_results)
        if n_algorithms == 0:
            return
        
        # Create subplots for each algorithm
        for i, (algorithm, result) in enumerate(self.clustering_results.items()):
            if 'error' in result:
                continue
            
            # Use the main ax for the first algorithm, create insets for others
            if i == 0:
                current_ax = ax
            else:
                # Create inset axes
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                current_ax = inset_axes(ax, width="30%", height="30%", 
                                      loc=2+i, borderpad=2)
            
            self._plot_pca_scatter(current_ax, result['labels'], 
                                 f"{result['algorithm']}")
            
            if i > 0:
                current_ax.set_xlabel('')
                current_ax.set_ylabel('')
                current_ax.tick_params(labelsize=8)
    
    def _create_individual_plots(self) -> None:
        """Create individual detailed plots"""
        logger.info("Creating individual detailed plots...")
        
        # 1. Detailed original vs clustered comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original labels
        self._plot_pca_scatter(axes[0, 0], self.sampled_labels, "Original Labels",
                              colors=['#FF6B6B', '#4ECDC4'], labels=['Others', 'Solar Panels'])
        
        # Best clustering result
        if self.clustering_results:
            best_algorithm = max(self.clustering_results.keys(), 
                               key=lambda k: self.evaluation_results.get(k, {}).get('silhouette_score', 0))
            best_result = self.clustering_results[best_algorithm]
            self._plot_pca_scatter(axes[0, 1], best_result['labels'], 
                                 f"Best Clustering ({best_result['algorithm']})")
        
        # Solar panel distribution
        self._plot_solar_panel_focus(axes[1, 0])
        
        # Density map
        self._plot_density_map(axes[1, 1])
        
        plt.tight_layout()
        individual_path = os.path.join(self.output_dir, f'pca_detailed_analysis_{self.year}.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        logger.info(f"Detailed analysis saved: {individual_path}")
        plt.show()
        
        # 2. Component analysis plot
        if self.pca_model is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Explained variance
            self._plot_explained_variance(ax1)
            
            # Component loadings
            self._plot_component_analysis(ax2)
            
            plt.tight_layout()
            components_path = os.path.join(self.output_dir, f'pca_components_analysis_{self.year}.png')
            plt.savefig(components_path, dpi=300, bbox_inches='tight')
            logger.info(f"Components analysis saved: {components_path}")
            plt.show()
    
    def run_simple_pipeline(self, others_multiplier: int = 10, 
                          max_solar_samples: int = None,
                          use_robust_scaler: bool = True) -> Dict:
        """
        Run simplified PCA visualization pipeline
        
        Args:
            others_multiplier: Multiplier for others class sampling
            max_solar_samples: Maximum solar panel samples
            use_robust_scaler: Whether to use RobustScaler
            
        Returns:
            Dictionary containing results
        """
        start_time = time.time()
        logger.info("Starting simplified PCA visualization pipeline...")
        
        try:
            # 1. Load data
            self.load_data()
            
            # 2. Sample data
            self.sample_data(others_multiplier=others_multiplier, 
                           max_solar_samples=max_solar_samples)
            
            # 3. Preprocess data (simplified)
            self.preprocess_data(use_robust_scaler=use_robust_scaler, 
                               remove_outliers=False)
            
            # 4. Apply PCA
            self.apply_pca(n_components=2)
            
            # 5. Create simple visualization
            self.create_simple_visualization()
            
            total_time = time.time() - start_time
            
            # Compile results
            results = {
                'execution_time': total_time,
                'data_shape': self.sampled_embeddings.shape,
                'solar_panels': int(np.sum(self.sampled_labels == 1)),
                'others': int(np.sum(self.sampled_labels == 0)),
                'pca_shape': self.pca_embeddings.shape,
                'explained_variance': self.pca_model.explained_variance_ratio_.tolist()
            }
            
            logger.info(f"Simple pipeline completed! Total time: {total_time:.2f}s")
            self._print_simple_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def run_full_enhanced_pipeline(self, others_multiplier: int = 10, 
                                 max_solar_samples: int = None,
                                 optimize_pca: bool = True,
                                 clustering_algorithms: List[str] = None) -> Dict:
        """
        Run the complete enhanced PCA pipeline
        
        Args:
            others_multiplier: Multiplier for others samples relative to solar panels
            max_solar_samples: Maximum number of solar panel samples
            optimize_pca: Whether to optimize PCA parameters
            clustering_algorithms: List of clustering algorithms to apply
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting Enhanced PCA Pipeline...")
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Sample data
            self.sample_data(others_multiplier=others_multiplier, 
                           max_solar_samples=max_solar_samples)
            
            # Step 3: Preprocess data
            self.preprocess_data()
            
            # Step 4: Optimize PCA parameters (optional)
            optimization_results = None
            if optimize_pca:
                optimization_results = self.optimize_pca_parameters()
            
            # Step 5: Apply PCA
            self.fit_pca()
            
            # Step 6: Apply clustering algorithms
            clustering_results = self.apply_clustering_algorithms(clustering_algorithms)
            
            # Step 7: Evaluate clustering quality
            evaluation_results = self.evaluate_clustering_quality()
            self.evaluation_results = evaluation_results
            
            # Step 8: Create visualizations (simplified by default)
            self.create_enhanced_visualization(save_individual=False)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Compile results
            results = {
                'data_info': {
                    'total_samples': len(self.sampled_embeddings),
                    'solar_samples': np.sum(self.sampled_labels == 1),
                    'others_samples': np.sum(self.sampled_labels == 0),
                    'feature_dimensions': self.preprocessed_embeddings.shape[1]
                },
                'pca_info': {
                    'components': self.pca_model.n_components_,
                    'explained_variance_ratio': self.pca_model.explained_variance_ratio_.tolist(),
                    'total_explained_variance': np.sum(self.pca_model.explained_variance_ratio_)
                },
                'optimization_results': optimization_results,
                'clustering_results': clustering_results,
                'evaluation_results': evaluation_results,
                'execution_time': total_time
            }
            
            # Print summary
            self._print_pipeline_summary(results)
            
            logger.info(f"Enhanced PCA Pipeline completed successfully in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _print_simple_summary(self, results: Dict) -> None:
        """Print a simple summary of the pipeline results"""
        print("\n" + "="*60)
        print("SIMPLE PCA VISUALIZATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 DATA PROCESSED:")
        print(f"   Total Samples: {results['data_shape'][0]:,}")
        print(f"   Solar Panels: {results['solar_panels']:,}")
        print(f"   Others: {results['others']:,}")
        print(f"   Features: {results['data_shape'][1]}")
        
        print(f"\n🔍 PCA RESULTS:")
        print(f"   Components: {results['pca_shape'][1]}")
        print(f"   Explained Variance (PC1): {results['explained_variance'][0]:.4f}")
        print(f"   Explained Variance (PC2): {results['explained_variance'][1]:.4f}")
        print(f"   Total Explained: {sum(results['explained_variance']):.4f}")
        
        print(f"\n⏱️  EXECUTION TIME: {results['execution_time']:.2f} seconds")
        print("="*60)
    
    def _print_pipeline_summary(self, results: Dict) -> None:
        """Print a comprehensive summary of the pipeline results"""
        print("\n" + "="*80)
        print("ENHANCED PCA VISUALIZATION PIPELINE SUMMARY")
        print("="*80)
        
        # Data information
        data_info = results['data_info']
        print(f"\n📊 DATA INFORMATION:")
        print(f"   Total Samples: {data_info['total_samples']:,}")
        print(f"   Solar Panels: {data_info['solar_samples']:,} ({data_info['solar_samples']/data_info['total_samples']*100:.1f}%)")
        print(f"   Others: {data_info['others_samples']:,} ({data_info['others_samples']/data_info['total_samples']*100:.1f}%)")
        print(f"   Feature Dimensions: {data_info['feature_dimensions']}")
        
        # PCA information
        pca_info = results['pca_info']
        print(f"\n🔍 PCA ANALYSIS:")
        print(f"   Components Used: {pca_info['components']}")
        print(f"   Explained Variance (PC1): {pca_info['explained_variance_ratio'][0]:.4f}")
        print(f"   Explained Variance (PC2): {pca_info['explained_variance_ratio'][1]:.4f}")
        print(f"   Total Explained Variance: {pca_info['total_explained_variance']:.4f}")
        
        # Clustering results
        clustering_results = results['clustering_results']
        evaluation_results = results['evaluation_results']
        
        print(f"\n🎯 CLUSTERING RESULTS:")
        for algorithm, result in clustering_results.items():
            if 'error' not in result:
                eval_metrics = evaluation_results.get(algorithm, {})
                print(f"   {algorithm.upper()}:")
                print(f"     Clusters: {result['n_clusters']}")
                if result.get('n_noise', 0) > 0:
                    print(f"     Noise Points: {result['n_noise']}")
                if 'silhouette_score' in eval_metrics:
                    print(f"     Silhouette Score: {eval_metrics['silhouette_score']:.4f}")
                if 'cluster_purity' in eval_metrics:
                    print(f"     Cluster Purity: {eval_metrics['cluster_purity']:.4f}")
        
        # Best performing algorithm
        if evaluation_results:
            best_algorithm = max(evaluation_results.keys(), 
                               key=lambda k: evaluation_results[k].get('silhouette_score', 0))
            best_score = evaluation_results[best_algorithm]['silhouette_score']
            print(f"\n🏆 BEST PERFORMING ALGORITHM: {best_algorithm.upper()} (Silhouette: {best_score:.4f})")
        
        # Execution time
        print(f"\n⏱️  EXECUTION TIME: {results['execution_time']:.2f} seconds")
        
        print("="*80)


def main():
    """Main function with enhanced argument parsing"""
    parser = argparse.ArgumentParser(
        description='Enhanced PCA Visualization for Solar Panel Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enhanced visualization
  python pca_visualization_enhanced.py --data_dir /path/to/data --output_dir /path/to/output
  
  # Full optimization with KMeans
  python pca_visualization_enhanced.py --data_dir /path/to/data --output_dir /path/to/output --optimize_pca --algorithms kmeans
  
  # Quick test with limited samples
  python pca_visualization_enhanced.py --data_dir /path/to/data --output_dir /path/to/output --max_solar_samples 5000 --others_multiplier 5
        """
    )
    
    # Required arguments
    parser.add_argument('--data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile',
                       help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection',
                       help='Directory to save visualization outputs')
    
    # Optional arguments
    parser.add_argument('--year', type=int, default=2024,
                       help='Year of data to visualize (default: 2024)')
    parser.add_argument('--others_multiplier', type=int, default=10,
                       help='Multiplier for others samples relative to solar panels (default: 10)')
    parser.add_argument('--max_solar_samples', type=int, default=None,
                       help='Maximum number of solar panel samples (default: all)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Advanced options
    parser.add_argument('--optimize_pca', action='store_true',
                       help='Optimize PCA parameters by testing different numbers of components')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['kmeans'],
                       default=['kmeans'],
                       help='Clustering algorithms to apply (default: kmeans only)')
    parser.add_argument('--no_individual_plots', action='store_true',
                       help='Skip creating individual detailed plots')
    
    args = parser.parse_args()
    
    # Initialize enhanced visualizer
    visualizer = EnhancedPCAVisualizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        year=args.year,
        random_seed=args.random_seed
    )
    
    # Run simple pipeline by default, enhanced if optimization requested
    if args.optimize_pca:
        results = visualizer.run_full_enhanced_pipeline(
            others_multiplier=args.others_multiplier,
            max_solar_samples=args.max_solar_samples,
            optimize_pca=args.optimize_pca,
            clustering_algorithms=args.algorithms
        )
    else:
        results = visualizer.run_simple_pipeline(
            others_multiplier=args.others_multiplier,
            max_solar_samples=args.max_solar_samples
        )
    
    logger.info("Enhanced PCA visualization completed successfully!")


if __name__ == "__main__":
    main()