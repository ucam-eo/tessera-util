#!/usr/bin/env python3
"""
Enhanced UMAP Visualization for Solar Panel Detection
====================================================

This enhanced script creates optimized UMAP visualizations specifically designed 
to improve solar panel clustering. It includes:

1. Multi-stage dimensionality reduction (PCA + UMAP)
2. Advanced preprocessing (standardization, outlier detection)
3. Parameter grid search for optimal UMAP settings
4. Multiple clustering algorithms (HDBSCAN, Spectral, GMM)
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
import umap
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
from sklearn.cluster import HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
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
        logging.FileHandler('umap_visualization_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedUMAPVisualizer:
    """
    Enhanced UMAP Visualizer with advanced clustering optimization for Solar Panel Detection
    """
    
    def __init__(self, data_dir: str, output_dir: str, year: int = 2024, random_seed: int = 42):
        """
        Initialize Enhanced UMAP Visualizer
        
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
        self.umap_embeddings = None
        self.cluster_labels = None
        
        # Initialize models
        self.scaler = None
        self.pca_model = None
        self.umap_model = None
        self.best_params = None
        self.clustering_results = {}
        
        logger.info(f"Initialized Enhanced UMAP Visualizer")
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
        
    def apply_pca(self, n_components: int = 50, explained_variance_threshold: float = 0.95) -> None:
        """
        Apply PCA for initial dimensionality reduction
        
        Args:
            n_components: Number of PCA components (or 'auto' for threshold-based)
            explained_variance_threshold: Minimum explained variance ratio
        """
        logger.info("Applying PCA for initial dimensionality reduction...")
        
        try:
            # Ensure n_components is not larger than min(n_samples, n_features)
            max_components = min(self.preprocessed_embeddings.shape[0], self.preprocessed_embeddings.shape[1])
            
            # Determine optimal number of components
            if n_components == 'auto':
                # Find number of components for desired explained variance
                # Use a smaller number for initial fit to avoid memory issues
                temp_components = min(50, max_components)
                pca_temp = PCA(n_components=temp_components, random_state=self.random_seed)
                pca_temp.fit(self.preprocessed_embeddings)
                cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum_var >= explained_variance_threshold) + 1
                logger.info(f"Auto-selected {n_components} components for {explained_variance_threshold*100:.1f}% variance")
            
            # Ensure n_components is valid
            n_components = min(n_components, max_components)
            logger.info(f"Using {n_components} PCA components (max possible: {max_components})")
            
            # Apply PCA with memory-efficient approach
            self.pca_model = PCA(n_components=n_components, random_state=self.random_seed)
            self.pca_embeddings = self.pca_model.fit_transform(self.preprocessed_embeddings)
            
            explained_variance = np.sum(self.pca_model.explained_variance_ratio_)
            logger.info(f"PCA completed:")
            logger.info(f"  Components: {n_components}")
            logger.info(f"  Explained variance: {explained_variance*100:.2f}%")
            logger.info(f"  PCA embeddings shape: {self.pca_embeddings.shape}")
            
        except Exception as e:
            logger.error(f"PCA failed: {e}")
            # Fallback: use original embeddings
            logger.warning("Using original preprocessed embeddings instead of PCA")
            self.pca_embeddings = self.preprocessed_embeddings
            self.pca_model = None
        
    def optimize_umap_parameters(self, param_grid: Dict = None, cv_folds: int = 3) -> Dict:
        """
        Optimize UMAP parameters using grid search with clustering quality metrics
        
        Args:
            param_grid: Parameter grid for search
            cv_folds: Number of cross-validation folds
            
        Returns:
            Best parameters found
        """
        logger.info("Starting UMAP parameter optimization...")
        
        if param_grid is None:
            # Enhanced parameter grid for solar panel clustering
            param_grid = {
                'n_neighbors': [15, 30, 50, 100, 150, 200],
                'min_dist': [0.0, 0.01, 0.05, 0.1, 0.2],
                'metric': ['euclidean', 'cosine', 'manhattan'],
                'n_epochs': [200, 500, 1000]
            }
        
        logger.info(f"Parameter grid: {param_grid}")
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        logger.info(f"Total parameter combinations: {total_combinations}")
        
        best_score = -1
        best_params = None
        results = []
        
        # Use smaller subset for parameter optimization to speed up
        if len(self.pca_embeddings) > 10000:
            opt_indices = np.random.choice(len(self.pca_embeddings), 10000, replace=False)
            opt_embeddings = self.pca_embeddings[opt_indices]
            opt_labels = self.sampled_labels[opt_indices]
        else:
            opt_embeddings = self.pca_embeddings
            opt_labels = self.sampled_labels
        
        # Grid search
        param_combinations = list(ParameterGrid(param_grid))
        
        for i, params in enumerate(tqdm(param_combinations, desc="Optimizing UMAP parameters")):
            try:
                # Fit UMAP with current parameters
                umap_model = umap.UMAP(
                    n_components=2,
                    random_state=self.random_seed,
                    n_jobs=1,  # Use single job for stability during optimization
                    **params
                )
                
                umap_result = umap_model.fit_transform(opt_embeddings)
                
                # Calculate clustering quality metrics
                silhouette = silhouette_score(umap_result, opt_labels)
                calinski_harabasz = calinski_harabasz_score(umap_result, opt_labels)
                davies_bouldin = davies_bouldin_score(umap_result, opt_labels)
                
                # Combined score (higher is better)
                # Normalize davies_bouldin (lower is better) by taking negative
                combined_score = silhouette + (calinski_harabasz / 1000) - davies_bouldin
                
                results.append({
                    'params': params,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski_harabasz,
                    'davies_bouldin': davies_bouldin,
                    'combined_score': combined_score
                })
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_params = params.copy()
                
                if i % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(param_combinations)}, Best score: {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate parameters {params}: {str(e)}")
                continue
        
        # Save optimization results
        results_path = os.path.join(self.output_dir, f'umap_optimization_results_{self.year}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        self.best_params = best_params
        logger.info(f"Parameter optimization completed!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best combined score: {best_score:.4f}")
        logger.info(f"Optimization results saved to: {results_path}")
        
        return best_params
        
    def fit_umap(self, params: Dict = None, n_jobs: int = -1) -> None:
        """
        Fit UMAP model with optimized or provided parameters
        
        Args:
            params: UMAP parameters (use optimized if None)
            n_jobs: Number of parallel jobs
        """
        if params is None:
            if self.best_params is None:
                logger.info("No optimized parameters found, using default enhanced parameters...")
                params = {
                    'n_neighbors': 100,
                    'min_dist': 0.01,
                    'metric': 'cosine',
                    'n_epochs': 500
                }
            else:
                params = self.best_params
        
        logger.info(f"Fitting UMAP model with parameters: {params}")
        
        start_time = time.time()
        
        # Initialize UMAP with optimized parameters
        self.umap_model = umap.UMAP(
            n_components=2,
            random_state=self.random_seed,
            n_jobs=n_jobs,
            verbose=True,
            **params
        )
        
        # Fit and transform
        logger.info("Starting UMAP transformation...")
        self.umap_embeddings = self.umap_model.fit_transform(self.pca_embeddings)
        
        fit_time = time.time() - start_time
        logger.info(f"UMAP fitting completed in {fit_time:.2f}s")
        logger.info(f"UMAP embeddings shape: {self.umap_embeddings.shape}")
        
        # Save UMAP model and embeddings
        model_path = os.path.join(self.output_dir, f'umap_model_enhanced_{self.year}.pkl')
        embeddings_path = os.path.join(self.output_dir, f'umap_embeddings_enhanced_{self.year}.npy')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.umap_model, f)
        np.save(embeddings_path, self.umap_embeddings)
        
        logger.info(f"Enhanced UMAP model saved to: {model_path}")
        logger.info(f"Enhanced UMAP embeddings saved to: {embeddings_path}")
        
    def apply_clustering_algorithms(self, algorithms: List[str] = None) -> Dict:
        """
        Apply multiple clustering algorithms to UMAP embeddings
        
        Args:
            algorithms: List of clustering algorithms to apply
            
        Returns:
            Dictionary containing clustering results and metrics
        """
        if algorithms is None:
            algorithms = ['hdbscan', 'spectral', 'gmm']
        
        logger.info(f"Applying clustering algorithms: {algorithms}")
        
        results = {}
        
        for algorithm in algorithms:
            logger.info(f"Running {algorithm.upper()} clustering...")
            
            try:
                if algorithm == 'hdbscan':
                    results[algorithm] = self._apply_hdbscan()
                elif algorithm == 'spectral':
                    results[algorithm] = self._apply_spectral_clustering()
                elif algorithm == 'gmm':
                    results[algorithm] = self._apply_gaussian_mixture()
                else:
                    logger.warning(f"Unknown clustering algorithm: {algorithm}")
                    continue
                    
                logger.info(f"{algorithm.upper()} clustering completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to apply {algorithm} clustering: {str(e)}")
                results[algorithm] = None
        
        self.clustering_results = results
        
        # Save clustering results
        results_path = os.path.join(self.output_dir, f'clustering_results_{self.year}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Clustering results saved to: {results_path}")
        return results
        
    def _apply_hdbscan(self) -> Dict:
        """Apply HDBSCAN clustering with parameter optimization"""
        from sklearn.cluster import HDBSCAN
        
        # Parameter grid for HDBSCAN optimization
        param_grid = {
            'min_cluster_size': [10, 20, 50, 100],
            'min_samples': [5, 10, 20],
            'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.5]
        }
        
        best_score = -1
        best_params = None
        best_labels = None
        
        for params in ParameterGrid(param_grid):
            try:
                clusterer = HDBSCAN(**params)
                cluster_labels = clusterer.fit_predict(self.umap_embeddings)
                
                # Skip if all points are noise
                if len(np.unique(cluster_labels)) <= 1:
                    continue
                
                # Calculate silhouette score (exclude noise points)
                mask = cluster_labels != -1
                if np.sum(mask) < 10:  # Need at least 10 points for meaningful score
                    continue
                    
                score = silhouette_score(self.umap_embeddings[mask], cluster_labels[mask])
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_labels = cluster_labels
                    
            except Exception as e:
                continue
        
        # Calculate additional metrics
        n_clusters = len(np.unique(best_labels[best_labels != -1]))
        n_noise = np.sum(best_labels == -1)
        
        # Solar panel clustering analysis
        solar_mask = self.sampled_labels == 1
        solar_clusters = best_labels[solar_mask]
        solar_cluster_distribution = np.bincount(solar_clusters[solar_clusters != -1])
        
        return {
            'labels': best_labels,
            'params': best_params,
            'silhouette_score': best_score,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'solar_cluster_distribution': solar_cluster_distribution,
            'algorithm': 'HDBSCAN'
        }
        
    def _apply_spectral_clustering(self) -> Dict:
        """Apply Spectral clustering with parameter optimization"""
        
        # Parameter grid for Spectral clustering
        n_clusters_range = [2, 3, 5, 8, 10, 15, 20]
        affinity_options = ['rbf', 'nearest_neighbors']
        
        best_score = -1
        best_params = None
        best_labels = None
        
        for n_clusters in n_clusters_range:
            for affinity in affinity_options:
                try:
                    clusterer = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity=affinity,
                        random_state=self.random_seed,
                        n_jobs=-1
                    )
                    cluster_labels = clusterer.fit_predict(self.umap_embeddings)
                    
                    score = silhouette_score(self.umap_embeddings, cluster_labels)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'n_clusters': n_clusters, 'affinity': affinity}
                        best_labels = cluster_labels
                        
                except Exception as e:
                    continue
        
        # Solar panel clustering analysis
        solar_mask = self.sampled_labels == 1
        solar_clusters = best_labels[solar_mask]
        solar_cluster_distribution = np.bincount(solar_clusters)
        
        return {
            'labels': best_labels,
            'params': best_params,
            'silhouette_score': best_score,
            'n_clusters': best_params['n_clusters'],
            'solar_cluster_distribution': solar_cluster_distribution,
            'algorithm': 'Spectral'
        }
        
    def _apply_gaussian_mixture(self) -> Dict:
        """Apply Gaussian Mixture Model clustering with parameter optimization"""
        
        # Parameter grid for GMM
        n_components_range = [2, 3, 5, 8, 10, 15, 20]
        covariance_types = ['full', 'tied', 'diag', 'spherical']
        
        best_score = -1
        best_params = None
        best_labels = None
        best_model = None
        
        for n_components in n_components_range:
            for covariance_type in covariance_types:
                try:
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        random_state=self.random_seed
                    )
                    cluster_labels = gmm.fit_predict(self.umap_embeddings)
                    
                    score = silhouette_score(self.umap_embeddings, cluster_labels)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'n_components': n_components, 'covariance_type': covariance_type}
                        best_labels = cluster_labels
                        best_model = gmm
                        
                except Exception as e:
                    continue
        
        # Solar panel clustering analysis
        solar_mask = self.sampled_labels == 1
        solar_clusters = best_labels[solar_mask]
        solar_cluster_distribution = np.bincount(solar_clusters)
        
        return {
            'labels': best_labels,
            'params': best_params,
            'silhouette_score': best_score,
            'n_clusters': best_params['n_components'],
            'solar_cluster_distribution': solar_cluster_distribution,
            'model': best_model,
            'algorithm': 'GMM'
        }
        
    def evaluate_clustering_quality(self) -> Dict:
        """
        Comprehensive evaluation of clustering quality with focus on solar panel clustering
        
        Returns:
            Dictionary containing detailed clustering evaluation metrics
        """
        logger.info("Evaluating clustering quality...")
        
        evaluation_results = {}
        
        for algorithm, result in self.clustering_results.items():
            if result is None:
                continue
                
            logger.info(f"Evaluating {algorithm.upper()} clustering...")
            
            cluster_labels = result['labels']
            
            # Basic clustering metrics
            try:
                # Silhouette score
                if algorithm == 'hdbscan':
                    # Exclude noise points for HDBSCAN
                    mask = cluster_labels != -1
                    if np.sum(mask) > 10:
                        silhouette = silhouette_score(self.umap_embeddings[mask], cluster_labels[mask])
                    else:
                        silhouette = -1
                else:
                    silhouette = silhouette_score(self.umap_embeddings, cluster_labels)
                
                # Calinski-Harabasz score
                calinski_harabasz = calinski_harabasz_score(self.umap_embeddings, cluster_labels)
                
                # Davies-Bouldin score
                davies_bouldin = davies_bouldin_score(self.umap_embeddings, cluster_labels)
                
            except Exception as e:
                logger.warning(f"Failed to calculate basic metrics for {algorithm}: {str(e)}")
                silhouette = calinski_harabasz = davies_bouldin = -1
            
            # Solar panel specific metrics
            solar_mask = self.sampled_labels == 1
            others_mask = self.sampled_labels == 0
            
            solar_clusters = cluster_labels[solar_mask]
            others_clusters = cluster_labels[others_mask]
            
            # Solar panel clustering purity
            solar_cluster_counts = np.bincount(solar_clusters[solar_clusters != -1] if algorithm == 'hdbscan' else solar_clusters)
            solar_purity = np.max(solar_cluster_counts) / len(solar_clusters) if len(solar_cluster_counts) > 0 else 0
            
            # Cluster homogeneity for solar panels
            unique_solar_clusters = np.unique(solar_clusters[solar_clusters != -1] if algorithm == 'hdbscan' else solar_clusters)
            solar_homogeneity = len(unique_solar_clusters) / len(solar_clusters) if len(solar_clusters) > 0 else 0
            
            # Inter-cluster separation
            cluster_centers = []
            unique_clusters = np.unique(cluster_labels[cluster_labels != -1] if algorithm == 'hdbscan' else cluster_labels)
            
            for cluster_id in unique_clusters:
                cluster_points = self.umap_embeddings[cluster_labels == cluster_id]
                if len(cluster_points) > 0:
                    cluster_centers.append(np.mean(cluster_points, axis=0))
            
            if len(cluster_centers) > 1:
                cluster_centers = np.array(cluster_centers)
                # Calculate minimum distance between cluster centers
                from scipy.spatial.distance import pdist
                min_separation = np.min(pdist(cluster_centers))
            else:
                min_separation = 0
            
            evaluation_results[algorithm] = {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'n_clusters': len(unique_clusters),
                'solar_purity': solar_purity,
                'solar_homogeneity': solar_homogeneity,
                'min_cluster_separation': min_separation,
                'solar_cluster_distribution': result.get('solar_cluster_distribution', []),
                'params': result.get('params', {})
            }
            
            logger.info(f"{algorithm.upper()} evaluation:")
            logger.info(f"  Silhouette score: {silhouette:.4f}")
            logger.info(f"  Calinski-Harabasz score: {calinski_harabasz:.2f}")
            logger.info(f"  Davies-Bouldin score: {davies_bouldin:.4f}")
            logger.info(f"  Solar purity: {solar_purity:.4f}")
            logger.info(f"  Number of clusters: {len(unique_clusters)}")
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, f'clustering_evaluation_{self.year}.pkl')
        with open(eval_path, 'wb') as f:
            pickle.dump(evaluation_results, f)
        
        logger.info(f"Clustering evaluation results saved to: {eval_path}")
        return evaluation_results
        
    def create_enhanced_visualization(self, figsize: Tuple[int, int] = (20, 15), 
                                    save_individual: bool = True) -> None:
        """
        Create comprehensive enhanced visualizations with multiple clustering results
        
        Args:
            figsize: Figure size for the visualization
            save_individual: Whether to save individual algorithm plots
        """
        logger.info("Creating enhanced visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create main comparison figure
        n_algorithms = len(self.clustering_results)
        if n_algorithms == 0:
            logger.warning("No clustering results available for visualization")
            return
        
        # Create subplot layout
        fig, axes = plt.subplots(2, max(2, (n_algorithms + 1) // 2), figsize=figsize)
        if n_algorithms == 1:
            axes = [axes]
        axes = axes.flatten()
        
        # Plot original labels
        ax = axes[0]
        scatter = ax.scatter(
            self.umap_embeddings[:, 0], 
            self.umap_embeddings[:, 1],
            c=self.sampled_labels,
            cmap='RdYlBu_r',
            alpha=0.7,
            s=20
        )
        ax.set_title('Original Labels\n(Red: Solar Panels, Blue: Others)', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Label', fontsize=10)
        
        # Plot clustering results
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        for i, (algorithm, result) in enumerate(self.clustering_results.items(), 1):
            if result is None or i >= len(axes):
                continue
                
            ax = axes[i]
            cluster_labels = result['labels']
            
            # Handle noise points for HDBSCAN
            if algorithm == 'hdbscan':
                # Plot noise points in gray
                noise_mask = cluster_labels == -1
                if np.any(noise_mask):
                    ax.scatter(
                        self.umap_embeddings[noise_mask, 0],
                        self.umap_embeddings[noise_mask, 1],
                        c='lightgray',
                        alpha=0.3,
                        s=10,
                        label='Noise'
                    )
                
                # Plot clustered points
                clustered_mask = cluster_labels != -1
                if np.any(clustered_mask):
                    scatter = ax.scatter(
                        self.umap_embeddings[clustered_mask, 0],
                        self.umap_embeddings[clustered_mask, 1],
                        c=cluster_labels[clustered_mask],
                        cmap='tab20',
                        alpha=0.7,
                        s=20
                    )
            else:
                scatter = ax.scatter(
                    self.umap_embeddings[:, 0],
                    self.umap_embeddings[:, 1],
                    c=cluster_labels,
                    cmap='tab20',
                    alpha=0.7,
                    s=20
                )
            
            # Add title with metrics
            silhouette = result.get('silhouette_score', 0)
            n_clusters = result.get('n_clusters', 0)
            title = f'{algorithm.upper()}\nClusters: {n_clusters}, Silhouette: {silhouette:.3f}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('UMAP 1', fontsize=10)
            ax.set_ylabel('UMAP 2', fontsize=10)
        
        # Hide unused subplots
        for i in range(len(self.clustering_results) + 1, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save main comparison plot
        main_plot_path = os.path.join(self.output_dir, f'enhanced_umap_comparison_{self.year}.png')
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Main comparison plot saved to: {main_plot_path}")
        
        plt.show()
        plt.close()
        
        # Create individual detailed plots if requested
        if save_individual:
            self._create_individual_plots()
        
        # Create solar panel focus plots
        self._create_solar_panel_focus_plots()
        
    def _create_individual_plots(self) -> None:
        """Create detailed individual plots for each clustering algorithm"""
        logger.info("Creating individual detailed plots...")
        
        for algorithm, result in self.clustering_results.items():
            if result is None:
                continue
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            cluster_labels = result['labels']
            
            # Plot 1: Clustering result
            if algorithm == 'hdbscan':
                noise_mask = cluster_labels == -1
                clustered_mask = cluster_labels != -1
                
                if np.any(noise_mask):
                    ax1.scatter(
                        self.umap_embeddings[noise_mask, 0],
                        self.umap_embeddings[noise_mask, 1],
                        c='lightgray', alpha=0.3, s=10, label='Noise'
                    )
                
                if np.any(clustered_mask):
                    scatter1 = ax1.scatter(
                        self.umap_embeddings[clustered_mask, 0],
                        self.umap_embeddings[clustered_mask, 1],
                        c=cluster_labels[clustered_mask],
                        cmap='tab20', alpha=0.7, s=20
                    )
            else:
                scatter1 = ax1.scatter(
                    self.umap_embeddings[:, 0],
                    self.umap_embeddings[:, 1],
                    c=cluster_labels,
                    cmap='tab20', alpha=0.7, s=20
                )
            
            ax1.set_title(f'{algorithm.upper()} Clustering Result', fontsize=14, fontweight='bold')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            
            # Plot 2: Solar panels highlighted
            solar_mask = self.sampled_labels == 1
            ax2.scatter(
                self.umap_embeddings[~solar_mask, 0],
                self.umap_embeddings[~solar_mask, 1],
                c='lightblue', alpha=0.3, s=10, label='Others'
            )
            ax2.scatter(
                self.umap_embeddings[solar_mask, 0],
                self.umap_embeddings[solar_mask, 1],
                c='red', alpha=0.8, s=30, label='Solar Panels'
            )
            ax2.set_title('Solar Panel Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            ax2.legend()
            
            # Plot 3: Solar panels with cluster colors
            solar_clusters = cluster_labels[solar_mask]
            if algorithm == 'hdbscan':
                solar_clustered_mask = solar_clusters != -1
                if np.any(solar_clustered_mask):
                    solar_coords = self.umap_embeddings[solar_mask]
                    ax3.scatter(
                        solar_coords[solar_clustered_mask, 0],
                        solar_coords[solar_clustered_mask, 1],
                        c=solar_clusters[solar_clustered_mask],
                        cmap='tab20', alpha=0.8, s=30
                    )
                    if np.any(~solar_clustered_mask):
                        ax3.scatter(
                            solar_coords[~solar_clustered_mask, 0],
                            solar_coords[~solar_clustered_mask, 1],
                            c='gray', alpha=0.5, s=20, label='Noise'
                        )
            else:
                ax3.scatter(
                    self.umap_embeddings[solar_mask, 0],
                    self.umap_embeddings[solar_mask, 1],
                    c=solar_clusters,
                    cmap='tab20', alpha=0.8, s=30
                )
            
            ax3.set_title('Solar Panel Clusters', fontsize=14, fontweight='bold')
            ax3.set_xlabel('UMAP 1')
            ax3.set_ylabel('UMAP 2')
            
            # Plot 4: Cluster statistics
            ax4.axis('off')
            
            # Calculate and display statistics
            stats_text = f"{algorithm.upper()} Clustering Statistics\n\n"
            stats_text += f"Total Clusters: {result.get('n_clusters', 0)}\n"
            stats_text += f"Silhouette Score: {result.get('silhouette_score', 0):.4f}\n"
            
            if 'n_noise' in result:
                stats_text += f"Noise Points: {result['n_noise']}\n"
            
            # Solar panel statistics
            solar_cluster_dist = result.get('solar_cluster_distribution', [])
            if len(solar_cluster_dist) > 0:
                stats_text += f"\nSolar Panel Distribution:\n"
                for i, count in enumerate(solar_cluster_dist):
                    if count > 0:
                        stats_text += f"  Cluster {i}: {count} panels\n"
            
            # Parameters
            params = result.get('params', {})
            if params:
                stats_text += f"\nOptimal Parameters:\n"
                for key, value in params.items():
                    stats_text += f"  {key}: {value}\n"
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save individual plot
            individual_path = os.path.join(self.output_dir, f'{algorithm}_detailed_{self.year}.png')
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            logger.info(f"{algorithm.upper()} detailed plot saved to: {individual_path}")
            
            plt.close()
            
    def _create_solar_panel_focus_plots(self) -> None:
        """Create specialized plots focusing on solar panel clustering quality"""
        logger.info("Creating solar panel focus plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Solar panel density heatmap
        ax = axes[0, 0]
        solar_mask = self.sampled_labels == 1
        solar_coords = self.umap_embeddings[solar_mask]
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(
            solar_coords[:, 0], solar_coords[:, 1], bins=50
        )
        
        im = ax.imshow(hist.T, origin='lower', cmap='Reds', alpha=0.8,
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        ax.scatter(solar_coords[:, 0], solar_coords[:, 1], 
                  c='darkred', alpha=0.6, s=10)
        ax.set_title('Solar Panel Density Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(im, ax=ax, label='Density')
        
        # Plot 2: Best clustering result for solar panels
        ax = axes[0, 1]
        if self.clustering_results:
            # Find best algorithm based on silhouette score
            best_algorithm = max(
                self.clustering_results.keys(),
                key=lambda k: self.clustering_results[k].get('silhouette_score', -1) 
                if self.clustering_results[k] else -1
            )
            
            best_result = self.clustering_results[best_algorithm]
            if best_result:
                cluster_labels = best_result['labels']
                
                # Plot all points lightly
                ax.scatter(
                    self.umap_embeddings[:, 0],
                    self.umap_embeddings[:, 1],
                    c='lightgray', alpha=0.2, s=5
                )
                
                # Highlight solar panels with cluster colors
                if best_algorithm == 'hdbscan':
                    solar_clusters = cluster_labels[solar_mask]
                    solar_clustered_mask = solar_clusters != -1
                    if np.any(solar_clustered_mask):
                        ax.scatter(
                            solar_coords[solar_clustered_mask, 0],
                            solar_coords[solar_clustered_mask, 1],
                            c=solar_clusters[solar_clustered_mask],
                            cmap='tab20', alpha=0.8, s=30
                        )
                else:
                    ax.scatter(
                        solar_coords[:, 0], solar_coords[:, 1],
                        c=cluster_labels[solar_mask],
                        cmap='tab20', alpha=0.8, s=30
                    )
                
                ax.set_title(f'Best Solar Panel Clustering\n({best_algorithm.upper()})', 
                           fontsize=14, fontweight='bold')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        # Plot 3: Clustering quality comparison
        ax = axes[1, 0]
        algorithms = []
        silhouette_scores = []
        
        for algorithm, result in self.clustering_results.items():
            if result and 'silhouette_score' in result:
                algorithms.append(algorithm.upper())
                silhouette_scores.append(result['silhouette_score'])
        
        if algorithms:
            bars = ax.bar(algorithms, silhouette_scores, color='skyblue', alpha=0.7)
            ax.set_title('Clustering Quality Comparison\n(Silhouette Score)', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Silhouette Score')
            ax.set_ylim(0, max(silhouette_scores) * 1.1 if silhouette_scores else 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, silhouette_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 4: Solar panel cluster size distribution
        ax = axes[1, 1]
        
        # Combine all solar panel cluster distributions
        all_distributions = []
        algorithm_names = []
        
        for algorithm, result in self.clustering_results.items():
            if result and 'solar_cluster_distribution' in result:
                dist = result['solar_cluster_distribution']
                if len(dist) > 0:
                    all_distributions.extend(dist[dist > 0])  # Only non-zero clusters
                    algorithm_names.extend([algorithm.upper()] * len(dist[dist > 0]))
        
        if all_distributions:
            ax.hist(all_distributions, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax.set_title('Solar Panel Cluster Size Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Cluster Size (Number of Solar Panels)')
            ax.set_ylabel('Frequency')
        else:
            ax.text(0.5, 0.5, 'No cluster data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Solar Panel Cluster Size Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save solar panel focus plot
        focus_path = os.path.join(self.output_dir, f'solar_panel_focus_{self.year}.png')
        plt.savefig(focus_path, dpi=300, bbox_inches='tight')
        logger.info(f"Solar panel focus plot saved to: {focus_path}")
        
        plt.close()
        
    def run_full_enhanced_pipeline(self, others_multiplier: int = 10, 
                                 max_solar_samples: int = None,
                                 pca_components: int = 50,
                                 optimize_umap: bool = True,
                                 clustering_algorithms: List[str] = None) -> Dict:
        """
        Run the complete enhanced UMAP visualization pipeline
        
        Args:
            others_multiplier: Multiplier for others samples relative to solar panels
            max_solar_samples: Maximum number of solar panel samples
            pca_components: Number of PCA components
            optimize_umap: Whether to optimize UMAP parameters
            clustering_algorithms: List of clustering algorithms to apply
            
        Returns:
            Dictionary containing all results and evaluation metrics
        """
        logger.info("Starting enhanced UMAP visualization pipeline...")
        
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Load data
            logger.info("Step 1: Loading data...")
            self.load_data()
            
            # Step 2: Sample data
            logger.info("Step 2: Sampling data...")
            self.sample_data(others_multiplier=others_multiplier, 
                           max_solar_samples=max_solar_samples)
            
            # Step 3: Preprocess data
            logger.info("Step 3: Preprocessing data...")
            self.preprocess_data()
            
            # Step 4: Apply PCA
            logger.info("Step 4: Applying PCA...")
            self.apply_pca(n_components=pca_components)
            
            # Step 5: Optimize UMAP parameters (optional)
            if optimize_umap:
                logger.info("Step 5: Optimizing UMAP parameters...")
                self.optimize_umap_parameters()
            
            # Step 6: Fit UMAP
            logger.info("Step 6: Fitting UMAP...")
            self.fit_umap()
            
            # Step 7: Apply clustering algorithms
            logger.info("Step 7: Applying clustering algorithms...")
            self.apply_clustering_algorithms(algorithms=clustering_algorithms)
            
            # Step 8: Evaluate clustering quality
            logger.info("Step 8: Evaluating clustering quality...")
            evaluation_results = self.evaluate_clustering_quality()
            
            # Step 9: Create visualizations
            logger.info("Step 9: Creating enhanced visualizations...")
            self.create_enhanced_visualization()
            
            pipeline_time = time.time() - pipeline_start_time
            
            # Compile final results
            final_results = {
                'pipeline_time': pipeline_time,
                'data_info': {
                    'total_samples': len(self.sampled_embeddings),
                    'solar_panels': np.sum(self.sampled_labels == 1),
                    'others': np.sum(self.sampled_labels == 0),
                    'original_dimensions': self.embeddings.shape[1],
                    'pca_dimensions': self.pca_embeddings.shape[1]
                },
                'umap_params': self.best_params,
                'clustering_results': self.clustering_results,
                'evaluation_results': evaluation_results
            }
            
            # Save final results
            results_path = os.path.join(self.output_dir, f'enhanced_pipeline_results_{self.year}.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(final_results, f)
            
            logger.info(f"Enhanced pipeline completed successfully in {pipeline_time:.2f}s")
            logger.info(f"Final results saved to: {results_path}")
            
            # Print summary
            self._print_pipeline_summary(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced pipeline failed: {str(e)}")
            raise
            
    def _print_pipeline_summary(self, results: Dict) -> None:
        """Print a comprehensive summary of the pipeline results"""
        logger.info("\n" + "="*80)
        logger.info("ENHANCED UMAP VISUALIZATION PIPELINE SUMMARY")
        logger.info("="*80)
        
        # Data info
        data_info = results['data_info']
        logger.info(f"Data Processing:")
        logger.info(f"  Total samples analyzed: {data_info['total_samples']:,}")
        logger.info(f"  Solar panels: {data_info['solar_panels']:,}")
        logger.info(f"  Others: {data_info['others']:,}")
        logger.info(f"  Dimensionality reduction: {data_info['original_dimensions']} → {data_info['pca_dimensions']} → 2")
        
        # UMAP optimization
        if results['umap_params']:
            logger.info(f"\nOptimal UMAP Parameters:")
            for key, value in results['umap_params'].items():
                logger.info(f"  {key}: {value}")
        
        # Clustering results
        logger.info(f"\nClustering Results:")
        evaluation_results = results['evaluation_results']
        
        best_algorithm = None
        best_score = -1
        
        for algorithm, eval_result in evaluation_results.items():
            silhouette = eval_result['silhouette_score']
            n_clusters = eval_result['n_clusters']
            solar_purity = eval_result['solar_purity']
            
            logger.info(f"  {algorithm.upper()}:")
            logger.info(f"    Silhouette Score: {silhouette:.4f}")
            logger.info(f"    Number of Clusters: {n_clusters}")
            logger.info(f"    Solar Panel Purity: {solar_purity:.4f}")
            
            if silhouette > best_score:
                best_score = silhouette
                best_algorithm = algorithm
        
        if best_algorithm:
            logger.info(f"\nBest Clustering Algorithm: {best_algorithm.upper()}")
            logger.info(f"Best Silhouette Score: {best_score:.4f}")
        
        logger.info(f"\nPipeline completed in {results['pipeline_time']:.2f} seconds")
        logger.info("="*80)


def main():
    """Main function with enhanced argument parsing"""
    parser = argparse.ArgumentParser(
        description='Enhanced UMAP Visualization for Solar Panel Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enhanced visualization
  python umap_visualization_enhanced.py --data_dir /path/to/data --output_dir /path/to/output
  
  # Full optimization with all algorithms
  python umap_visualization_enhanced.py --data_dir /path/to/data --output_dir /path/to/output --optimize_umap --algorithms hdbscan spectral gmm
  
  # Quick test with limited samples
  python umap_visualization_enhanced.py --data_dir /path/to/data --output_dir /path/to/output --max_solar_samples 5000 --others_multiplier 5
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
    parser.add_argument('--pca_components', type=int, default=50,
                       help='Number of PCA components (default: 50)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Advanced options
    parser.add_argument('--optimize_umap', action='store_true',
                       help='Optimize UMAP parameters using grid search')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['hdbscan', 'spectral', 'gmm'],
                       default=['hdbscan', 'spectral', 'gmm'],
                       help='Clustering algorithms to apply (default: all)')
    parser.add_argument('--no_individual_plots', action='store_true',
                       help='Skip creating individual detailed plots')
    
    args = parser.parse_args()
    
    # Initialize enhanced visualizer
    visualizer = EnhancedUMAPVisualizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        year=args.year,
        random_seed=args.random_seed
    )
    
    # Run enhanced pipeline
    results = visualizer.run_full_enhanced_pipeline(
        others_multiplier=args.others_multiplier,
        max_solar_samples=args.max_solar_samples,
        pca_components=args.pca_components,
        optimize_umap=args.optimize_umap,
        clustering_algorithms=args.algorithms
    )
    
    logger.info("Enhanced UMAP visualization completed successfully!")


if __name__ == "__main__":
    main()