import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import pickle
import os
import time
import logging
from typing import Dict, Tuple, Any
import argparse

from data_preprocessing import load_processed_data
from sampling import create_sampler
from evaluation import (
    calculate_metrics, print_detailed_metrics, compare_models,
    find_best_threshold
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PVDetectionTrainer:
    """
    Trainer class for PV detection using LightGBM and XGBoost.
    """

    def __init__(self, data_dir: str, pos_neg_ratio: float = 0.33, random_seed: int = 42, max_positive_samples: int = None, use_cache: bool = False, cache_dir: str = None,
                 use_extra_training_data: bool = False, extra_data_dir: str = None, uk_data_dir: str = None, disable_main_training_data: bool = False):
        """
        Initialize the PV Detection Trainer.

        Args:
            data_dir: directory containing the data files
            pos_neg_ratio: ratio of positive to negative samples in balanced sampling
            random_seed: random seed for reproducibility
            max_positive_samples: maximum number of positive samples to use for training
            use_cache: if True, use cached coordinates and fast loading via memmap
            cache_dir: directory to store cache files (default: data_dir/cache)
            use_extra_training_data: if True, use additional training data from extra directories
            extra_data_dir: directory containing extra training data
            uk_data_dir: directory containing UK training data
            disable_main_training_data: if True, disable main training data and use only extra data
        """
        self.data_dir = data_dir
        self.pos_neg_ratio = pos_neg_ratio
        self.random_seed = random_seed
        self.max_positive_samples = max_positive_samples
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.use_extra_training_data = use_extra_training_data
        self.extra_data_dir = extra_data_dir
        self.uk_data_dir = uk_data_dir
        self.disable_main_training_data = disable_main_training_data
        self.data = None
        self.sampler = None
        self.models = {}
        self.results = {}

        logger.info(f"Initializing PV Detection Trainer")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Positive:Negative ratio: 1:{1/pos_neg_ratio:.1f}")
        logger.info(f"Random seed: {random_seed}")
        logger.info(f"Use cache: {use_cache}")
        logger.info(f"Use extra training data: {use_extra_training_data}")
        if disable_main_training_data:
            logger.info("Main training data disabled - using only extra training data")
        if max_positive_samples is not None:
            logger.info(f"Maximum positive samples for training: {max_positive_samples}")
        else:
            logger.info("No limit on positive samples for training")

    def load_data(self, years: list = None) -> None:
        """Load and preprocess data."""
        logger.info("Loading and preprocessing data...")
        if years is None:
            years = list(range(2017, 2025))  # Default to 2017-2024
        
        # Initialize data structure
        data = {'train_features': None, 'train_labels': None, 'train_coords': None, 'train_years': None,
                'val_features': None, 'val_labels': None, 'val_coords': None, 'val_years': None,
                'years': years, 'labels': None}
        
        # Load main training data unless disabled
        if not self.disable_main_training_data:
            data = load_processed_data(
                self.data_dir, 
                max_positive_samples=self.max_positive_samples, 
                years=years,
                use_cache=self.use_cache,
                cache_dir=self.cache_dir,
                pos_neg_ratio=self.pos_neg_ratio
            )
            logger.info(f"Loaded {len(data['train_features']):,} main training samples")
        else:
            logger.info("Main training data disabled - skipping main data loading")

        # Load extra training data if requested
        if self.use_extra_training_data:
            logger.info("Loading extra training data...")
            try:
                from extra_data_loader import ExtraDataLoader
                extra_loader = ExtraDataLoader(self.extra_data_dir, self.uk_data_dir)
                extra_data = extra_loader.load_all_extra_data(years)
            except ImportError as e:
                logger.error(f"Failed to import ExtraDataLoader: {e}")
                logger.error("Please install required dependencies (e.g., rasterio) to use extra training data")
                if self.disable_main_training_data:
                    raise ValueError("Cannot use only extra data - ExtraDataLoader import failed")
                else:
                    logger.warning("Continuing with main training data only")
                    extra_data = {'features': np.array([]), 'labels': np.array([]), 'coords': np.array([]), 'years': np.array([])}
            
            if extra_data['features'].shape[0] > 0:
                logger.info(f"Loaded {len(extra_data['features']):,} extra training samples")
                
                # Extra features are already in 2D format (N, 128) for pixel-level learning
                extra_features = extra_data['features']
                logger.info(f"Extra features shape: {extra_features.shape}")
                
                # Combine main and extra data if both exist
                if not self.disable_main_training_data and data['train_features'] is not None:
                    logger.info("Combining main and extra training data...")
                    
                    # Combine training data
                    data['train_features'] = np.concatenate([data['train_features'], extra_features], axis=0)
                    data['train_labels'] = np.concatenate([data['train_labels'], extra_data['labels']], axis=0)
                    data['train_coords'] = np.concatenate([data['train_coords'], extra_data['coords']], axis=0)
                    data['train_years'] = np.concatenate([data['train_years'], extra_data['years']], axis=0)
                    
                    logger.info(f"Combined dataset: {len(data['train_features']):,} total training samples")
                    
                elif self.disable_main_training_data:
                    # Use only extra data - split it into train/validation sets
                    logger.info("Using only extra training data...")
                    
                    # Split extra data into train/validation sets (90/10 split)
                    n_samples = len(extra_features)
                    indices = np.random.permutation(n_samples)
                    n_train = int(0.9 * n_samples)
                    
                    train_indices = indices[:n_train]
                    val_indices = indices[n_train:]
                    
                    data['train_features'] = extra_features[train_indices]
                    data['train_labels'] = extra_data['labels'][train_indices]
                    data['train_coords'] = extra_data['coords'][train_indices]
                    data['train_years'] = extra_data['years'][train_indices]
                    
                    data['val_features'] = extra_features[val_indices]
                    data['val_labels'] = extra_data['labels'][val_indices]
                    data['val_coords'] = extra_data['coords'][val_indices]
                    data['val_years'] = extra_data['years'][val_indices]
                    
                    logger.info(f"Extra data split - Train: {len(data['train_features']):,}, Val: {len(data['val_features']):,}")
            else:
                logger.warning("No extra training data found!")
                if self.disable_main_training_data:
                    raise ValueError("No training data available - main data disabled and no extra data found")

        # Validate that we have training data
        if data['train_features'] is None or len(data['train_features']) == 0:
            raise ValueError("No training data available")

        self.data = data

        # Create balanced sampler only if not using cache (cache already does balanced sampling)
        if not self.use_cache:
            self.sampler = create_sampler(
                self.data['train_features'],
                self.data['train_labels'],
                pos_neg_ratio=self.pos_neg_ratio,
                random_seed=self.random_seed
            )
        else:
            logger.info("Using cache mode - balanced sampling already applied during data loading")
            self.sampler = None

        logger.info("Data loading completed.")

    def train_lightgbm(self, use_balanced_sampling: bool = True, quick_eval: bool = False, **kwargs) -> None:
        """
        Train LightGBM model.

        Args:
            use_balanced_sampling: whether to use balanced sampling
            **kwargs: additional parameters for LightGBM
        """
        logger.info("Training LightGBM model...")

        # Default parameters optimized for binary classification with imbalanced data
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_seed,
            'n_jobs': 64  # Use all available cores
        }

        # Update with provided parameters
        params = {**default_params, **kwargs}

        if use_balanced_sampling:
            # Create balanced training dataset
            train_features, train_labels = self.sampler.create_balanced_dataset()
            logger.info("Using balanced sampling for training")
        else:
            # Use class weights for handling imbalance
            class_weights = self.sampler.get_class_weights()
            params['class_weight'] = class_weights
            train_features = self.data['train_features']
            train_labels = self.data['train_labels']
            logger.info("Using class weights for handling imbalance")

        # Create LightGBM datasets
        train_data = lgb.Dataset(train_features, label=train_labels)
        val_data = lgb.Dataset(
            self.data['val_features'],
            label=self.data['val_labels'],
            reference=train_data
        )

        # Train model
        start_time = time.time()
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=5000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        training_time = time.time() - start_time

        self.models['lightgbm'] = model
        logger.info(f"LightGBM training completed in {training_time:.2f} seconds")

        # Evaluate model
        self._evaluate_model('lightgbm', model, quick_eval=quick_eval)

    def train_xgboost(self, use_balanced_sampling: bool = True, quick_eval: bool = False, **kwargs) -> None:
        """
        Train XGBoost model.

        Args:
            use_balanced_sampling: whether to use balanced sampling
            **kwargs: additional parameters for XGBoost
        """
        logger.info("Training XGBoost model...")

        # Default parameters
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'random_state': self.random_seed,
            'n_jobs': 64,  # Use all available cores
            'verbosity': 1
        }

        # Update with provided parameters
        params = {**default_params, **kwargs}

        if use_balanced_sampling and self.sampler is not None:
            # Create balanced training dataset
            train_features, train_labels = self.sampler.create_balanced_dataset()
            logger.info("Using balanced sampling for training")
        else:
            # Calculate scale_pos_weight for handling imbalance
            neg_count = np.sum(self.data['train_labels'] == 0)
            pos_count = np.sum(self.data['train_labels'] == 1)
            scale_pos_weight = neg_count / pos_count
            params['scale_pos_weight'] = scale_pos_weight
            train_features = self.data['train_features']
            train_labels = self.data['train_labels']
            logger.info(f"Using scale_pos_weight={scale_pos_weight:.2f} for handling imbalance")

        # Create DMatrix objects
        dtrain = xgb.DMatrix(train_features, label=train_labels)
        dval = xgb.DMatrix(self.data['val_features'], label=self.data['val_labels'])

        # Train model
        start_time = time.time()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        training_time = time.time() - start_time

        self.models['xgboost'] = model
        logger.info(f"XGBoost training completed in {training_time:.2f} seconds")

        # Evaluate model
        self._evaluate_model('xgboost', model, quick_eval=quick_eval)

    def _evaluate_model(self, model_name: str, model: Any, quick_eval: bool = False, yearly_eval: bool = True) -> None:
        """
        Evaluate a trained model.

        Args:
            model_name: name of the model
            model: trained model object
            quick_eval: if True, perform quick evaluation without threshold optimization
            yearly_eval: if True, perform year-by-year evaluation in addition to overall evaluation
        """
        logger.info(f"Evaluating {model_name} model...")
        
        # For cache mode, use simplified evaluation on training set
        if self.use_cache and self.data.get('use_cache', False):
            logger.info("Cache mode: Using simplified evaluation on training set")
            
            # Use training data for evaluation (since it's already balanced and representative)
            eval_features = self.data['train_features']
            eval_labels = self.data['train_labels']
            eval_years = self.data['train_years']
            
            # Make predictions
            if model_name == 'lightgbm':
                predictions = model.predict(eval_features)
            elif model_name == 'xgboost':
                deval = xgb.DMatrix(eval_features)
                predictions = model.predict(deval)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Use fixed threshold for quick evaluation
            threshold = 0.995
            binary_predictions = (predictions > threshold).astype(int)
            
            # Calculate overall metrics
            metrics = calculate_metrics(eval_labels, binary_predictions, predictions)
            
            logger.info(f"{model_name} - Overall Performance (Training Set):")
            print_detailed_metrics(
                eval_labels, binary_predictions, predictions,
                f"{model_name.upper()} - Training Set (Cache Mode)"
            )
            
            # Store results
            self.results[model_name] = {
                'metrics': metrics,
                'threshold': threshold,
                'evaluation_type': 'training_set_cache_mode'
            }
            
            return

        # Make predictions on validation set
        if model_name == 'lightgbm':
            val_pred_proba = model.predict(self.data['val_features'])
        elif model_name == 'xgboost':
            dval = xgb.DMatrix(self.data['val_features'])
            val_pred_proba = model.predict(dval)

        if quick_eval:
            # Quick evaluation with default threshold 0.995
            best_threshold = 0.995
            val_pred = (val_pred_proba >= best_threshold).astype(int)
            
            # Calculate basic metrics
            metrics = calculate_metrics(self.data['val_labels'], val_pred, val_pred_proba)
            metrics['best_threshold'] = best_threshold
            self.results[model_name] = metrics
            
            logger.info(f"Quick evaluation completed for {model_name}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics.get('auc_roc', 'N/A')}")
        else:
            # Full evaluation with threshold optimization
            best_threshold, best_metrics = find_best_threshold(
                self.data['val_labels'], val_pred_proba, metric='f1'
            )

            # Make final predictions with optimal threshold
            val_pred = (val_pred_proba >= best_threshold).astype(int)

            # Calculate and store metrics
            metrics = calculate_metrics(self.data['val_labels'], val_pred, val_pred_proba)
            metrics['best_threshold'] = best_threshold
            self.results[model_name] = metrics

            # Print detailed results
            print_detailed_metrics(
                self.data['val_labels'], val_pred, val_pred_proba,
                f"{model_name.upper()} - Validation Set"
            )

            logger.info(f"Optimal threshold for {model_name}: {best_threshold:.3f}")

        # Perform year-by-year evaluation if requested and multi-year data is available
        if yearly_eval and 'val_years' in self.data and len(np.unique(self.data['val_years'])) > 1:
            logger.info(f"Performing year-by-year evaluation for {model_name}...")
            yearly_results = {}
            
            unique_years = np.unique(self.data['val_years'])
            for year in sorted(unique_years):
                year_mask = self.data['val_years'] == year
                year_labels = self.data['val_labels'][year_mask]
                year_pred_proba = val_pred_proba[year_mask]
                year_pred = (year_pred_proba >= best_threshold).astype(int)
                
                if len(year_labels) > 0:
                    year_metrics = calculate_metrics(year_labels, year_pred, year_pred_proba)
                    yearly_results[year] = year_metrics
                    
                    logger.info(f"Year {year} - Samples: {len(year_labels):,}, "
                              f"Accuracy: {year_metrics['accuracy']:.4f}, "
                              f"F1: {year_metrics['f1']:.4f}, "
                              f"Precision: {year_metrics['precision']:.4f}, "
                              f"Recall: {year_metrics['recall']:.4f}")
            
            # Store yearly results
            self.results[f"{model_name}_yearly"] = yearly_results
            
            # Calculate average metrics across years
            if yearly_results:
                avg_metrics = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    values = [yearly_results[year][metric] for year in yearly_results if metric in yearly_results[year]]
                    if values:
                        avg_metrics[f"avg_{metric}"] = np.mean(values)
                        avg_metrics[f"std_{metric}"] = np.std(values)
                
                logger.info(f"Average metrics across years:")
                for metric, value in avg_metrics.items():
                    if metric.startswith('avg_'):
                        std_metric = metric.replace('avg_', 'std_')
                        std_value = avg_metrics.get(std_metric, 0)
                        logger.info(f"  {metric}: {value:.4f} ± {std_value:.4f}")
                
                self.results[f"{model_name}_avg_yearly"] = avg_metrics

    def save_models(self, output_dir: str) -> None:
        """
        Save trained models and results.

        Args:
            output_dir: directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)

        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{model_name}_model.pkl")

            if model_name == 'lightgbm':
                model.save_model(model_path.replace('.pkl', '.txt'))
                logger.info(f"LightGBM model saved to {model_path.replace('.pkl', '.txt')}")
            elif model_name == 'xgboost':
                model.save_model(model_path.replace('.pkl', '.json'))
                logger.info(f"XGBoost model saved to {model_path.replace('.pkl', '.json')}")

            # Also save as pickle for easy loading
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Save results
        results_path = os.path.join(output_dir, 'results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)

        logger.info(f"Results saved to {results_path}")

    def compare_models(self) -> None:
        """Compare all trained models."""
        if len(self.results) > 1:
            compare_models(self.results)
        else:
            logger.info("Need at least 2 models to compare.")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PV detection models')
    parser.add_argument('--data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1',
                       help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection/models',
                       help='Directory to save trained models')
    parser.add_argument('--pos_neg_ratio', type=float, default=0.33,
                       help='Positive to negative ratio for balanced sampling (0.2 = 1:5)')
    parser.add_argument('--models', nargs='+', default=['xgboost'],
                       choices=['lightgbm', 'xgboost'],
                       help='Models to train')
    parser.add_argument('--balanced_sampling', action='store_true', default=True,
                       help='Use balanced sampling instead of class weights')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--quick_eval', action='store_true', default=False,
                       help='Use quick evaluation (default threshold 0.995) instead of threshold optimization')
    parser.add_argument('--max_positive_samples', type=int, default=None,
                       help='Maximum number of positive samples to use for training. Excess positive samples will be moved to validation set.')
    parser.add_argument('--use_cache', action='store_true', default=False,
                       help='Use cached coordinates and memmap for fast data loading')
    parser.add_argument('--cache_dir', type=str, default="/maps/zf281/btfm4rs/src/pv_detection/cache",
                       help='Directory to store cache files (default: data_dir/cache)')
    parser.add_argument('--use_extra_training_data', action='store_true', default=False,
                       help='Use additional training data from extra_training_data directory')
    parser.add_argument('--extra_data_dir', type=str, 
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data',
                       help='Directory containing extra training data')
    parser.add_argument('--uk_data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/uk',
                       help='Directory containing UK training data')
    parser.add_argument('--disable_main_training_data', action='store_true', default=False,
                       help='Disable main training data and use only extra training data')
    parser.add_argument('--years', type=str, default=None,
                       help='Years to include in training (comma-separated, e.g., "2019,2020,2021"). If not specified, use all years 2017-2024')

    args = parser.parse_args()

    # Parse years
    if args.years is not None:
        years = [int(year.strip()) for year in args.years.split(',')]
        logger.info(f"Using specified years: {years}")
    else:
        years = None  # Will default to 2017-2024 in load_data method
        logger.info("Using default years: 2017-2024")

    # Validate arguments
    if args.disable_main_training_data and not args.use_extra_training_data:
        parser.error("Cannot disable main training data without enabling extra training data (--use_extra_training_data)")

    # Initialize trainer
    trainer = PVDetectionTrainer(
        data_dir=args.data_dir,
        pos_neg_ratio=args.pos_neg_ratio,
        max_positive_samples=args.max_positive_samples,
        random_seed=args.random_seed,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        use_extra_training_data=args.use_extra_training_data,
        extra_data_dir=args.extra_data_dir,
        uk_data_dir=args.uk_data_dir,
        disable_main_training_data=args.disable_main_training_data
    )

    # Load data
    trainer.load_data(years=years)

    # Train models
    if 'lightgbm' in args.models:
        trainer.train_lightgbm(use_balanced_sampling=args.balanced_sampling, quick_eval=args.quick_eval)

    if 'xgboost' in args.models:
        trainer.train_xgboost(use_balanced_sampling=args.balanced_sampling, quick_eval=args.quick_eval)

    # Compare models if multiple were trained
    trainer.compare_models()

    # Save models
    trainer.save_models(args.output_dir)

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()