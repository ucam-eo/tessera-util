import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import time
import logging
import argparse
import json
from typing import Dict, Tuple
from tqdm import tqdm

from patch_data_preprocessing import (
    load_patch_data, train_val_split_patches, PVPatchDataset,
    create_balanced_dataset
)
from cnn_models import create_model
from evaluation import (
    calculate_metrics, print_detailed_metrics, find_best_threshold
)
from extra_data_loader import ExtraDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CNNTrainer:
    """
    Trainer class for CNN-based PV detection.
    """

    def __init__(self, config: Dict, years: list = None, use_cache: bool = False, cache_dir: str = None,
                 use_extra_training_data: bool = False, extra_data_dir: str = None, uk_data_dir: str = None,
                 disable_main_training_data: bool = False):
        """
        Initialize trainer.

        Args:
            config: training configuration dictionary
            years: list of years to include in training
            use_cache: whether to use cached data
            cache_dir: directory to store cache files
            use_extra_training_data: whether to use extra training data
            extra_data_dir: directory containing extra training data labels
            uk_data_dir: directory containing UK data for extra training data
            disable_main_training_data: whether to disable main training data and only use extra data
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Multi-year and cache settings
        self.years = years if years is not None else [2019]
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Extra training data settings
        self.use_extra_training_data = use_extra_training_data
        self.extra_data_dir = extra_data_dir
        self.uk_data_dir = uk_data_dir
        self.disable_main_training_data = disable_main_training_data
        
        logger.info(f"Training years: {self.years}")
        if self.use_extra_training_data:
            logger.info(f"Using extra training data from: {self.extra_data_dir}")
        if self.disable_main_training_data:
            logger.info("Main training data disabled - using only extra training data")
        logger.info(f"Use cache: {self.use_cache}")
        if self.cache_dir:
            logger.info(f"Cache directory: {self.cache_dir}")

        # Initialize model
        self.model = create_model(
            model_type=config['model_type'],
            input_channels=config['input_channels'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        ).to(self.device)

        # Loss function with class weights
        if config['use_class_weights']:
            # Will be set after loading data
            self.criterion = None
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")

        # Learning rate scheduler
        if config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['num_epochs']
            )
        elif config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config['step_size'], gamma=config['gamma']
            )
        elif config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=config['patience'], factor=config['gamma']
            )
        else:
            self.scheduler = None

        # Training history
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rates': []
        }

        self.best_val_f1 = 0.0
        self.best_model_state = None

    def load_data(self):
        """Load and prepare data."""
        logger.info("Loading patch data...")

        # Initialize data structure
        data = {'patches': None, 'labels': None, 'coords': None}
        
        # Load main training data unless disabled
        if not self.disable_main_training_data:
            # Load patches with multi-year and cache support
            data = load_patch_data(
                self.config['data_dir'],
                patch_size=self.config['patch_size'],
                years=self.years,
                use_cache=self.use_cache,
                cache_dir=self.cache_dir,
                pos_neg_ratio=self.config.get('pos_neg_ratio', 0.33)
            )
            logger.info(f"Loaded {len(data['patches']):,} main training patches")
        else:
            logger.info("Main training data disabled - skipping main data loading")

        # Load extra training data if requested
        extra_patches = None
        extra_labels = None
        extra_coords = None
        if self.use_extra_training_data:
            logger.info("Loading extra training data...")
            extra_loader = ExtraDataLoader(self.extra_data_dir, self.uk_data_dir)
            extra_data = extra_loader.load_all_extra_patches(self.years)
            
            if extra_data['patches'].shape[0] > 0:
                extra_patches = extra_data['patches']
                extra_labels = extra_data['labels']
                extra_coords = extra_data['coords']
                
                logger.info(f"Loaded {len(extra_patches):,} extra patches")
                
                # Print extra data statistics by year
                logger.info("Extra training data statistics by year:")
                for year in self.years:
                    year_mask = extra_data['years'] == year
                    if np.any(year_mask):
                        year_labels = extra_labels[year_mask]
                        pos_count = np.sum(year_labels == 1)
                        neg_count = np.sum(year_labels == 0)
                        logger.info(f"  Year {year}: {pos_count:,} positive, {neg_count:,} negative samples")
                
                # Handle data combination based on whether main data is disabled
                if self.disable_main_training_data:
                    # Use only extra data
                    logger.info("Using only extra training data...")
                    data['patches'] = extra_patches
                    data['labels'] = extra_labels
                    data['coords'] = extra_coords
                else:
                    # Combine with original data
                    logger.info("Combining original and extra training data...")
                    data['patches'] = np.concatenate([data['patches'], extra_patches], axis=0)
                    data['labels'] = np.concatenate([data['labels'], extra_labels], axis=0)
                    data['coords'] = np.concatenate([data['coords'], extra_coords], axis=0)
                
                logger.info(f"Final dataset size: {len(data['patches']):,} patches")
            else:
                logger.info("No extra training data found for the specified years")
                if self.disable_main_training_data:
                    raise ValueError("No training data available: main data disabled and no extra data found")
        elif self.disable_main_training_data:
            raise ValueError("Cannot disable main training data without enabling extra training data")

        # Validate that we have data
        if data['patches'] is None or len(data['patches']) == 0:
            raise ValueError("No training data loaded")

        # In cache mode, data is already balanced, so we skip train_val_split_patches
        # and balanced sampling
        if self.use_cache:
            logger.info("Using cached balanced data, skipping train/val split and balanced sampling")
            
            # For cache mode, we use the data as-is for training
            # and create a simple validation split
            n_samples = len(data['patches'])
            val_size = int(n_samples * self.config['val_ratio'])
            
            # Simple random split
            np.random.seed(self.config['random_seed'])
            indices = np.random.permutation(n_samples)
            
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            train_patches = data['patches'][train_indices]
            train_labels = data['labels'][train_indices]
            
            val_patches = data['patches'][val_indices]
            val_labels = data['labels'][val_indices]
            
            logger.info(f"Cache mode - Training patches: {len(train_patches):,}")
            logger.info(f"Cache mode - Validation patches: {len(val_patches):,}")
            
        else:
            # Original mode: split data first, then apply balanced sampling
            train_patches, train_labels, train_coords, val_patches, val_labels, val_coords = train_val_split_patches(
                data['patches'], data['labels'], data['coords'],
                val_ratio=self.config['val_ratio'],
                random_seed=self.config['random_seed']
            )

            # Apply balanced sampling if requested
            if self.config['use_balanced_sampling']:
                logger.info("Creating balanced training dataset...")
                train_patches, train_labels, train_coords = create_balanced_dataset(
                    train_patches, train_labels, train_coords,
                    max_neg_pos_ratio=self.config['max_neg_pos_ratio'],
                    random_seed=self.config['random_seed']
                )

                logger.info("Creating balanced validation dataset...")
                val_patches, val_labels, val_coords = create_balanced_dataset(
                    val_patches, val_labels, val_coords,
                    max_neg_pos_ratio=self.config['max_neg_pos_ratio'],
                    random_seed=self.config['random_seed'] + 1  # Different seed for val
                )

        # Create datasets
        self.train_dataset = PVPatchDataset(
            train_patches, train_labels,
            normalize=self.config['normalize_patches']
        )
        self.val_dataset = PVPatchDataset(
            val_patches, val_labels,
            normalize=self.config['normalize_patches']
        )

        # Set up class weights if requested
        if self.config['use_class_weights']:
            unique_labels, counts = np.unique(train_labels, return_counts=True)
            total_samples = len(train_labels)
            n_classes = len(unique_labels)

            class_weights = []
            for i in range(n_classes):
                if i in unique_labels:
                    idx = np.where(unique_labels == i)[0][0]
                    weight = total_samples / (n_classes * counts[idx])
                    class_weights.append(weight)
                else:
                    class_weights.append(1.0)

            class_weights = torch.FloatTensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )

        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")

        # Print final class distribution
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        logger.info("Final training class distribution:")
        for val, count in zip(train_unique, train_counts):
            logger.info(f"  Class {val}: {count:,} patches ({100 * count / len(train_labels):.2f}%)")

        val_unique, val_counts = np.unique(val_labels, return_counts=True)
        logger.info("Final validation class distribution:")
        for val, count in zip(val_unique, val_counts):
            logger.info(f"  Class {val}: {count:,} patches ({100 * count / len(val_labels):.2f}%)")

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self) -> Tuple[float, float, float, Dict]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_probabilities = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Store predictions and probabilities for detailed metrics
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1
                all_targets.extend(targets.cpu().numpy())

                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        # Calculate detailed metrics
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)

        metrics = calculate_metrics(all_targets, all_predictions, all_probabilities)
        val_f1 = metrics['f1']

        return epoch_loss, epoch_acc, val_f1, metrics

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(self.config['num_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_f1, val_metrics = self.validate_epoch()

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['val_f1'].append(val_f1)
            self.train_history['learning_rates'].append(current_lr)

            # Log epoch results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                logger.info(f"New best validation F1: {val_f1:.4f}")

            # Early stopping
            if self.config['early_stopping']:
                if epoch >= self.config['early_stopping_patience']:
                    recent_f1 = self.train_history['val_f1'][-self.config['early_stopping_patience']:]
                    if all(f1 <= self.best_val_f1 for f1 in recent_f1):
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {training_time:.2f} seconds")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

    def evaluate_final_model(self, quick_eval: bool = False):
        """Evaluate the final model and print detailed metrics."""
        logger.info("Evaluating final model...")

        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Final evaluation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)

        if quick_eval:
            # Quick evaluation with fixed threshold
            best_threshold = 0.999
            optimal_predictions = (all_probabilities >= best_threshold).astype(int)
            
            print_detailed_metrics(
                all_targets, optimal_predictions, all_probabilities,
                f"CNN Model - Validation Set (Quick Eval, Threshold {best_threshold})"
            )
            
            best_metrics = calculate_metrics(all_targets, optimal_predictions, all_probabilities)
            logger.info(f"Quick evaluation completed with threshold {best_threshold}")
        else:
            # Find optimal threshold
            best_threshold, best_metrics = find_best_threshold(
                all_targets, all_probabilities, metric='f1'
            )

            # Calculate metrics with optimal threshold
            optimal_predictions = (all_probabilities >= best_threshold).astype(int)

            print_detailed_metrics(
                all_targets, optimal_predictions, all_probabilities,
                "CNN Model - Validation Set (Optimal Threshold)"
            )

        return best_threshold, best_metrics

    def save_model(self, output_dir: str, threshold: float = None):
        """Save the trained model and training history."""
        os.makedirs(output_dir, exist_ok=True)

        # Save model state dict
        model_path = os.path.join(output_dir, 'cnn_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_f1': self.best_val_f1,
            'optimal_threshold': threshold
        }, model_path)

        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)

        # Save config
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Training history saved to: {history_path}")
        logger.info(f"Config saved to: {config_path}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CNN for PV detection')

    # Data parameters
    parser.add_argument('--data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1',
                       help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection/cnn_models',
                       help='Directory to save trained models')
    parser.add_argument('--years', type=str, default='2017,2018,2019,2020,2021,2022,2023,2024',
                       help='Years to include in training (comma-separated, e.g., "2019,2020,2021")')
    parser.add_argument('--use_cache', action='store_true',
                       help='Use cached data for faster loading')
    parser.add_argument('--cache_dir', type=str, default='/maps/zf281/btfm4rs/src/pv_detection/cnn_cache',
                       help='Directory to store cache files')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='deep',
                       choices=['simple', 'deep'], help='CNN model type')
    parser.add_argument('--patch_size', type=int, default=3,
                       help='Patch size for CNN input')
    parser.add_argument('--input_channels', type=int, default=128,
                       help='Number of input channels')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.999,
                       help='Momentum for SGD optimizer')

    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                       help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for ReduceLROnPlateau scheduler')

    # Data handling
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--use_balanced_sampling', action='store_true',
                       help='Create balanced dataset by subsampling negatives')
    parser.add_argument('--max_neg_pos_ratio', type=float, default=9.0,
                       help='Maximum ratio of negative to positive samples (for balanced sampling)')
    parser.add_argument('--pos_neg_ratio', type=float, default=0.2,
                       help='Ratio of positive to negative samples for cache mode')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights in loss function')
    parser.add_argument('--normalize_patches', action='store_true', default=False,
                       help='Normalize patch data')

    # Training settings
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--quick_eval', action='store_true', default=False,
                       help='Use quick evaluation (default threshold 0.999) instead of threshold optimization')
    
    # Extra training data
    parser.add_argument('--use_extra_training_data', action='store_true',
                       help='Use additional training data from extra_training_data directory')
    parser.add_argument('--extra_data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data',
                       help='Directory containing extra training data labels')
    parser.add_argument('--uk_data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/uk',
                       help='Directory containing UK data for extra training data')
    parser.add_argument('--disable_main_training_data', action='store_true',
                       help='Disable main training data from --data_dir, only use extra training data')

    args = parser.parse_args()

    # Parse years
    years = [int(year.strip()) for year in args.years.split(',')]

    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    # Create config dictionary
    config = {
        'data_dir': args.data_dir,
        'model_type': args.model_type,
        'patch_size': args.patch_size,
        'input_channels': args.input_channels,
        'num_classes': 2,
        'dropout_rate': args.dropout_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'momentum': args.momentum,
        'scheduler': args.scheduler if args.scheduler != 'none' else None,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'patience': args.patience,
        'val_ratio': args.val_ratio,
        'use_balanced_sampling': args.use_balanced_sampling,
        'max_neg_pos_ratio': args.max_neg_pos_ratio,
        'pos_neg_ratio': args.pos_neg_ratio,
        'use_class_weights': args.use_class_weights,
        'normalize_patches': args.normalize_patches,
        'num_workers': args.num_workers,
        'early_stopping': args.early_stopping,
        'early_stopping_patience': args.early_stopping_patience,
        'random_seed': args.random_seed
    }

    logger.info("Training configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Initialize trainer
    trainer = CNNTrainer(
        config, 
        years=years, 
        use_cache=args.use_cache, 
        cache_dir=args.cache_dir,
        use_extra_training_data=args.use_extra_training_data,
        extra_data_dir=args.extra_data_dir,
        uk_data_dir=args.uk_data_dir,
        disable_main_training_data=args.disable_main_training_data
    )

    # Load data
    trainer.load_data()

    # Train model
    trainer.train()

    # Final evaluation
    threshold, metrics = trainer.evaluate_final_model(quick_eval=args.quick_eval)

    # Save model
    trainer.save_model(args.output_dir, threshold)

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()