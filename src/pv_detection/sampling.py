import numpy as np
from typing import Tuple, Generator
import logging

logger = logging.getLogger(__name__)

class BalancedSampler:
    """
    Balanced sampler for handling imbalanced datasets.
    Implements 1:2 or 1:3 positive to negative ratio sampling.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 pos_neg_ratio: float = 0.5, random_seed: int = 42):
        """
        Initialize balanced sampler.

        Args:
            features: array of shape (N, C)
            labels: array of shape (N,)
            pos_neg_ratio: ratio of positive to negative samples (0.5 = 1:2, 0.33 = 1:3)
            random_seed: random seed for reproducibility
        """
        self.features = features
        self.labels = labels
        self.pos_neg_ratio = pos_neg_ratio
        self.random_seed = random_seed

        # Find positive and negative indices
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]

        self.n_pos = len(self.pos_indices)
        self.n_neg = len(self.neg_indices)

        logger.info(f"Positive samples: {self.n_pos:,}")
        logger.info(f"Negative samples: {self.n_neg:,}")
        logger.info(f"Original ratio (pos:neg): 1:{self.n_neg/self.n_pos:.1f}")
        logger.info(f"Target ratio (pos:neg): 1:{1/self.pos_neg_ratio:.1f}")

        # Set random seed
        np.random.seed(random_seed)

    def sample_balanced_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a balanced batch.

        Args:
            batch_size: total batch size

        Returns:
            batch_features: array of shape (batch_size, C)
            batch_labels: array of shape (batch_size,)
        """
        # Calculate number of positive and negative samples in batch
        n_pos_batch = int(batch_size * self.pos_neg_ratio)
        n_neg_batch = batch_size - n_pos_batch

        # Sample positive indices (with replacement if needed)
        if n_pos_batch <= self.n_pos:
            pos_batch_indices = np.random.choice(self.pos_indices, n_pos_batch, replace=False)
        else:
            pos_batch_indices = np.random.choice(self.pos_indices, n_pos_batch, replace=True)

        # Sample negative indices (with replacement if needed)
        if n_neg_batch <= self.n_neg:
            neg_batch_indices = np.random.choice(self.neg_indices, n_neg_batch, replace=False)
        else:
            neg_batch_indices = np.random.choice(self.neg_indices, n_neg_batch, replace=True)

        # Combine indices
        batch_indices = np.concatenate([pos_batch_indices, neg_batch_indices])

        # Shuffle the batch
        np.random.shuffle(batch_indices)

        # Extract features and labels
        batch_features = self.features[batch_indices]
        batch_labels = self.labels[batch_indices]

        return batch_features, batch_labels

    def generate_balanced_batches(self, batch_size: int, n_batches: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate balanced batches.

        Args:
            batch_size: size of each batch
            n_batches: number of batches to generate

        Yields:
            (batch_features, batch_labels) tuples
        """
        for _ in range(n_batches):
            yield self.sample_balanced_batch(batch_size)

    def get_class_weights(self) -> dict:
        """
        Calculate class weights for model training.

        Returns:
            Dictionary mapping class labels to weights
        """
        total_samples = self.n_pos + self.n_neg

        # Calculate weights inversely proportional to class frequency
        pos_weight = total_samples / (2 * self.n_pos)
        neg_weight = total_samples / (2 * self.n_neg)

        weights = {0: neg_weight, 1: pos_weight}

        logger.info(f"Class weights: {weights}")

        return weights

    def create_balanced_dataset(self, target_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a balanced dataset by oversampling minority class.

        Args:
            target_size: target total size of balanced dataset. If None, use all positive samples

        Returns:
            balanced_features: array of shape (target_size, C)
            balanced_labels: array of shape (target_size,)
        """
        if target_size is None:
            # Default: use all positive samples and calculate negative samples based on ratio
            n_pos_target = self.n_pos
            n_neg_target = int(self.n_pos / self.pos_neg_ratio)
            target_size = n_pos_target + n_neg_target
        else:
            n_pos_target = int(target_size * self.pos_neg_ratio)
            n_neg_target = target_size - n_pos_target

        logger.info(f"Creating balanced dataset with {n_pos_target:,} positive and {n_neg_target:,} negative samples")

        # Sample positive samples (with replacement if needed)
        if n_pos_target <= self.n_pos:
            pos_sample_indices = np.random.choice(self.pos_indices, n_pos_target, replace=False)
        else:
            pos_sample_indices = np.random.choice(self.pos_indices, n_pos_target, replace=True)

        # Sample negative samples (with replacement if needed)
        if n_neg_target <= self.n_neg:
            neg_sample_indices = np.random.choice(self.neg_indices, n_neg_target, replace=False)
        else:
            neg_sample_indices = np.random.choice(self.neg_indices, n_neg_target, replace=True)

        # Combine indices and shuffle
        all_indices = np.concatenate([pos_sample_indices, neg_sample_indices])
        np.random.shuffle(all_indices)

        # Extract features and labels
        balanced_features = self.features[all_indices]
        balanced_labels = self.labels[all_indices]

        # Verify balance
        unique, counts = np.unique(balanced_labels, return_counts=True)
        logger.info("Balanced dataset class distribution:")
        for val, count in zip(unique, counts):
            logger.info(f"  Class {val}: {count:,} samples ({100 * count / len(balanced_labels):.2f}%)")

        return balanced_features, balanced_labels

def create_sampler(features: np.ndarray, labels: np.ndarray,
                  pos_neg_ratio: float = 0.33, random_seed: int = 42) -> BalancedSampler:
    """
    Create a balanced sampler with specified positive to negative ratio.

    Args:
        features: array of shape (N, C)
        labels: array of shape (N,)
        pos_neg_ratio: ratio of positive to negative samples (0.33 = 1:3, 0.5 = 1:2)
        random_seed: random seed for reproducibility

    Returns:
        BalancedSampler instance
    """
    return BalancedSampler(features, labels, pos_neg_ratio, random_seed)

if __name__ == "__main__":
    # Test the sampler
    from data_preprocessing import load_processed_data

    base_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1"
    data = load_processed_data(base_dir)

    # Create sampler with 1:3 ratio
    sampler = create_sampler(data['train_features'], data['train_labels'], pos_neg_ratio=0.33)

    # Test batch generation
    batch_features, batch_labels = sampler.sample_balanced_batch(1000)
    print(f"Batch shape: {batch_features.shape}")
    unique, counts = np.unique(batch_labels, return_counts=True)
    print(f"Batch class distribution: {dict(zip(unique, counts))}")