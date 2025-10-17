import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, confusion_matrix,
    classification_report
)
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for binary classification.

    Args:
        y_true: true labels (0/1)
        y_pred: predicted labels (0/1)
        y_pred_proba: predicted probabilities for class 1 (optional)

    Returns:
        Dictionary containing all metrics
    """
    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp

    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # False Positive Rate
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    # AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            metrics['auc_roc'] = None

    return metrics

def print_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None,
                          dataset_name: str = "Dataset") -> None:
    """
    Print detailed evaluation metrics in a formatted way.

    Args:
        y_true: true labels
        y_pred: predicted labels
        y_pred_proba: predicted probabilities (optional)
        dataset_name: name of the dataset (for display)
    """
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS - {dataset_name.upper()}")
    print(f"{'='*60}")

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    print(f"\nGround Truth Distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:,} samples ({100*count/len(y_true):.2f}%)")

    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print(f"\nPredicted Distribution:")
    for cls, count in zip(unique_pred, counts_pred):
        print(f"  Class {cls}: {count:,} samples ({100*count/len(y_pred):.2f}%)")

    # Main metrics
    print(f"\n{'-'*40}")
    print(f"MAIN CLASSIFICATION METRICS")
    print(f"{'-'*40}")
    print(f"Accuracy:     {metrics['accuracy']:.4f} ({100*metrics['accuracy']:.2f}%)")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")

    if metrics.get('auc_roc') is not None:
        print(f"AUC-ROC:      {metrics['auc_roc']:.4f}")

    # Confusion matrix
    print(f"\n{'-'*40}")
    print(f"CONFUSION MATRIX")
    print(f"{'-'*40}")
    print(f"True Negatives (TN):  {metrics['true_negatives']:,}")
    print(f"False Positives (FP): {metrics['false_positives']:,}")
    print(f"False Negatives (FN): {metrics['false_negatives']:,}")
    print(f"True Positives (TP):  {metrics['true_positives']:,}")

    # Error analysis for imbalanced data
    print(f"\n{'-'*40}")
    print(f"ERROR ANALYSIS")
    print(f"{'-'*40}")
    total_pos = metrics['true_positives'] + metrics['false_negatives']
    total_neg = metrics['true_negatives'] + metrics['false_positives']

    if total_pos > 0:
        pos_error_rate = metrics['false_negatives'] / total_pos
        print(f"Positive class error rate: {pos_error_rate:.4f} ({100*pos_error_rate:.2f}%)")

    if total_neg > 0:
        neg_error_rate = metrics['false_positives'] / total_neg
        print(f"Negative class error rate: {neg_error_rate:.4f} ({100*neg_error_rate:.2f}%)")

    print(f"{'='*60}\n")

def compare_models(results: Dict[str, Dict[str, float]]) -> None:
    """
    Compare metrics across multiple models.

    Args:
        results: Dictionary where keys are model names and values are metric dictionaries
    """
    if not results:
        return

    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}")

    # Get all metric names
    all_metrics = set()
    for model_metrics in results.values():
        all_metrics.update(model_metrics.keys())

    # Filter to main metrics for comparison
    main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    comparison_metrics = [m for m in main_metrics if m in all_metrics]

    # Print header
    print(f"{'Model':<20}", end="")
    for metric in comparison_metrics:
        print(f"{metric.capitalize():<12}", end="")
    print()

    print("-" * (20 + 12 * len(comparison_metrics)))

    # Print results for each model
    for model_name, metrics in results.items():
        print(f"{model_name:<20}", end="")
        for metric in comparison_metrics:
            value = metrics.get(metric, 0)
            if value is not None:
                print(f"{value:<12.4f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()

    print(f"{'='*80}\n")

def evaluate_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                      thresholds: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Evaluate different classification thresholds.

    Args:
        y_true: true labels
        y_pred_proba: predicted probabilities
        thresholds: array of thresholds to evaluate

    Returns:
        thresholds: array of thresholds
        metrics_by_threshold: dictionary containing metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    metrics_by_threshold = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': []
    }

    total_thresholds = len(thresholds)
    logger.info(f"Evaluating {total_thresholds} thresholds for optimal performance...")

    for i, threshold in enumerate(thresholds):
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred)

        metrics_by_threshold['precision'].append(metrics['precision'])
        metrics_by_threshold['recall'].append(metrics['recall'])
        metrics_by_threshold['f1'].append(metrics['f1'])
        metrics_by_threshold['accuracy'].append(metrics['accuracy'])

        # Progress logging every 10 thresholds or at the end
        if (i + 1) % 10 == 0 or (i + 1) == total_thresholds:
            progress = (i + 1) / total_thresholds * 100
            logger.info(f"Progress: {i + 1}/{total_thresholds} ({progress:.1f}%) - "
                       f"Current threshold: {threshold:.3f}, F1: {metrics['f1']:.4f}")

    # Convert to numpy arrays
    for key in metrics_by_threshold:
        metrics_by_threshold[key] = np.array(metrics_by_threshold[key])

    logger.info("Threshold evaluation completed.")
    return thresholds, metrics_by_threshold

def find_best_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                       metric: str = 'f1') -> Tuple[float, Dict[str, float]]:
    """
    Find the best threshold based on a specific metric.

    Args:
        y_true: true labels
        y_pred_proba: predicted probabilities
        metric: metric to optimize ('f1', 'precision', 'recall', 'accuracy')

    Returns:
        best_threshold: optimal threshold value
        best_metrics: metrics at the optimal threshold
    """
    logger.info(f"Starting threshold optimization for metric: {metric}")
    thresholds = np.arange(0.01, 1.0, 0.01)
    thresholds, metrics_by_threshold = evaluate_threshold(y_true, y_pred_proba, thresholds)

    # Find best threshold
    best_idx = np.argmax(metrics_by_threshold[metric])
    best_threshold = thresholds[best_idx]

    # Calculate metrics at best threshold
    y_pred_best = (y_pred_proba >= best_threshold).astype(int)
    best_metrics = calculate_metrics(y_true, y_pred_best, y_pred_proba)

    logger.info(f"Threshold optimization completed!")
    logger.info(f"Best threshold for {metric}: {best_threshold:.3f}")
    logger.info(f"Best {metric}: {best_metrics[metric]:.4f}")

    return best_threshold, best_metrics

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])  # Imbalanced
    y_pred_proba = np.random.random(1000)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print_detailed_metrics(y_true, y_pred, y_pred_proba, "Test Dataset")

    # Find best threshold
    best_threshold, best_metrics = find_best_threshold(y_true, y_pred_proba, 'f1')
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {best_metrics['f1']:.4f}")