"""
Performance metrics for mineral prospectivity predictions.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    accuracy_score
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Comprehensive performance metrics for mineral prospectivity models.
    
    Includes:
    - ROC-AUC and PR-AUC
    - Precision, Recall, F1
    - Capture efficiency curves
    - Economic value metrics
    """
    
    @staticmethod
    def compute_roc_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute ROC curve metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            Dictionary with ROC metrics
        """
        try:
            auc = roc_auc_score(y_true, y_pred)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            
            # Find optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            return {
                'roc_auc': float(auc),
                'optimal_threshold': float(optimal_threshold),
                'tpr_at_optimal': float(tpr[optimal_idx]),
                'fpr_at_optimal': float(fpr[optimal_idx])
            }
        except Exception as e:
            logger.error(f"Error computing ROC metrics: {e}")
            return {'roc_auc': 0.0}
    
    @staticmethod
    def compute_pr_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Precision-Recall curve metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            Dictionary with PR metrics
        """
        try:
            ap = average_precision_score(y_true, y_pred)
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            
            # F1 score at different thresholds
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            best_f1_idx = np.argmax(f1_scores)
            
            return {
                'average_precision': float(ap),
                'best_f1': float(f1_scores[best_f1_idx]),
                'precision_at_best_f1': float(precision[best_f1_idx]),
                'recall_at_best_f1': float(recall[best_f1_idx]),
                'threshold_at_best_f1': float(thresholds[best_f1_idx])
            }
        except Exception as e:
            logger.error(f"Error computing PR metrics: {e}")
            return {'average_precision': 0.0}
    
    @staticmethod
    def compute_capture_efficiency(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        percentiles: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Compute capture efficiency metrics.
        
        Capture efficiency measures what percentage of positive cases
        are captured in the top X% of predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            percentiles: List of percentiles to evaluate (default: [1, 5, 10, 20])
            
        Returns:
            Dictionary with capture efficiency at different percentiles
        """
        if percentiles is None:
            percentiles = [1, 5, 10, 20]
        
        # Sort by prediction (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        total_positives = y_true.sum()
        if total_positives == 0:
            return {f'capture_at_{p}pct': 0.0 for p in percentiles}
        
        capture_efficiency = {}
        
        for p in percentiles:
            # Number of samples in top p%
            n_top = int(len(y_true) * p / 100)
            if n_top == 0:
                n_top = 1
            
            # Count positives in top p%
            positives_in_top = sorted_labels[:n_top].sum()
            
            # Capture efficiency
            efficiency = positives_in_top / total_positives
            capture_efficiency[f'capture_at_{p}pct'] = float(efficiency)
        
        return capture_efficiency
    
    @staticmethod
    def compute_confusion_matrix_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute metrics from confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary with confusion matrix metrics
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        # Compute metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    @staticmethod
    def compute_economic_value(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        exploration_cost: float = 1.0,
        discovery_value: float = 100.0,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute economic value metrics for exploration decisions.
        
        Args:
            y_true: Ground truth labels (1 = deposit, 0 = no deposit)
            y_pred: Predicted probabilities
            exploration_cost: Cost per exploration
            discovery_value: Value of a discovery
            threshold: Decision threshold
            
        Returns:
            Dictionary with economic metrics
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Count outcomes
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
        tn = ((y_pred_binary == 0) & (y_true == 0)).sum()
        
        # Economic calculations
        exploration_locations = tp + fp
        total_exploration_cost = exploration_locations * exploration_cost
        total_discovery_value = tp * discovery_value
        net_value = total_discovery_value - total_exploration_cost
        
        # Value per exploration
        value_per_exploration = net_value / exploration_locations if exploration_locations > 0 else 0
        
        # Success rate
        success_rate = tp / exploration_locations if exploration_locations > 0 else 0
        
        return {
            'net_economic_value': float(net_value),
            'total_exploration_cost': float(total_exploration_cost),
            'total_discovery_value': float(total_discovery_value),
            'value_per_exploration': float(value_per_exploration),
            'success_rate': float(success_rate),
            'locations_explored': int(exploration_locations),
            'discoveries': int(tp),
            'missed_deposits': int(fn)
        }
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Compute comprehensive set of performance metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            uncertainties: Optional uncertainty estimates
            threshold: Classification threshold
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # ROC metrics
        metrics.update(PerformanceMetrics.compute_roc_metrics(y_true, y_pred))
        
        # PR metrics
        metrics.update(PerformanceMetrics.compute_pr_metrics(y_true, y_pred))
        
        # Capture efficiency
        metrics['capture_efficiency'] = PerformanceMetrics.compute_capture_efficiency(
            y_true, y_pred
        )
        
        # Confusion matrix metrics
        metrics['classification'] = PerformanceMetrics.compute_confusion_matrix_metrics(
            y_true, y_pred, threshold
        )
        
        # Economic value
        metrics['economic'] = PerformanceMetrics.compute_economic_value(
            y_true, y_pred, threshold=threshold
        )
        
        # Uncertainty-based metrics if available
        if uncertainties is not None:
            metrics['uncertainty_correlation'] = {
                'pred_uncertainty_corr': float(
                    np.corrcoef(y_pred, uncertainties)[0, 1]
                ),
                'error_uncertainty_corr': float(
                    np.corrcoef(np.abs(y_pred - y_true), uncertainties)[0, 1]
                )
            }
        
        logger.info("Computed comprehensive performance metrics")
        return metrics
