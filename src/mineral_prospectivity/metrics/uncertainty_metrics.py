"""
Uncertainty quality metrics.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UncertaintyMetrics:
    """
    Metrics for evaluating uncertainty quality.
    
    Includes:
    - Calibration metrics
    - Sharpness metrics
    - Uncertainty-error correlation
    - Coverage metrics
    """
    
    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures how well predicted uncertainties match actual errors.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            uncertainties: Predicted uncertainties
            num_bins: Number of bins
            
        Returns:
            ECE value
        """
        errors = np.abs(y_pred - y_true)
        
        # Bin by uncertainty
        bin_boundaries = np.percentile(uncertainties, np.linspace(0, 100, num_bins + 1))
        bin_indices = np.digitize(uncertainties, bin_boundaries[1:-1])
        
        ece = 0.0
        total_samples = len(uncertainties)
        
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_uncertainty = uncertainties[mask].mean()
                mean_error = errors[mask].mean()
                bin_weight = mask.sum() / total_samples
                
                ece += bin_weight * abs(mean_uncertainty - mean_error)
        
        return float(ece)
    
    @staticmethod
    def maximum_calibration_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE is the maximum deviation between uncertainty and error across bins.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            uncertainties: Predicted uncertainties
            num_bins: Number of bins
            
        Returns:
            MCE value
        """
        errors = np.abs(y_pred - y_true)
        
        bin_boundaries = np.percentile(uncertainties, np.linspace(0, 100, num_bins + 1))
        bin_indices = np.digitize(uncertainties, bin_boundaries[1:-1])
        
        max_error = 0.0
        
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_uncertainty = uncertainties[mask].mean()
                mean_error = errors[mask].mean()
                
                max_error = max(max_error, abs(mean_uncertainty - mean_error))
        
        return float(max_error)
    
    @staticmethod
    def sharpness(uncertainties: np.ndarray) -> float:
        """
        Compute sharpness metric.
        
        Sharpness is the average uncertainty. Lower is better (more confident).
        
        Args:
            uncertainties: Predicted uncertainties
            
        Returns:
            Sharpness value
        """
        return float(np.mean(uncertainties))
    
    @staticmethod
    def uncertainty_error_correlation(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute correlation between uncertainty and prediction error.
        
        Good uncertainty estimates should be correlated with errors.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            uncertainties: Predicted uncertainties
            
        Returns:
            Dictionary with correlation metrics
        """
        errors = np.abs(y_pred - y_true)
        
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(uncertainties, errors)
        
        # Spearman correlation (rank-based, more robust)
        spearman_corr, spearman_p = stats.spearmanr(uncertainties, errors)
        
        return {
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p)
        }
    
    @staticmethod
    def coverage(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Compute coverage: fraction of true values within confidence intervals.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            uncertainties: Predicted uncertainties (std)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Coverage fraction
        """
        # Get z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Confidence intervals
        lower = y_pred - z_score * uncertainties
        upper = y_pred + z_score * uncertainties
        
        # Check coverage
        within_interval = (y_true >= lower) & (y_true <= upper)
        coverage_fraction = within_interval.mean()
        
        return float(coverage_fraction)
    
    @staticmethod
    def negative_log_likelihood(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray
    ) -> float:
        """
        Compute negative log likelihood assuming Gaussian distribution.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            uncertainties: Predicted uncertainties (std)
            
        Returns:
            NLL value
        """
        # Gaussian NLL: 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
        variance = uncertainties ** 2 + 1e-10
        nll = 0.5 * (
            np.log(2 * np.pi * variance) +
            (y_true - y_pred) ** 2 / variance
        )
        
        return float(nll.mean())
    
    @staticmethod
    def continuous_ranked_probability_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray
    ) -> float:
        """
        Compute Continuous Ranked Probability Score (CRPS).
        
        CRPS is a proper scoring rule for probabilistic predictions.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions (mean)
            uncertainties: Predicted uncertainties (std)
            
        Returns:
            CRPS value
        """
        # For Gaussian distribution, CRPS has closed form:
        # CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
        # where z = (y_true - y_pred) / sigma
        
        z = (y_true - y_pred) / (uncertainties + 1e-10)
        
        # Standard normal CDF and PDF
        phi_z = stats.norm.cdf(z)
        pdf_z = stats.norm.pdf(z)
        
        crps = uncertainties * (
            z * (2 * phi_z - 1) +
            2 * pdf_z -
            1 / np.sqrt(np.pi)
        )
        
        return float(crps.mean())
    
    @staticmethod
    def uncertainty_quality_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute composite uncertainty quality score.
        
        Combines multiple metrics into an overall quality assessment.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            uncertainties: Predicted uncertainties
            
        Returns:
            Dictionary with quality metrics
        """
        # Compute individual metrics
        ece = UncertaintyMetrics.expected_calibration_error(
            y_true, y_pred, uncertainties
        )
        
        corr_metrics = UncertaintyMetrics.uncertainty_error_correlation(
            y_true, y_pred, uncertainties
        )
        
        coverage_95 = UncertaintyMetrics.coverage(
            y_true, y_pred, uncertainties, confidence_level=0.95
        )
        
        sharpness_val = UncertaintyMetrics.sharpness(uncertainties)
        
        nll = UncertaintyMetrics.negative_log_likelihood(
            y_true, y_pred, uncertainties
        )
        
        # Composite score (lower is better)
        # Normalize and combine
        composite = (
            ece +  # Calibration
            (1 - corr_metrics['spearman_correlation']) +  # Correlation
            abs(coverage_95 - 0.95) +  # Coverage deviation
            sharpness_val * 0.1 +  # Sharpness (weighted less)
            nll * 0.1  # NLL (weighted less)
        )
        
        return {
            'expected_calibration_error': ece,
            'uncertainty_error_correlation': corr_metrics['spearman_correlation'],
            'coverage_95': coverage_95,
            'sharpness': sharpness_val,
            'negative_log_likelihood': nll,
            'composite_quality_score': float(composite)
        }
    
    @staticmethod
    def compute_all_uncertainty_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        aleatoric_uncertainty: np.ndarray,
        epistemic_uncertainty: np.ndarray,
        total_uncertainty: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all uncertainty metrics for each uncertainty type.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            aleatoric_uncertainty: Aleatoric uncertainties
            epistemic_uncertainty: Epistemic uncertainties
            total_uncertainty: Total uncertainties
            
        Returns:
            Dictionary with metrics for each uncertainty type
        """
        metrics = {}
        
        for name, uncertainty in [
            ('aleatoric', aleatoric_uncertainty),
            ('epistemic', epistemic_uncertainty),
            ('total', total_uncertainty)
        ]:
            metrics[name] = UncertaintyMetrics.uncertainty_quality_score(
                y_true, y_pred, uncertainty
            )
        
        logger.info("Computed comprehensive uncertainty metrics")
        return metrics
