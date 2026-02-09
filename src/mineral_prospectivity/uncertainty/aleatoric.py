"""
Aleatoric uncertainty estimation.

Aleatoric uncertainty represents data uncertainty - inherent noise in the data
that cannot be reduced by collecting more training data.
"""

import torch
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AleatoricUncertainty:
    """
    Aleatoric (data) uncertainty estimation for mineral prospectivity predictions.
    
    This class provides methods to estimate and analyze aleatoric uncertainty,
    which is modeled through the VAE's decoder outputting both mean and variance.
    
    Aleatoric uncertainty captures:
    - Measurement noise in geophysical/geochemical data
    - Natural variability in geological features
    - Label noise in known mineral deposits
    """
    
    @staticmethod
    def compute_from_model_output(
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute aleatoric uncertainty from model predictions.
        
        Args:
            pred_mean: Predicted mean values
            pred_logvar: Predicted log variance
            
        Returns:
            Aleatoric uncertainty (standard deviation)
        """
        # Convert log variance to standard deviation
        aleatoric_std = torch.exp(0.5 * pred_logvar)
        return aleatoric_std
    
    @staticmethod
    def confidence_interval(
        pred_mean: torch.Tensor,
        aleatoric_std: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence intervals based on aleatoric uncertainty.
        
        Args:
            pred_mean: Predicted mean values
            aleatoric_std: Aleatoric standard deviation
            confidence_level: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Dictionary with lower and upper bounds
        """
        from scipy import stats
        
        # Get z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = pred_mean - z_score * aleatoric_std
        upper_bound = pred_mean + z_score * aleatoric_std
        
        # Clip to valid probability range [0, 1]
        lower_bound = torch.clamp(lower_bound, 0, 1)
        upper_bound = torch.clamp(upper_bound, 0, 1)
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    @staticmethod
    def analyze_uncertainty_distribution(
        aleatoric_uncertainty: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze the distribution of aleatoric uncertainty.
        
        Args:
            aleatoric_uncertainty: Array of aleatoric uncertainty values
            
        Returns:
            Dictionary with statistics
        """
        stats_dict = {
            'mean': float(np.mean(aleatoric_uncertainty)),
            'median': float(np.median(aleatoric_uncertainty)),
            'std': float(np.std(aleatoric_uncertainty)),
            'min': float(np.min(aleatoric_uncertainty)),
            'max': float(np.max(aleatoric_uncertainty)),
            'q25': float(np.percentile(aleatoric_uncertainty, 25)),
            'q75': float(np.percentile(aleatoric_uncertainty, 75)),
            'q95': float(np.percentile(aleatoric_uncertainty, 95)),
            'q99': float(np.percentile(aleatoric_uncertainty, 99))
        }
        
        logger.info(f"Aleatoric uncertainty statistics: {stats_dict}")
        return stats_dict
    
    @staticmethod
    def identify_high_uncertainty_regions(
        aleatoric_uncertainty: np.ndarray,
        threshold_percentile: float = 90
    ) -> np.ndarray:
        """
        Identify regions with high aleatoric uncertainty.
        
        Args:
            aleatoric_uncertainty: Uncertainty values
            threshold_percentile: Percentile threshold for "high" uncertainty
            
        Returns:
            Boolean mask indicating high uncertainty regions
        """
        threshold = np.percentile(aleatoric_uncertainty, threshold_percentile)
        high_uncertainty_mask = aleatoric_uncertainty > threshold
        
        n_high = high_uncertainty_mask.sum()
        logger.info(
            f"Identified {n_high} ({100*n_high/len(aleatoric_uncertainty):.1f}%) "
            f"high aleatoric uncertainty regions"
        )
        
        return high_uncertainty_mask
    
    @staticmethod
    def uncertainty_weighted_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        log_variance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty-weighted loss (negative log likelihood).
        
        This loss automatically learns to predict uncertainty by penalizing
        confident wrong predictions more than uncertain predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            log_variance: Predicted log variance
            
        Returns:
            Loss value
        """
        # Negative log likelihood assuming Gaussian distribution
        loss = 0.5 * (
            log_variance +
            (targets - predictions) ** 2 / torch.exp(log_variance)
        )
        
        return loss.mean()
