"""
Epistemic uncertainty estimation.

Epistemic uncertainty represents model uncertainty - uncertainty that can be
reduced by training on more data or using better models. It captures what
the model doesn't know.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EpistemicUncertainty:
    """
    Epistemic (model) uncertainty estimation for mineral prospectivity predictions.
    
    This class provides methods to estimate and analyze epistemic uncertainty,
    which is estimated through ensemble disagreement or Monte Carlo dropout.
    
    Epistemic uncertainty captures:
    - Model parameter uncertainty
    - Model structural uncertainty
    - Lack of training data in certain regions
    - Extrapolation beyond training distribution
    """
    
    @staticmethod
    def compute_from_ensemble(
        ensemble_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute epistemic uncertainty from ensemble predictions.
        
        Args:
            ensemble_predictions: Predictions from ensemble (n_models, n_samples, ...)
            
        Returns:
            Epistemic uncertainty (standard deviation across models)
        """
        epistemic_std = ensemble_predictions.std(dim=0)
        return epistemic_std
    
    @staticmethod
    def compute_ensemble_statistics(
        ensemble_predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive statistics from ensemble predictions.
        
        Args:
            ensemble_predictions: Predictions from ensemble (n_models, n_samples, ...)
            
        Returns:
            Dictionary with mean, std, min, max, etc.
        """
        return {
            'mean': ensemble_predictions.mean(dim=0),
            'std': ensemble_predictions.std(dim=0),
            'min': ensemble_predictions.min(dim=0)[0],
            'max': ensemble_predictions.max(dim=0)[0],
            'median': ensemble_predictions.median(dim=0)[0],
            'q25': ensemble_predictions.quantile(0.25, dim=0),
            'q75': ensemble_predictions.quantile(0.75, dim=0)
        }
    
    @staticmethod
    def mutual_information(
        ensemble_predictions: torch.Tensor,
        num_bins: int = 50
    ) -> torch.Tensor:
        """
        Compute mutual information as epistemic uncertainty metric.
        
        Mutual information between predictions and model parameters
        is a principled measure of epistemic uncertainty.
        
        Args:
            ensemble_predictions: Predictions from ensemble (n_models, n_samples, ...)
            num_bins: Number of bins for histogram
            
        Returns:
            Mutual information values
        """
        # This is a simplified approximation
        # True MI computation would require the full predictive distribution
        
        n_models = ensemble_predictions.shape[0]
        
        # Compute entropy of mean prediction
        mean_pred = ensemble_predictions.mean(dim=0)
        mean_pred_np = mean_pred.cpu().numpy().flatten()
        hist_mean, _ = np.histogram(mean_pred_np, bins=num_bins, density=True)
        hist_mean = hist_mean + 1e-10  # Avoid log(0)
        entropy_mean = -np.sum(hist_mean * np.log(hist_mean))
        
        # Compute mean entropy of individual predictions
        entropy_individual = 0.0
        for i in range(n_models):
            pred_i = ensemble_predictions[i].cpu().numpy().flatten()
            hist_i, _ = np.histogram(pred_i, bins=num_bins, density=True)
            hist_i = hist_i + 1e-10
            entropy_individual += -np.sum(hist_i * np.log(hist_i))
        
        entropy_individual /= n_models
        
        # Mutual information (approximation)
        mi = entropy_mean - entropy_individual
        
        return torch.tensor(mi)
    
    @staticmethod
    def prediction_entropy(
        ensemble_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute predictive entropy as epistemic uncertainty.
        
        Args:
            ensemble_predictions: Predictions from ensemble
            
        Returns:
            Entropy values
        """
        # Average predictions to get probability
        mean_pred = ensemble_predictions.mean(dim=0)
        
        # Compute binary entropy for binary classification
        # H(p) = -p*log(p) - (1-p)*log(1-p)
        eps = 1e-10
        mean_pred = torch.clamp(mean_pred, eps, 1 - eps)
        
        entropy = -(
            mean_pred * torch.log(mean_pred) +
            (1 - mean_pred) * torch.log(1 - mean_pred)
        )
        
        return entropy
    
    @staticmethod
    def analyze_uncertainty_distribution(
        epistemic_uncertainty: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze the distribution of epistemic uncertainty.
        
        Args:
            epistemic_uncertainty: Array of epistemic uncertainty values
            
        Returns:
            Dictionary with statistics
        """
        stats_dict = {
            'mean': float(np.mean(epistemic_uncertainty)),
            'median': float(np.median(epistemic_uncertainty)),
            'std': float(np.std(epistemic_uncertainty)),
            'min': float(np.min(epistemic_uncertainty)),
            'max': float(np.max(epistemic_uncertainty)),
            'q25': float(np.percentile(epistemic_uncertainty, 25)),
            'q75': float(np.percentile(epistemic_uncertainty, 75)),
            'q95': float(np.percentile(epistemic_uncertainty, 95)),
            'q99': float(np.percentile(epistemic_uncertainty, 99))
        }
        
        logger.info(f"Epistemic uncertainty statistics: {stats_dict}")
        return stats_dict
    
    @staticmethod
    def identify_high_uncertainty_regions(
        epistemic_uncertainty: np.ndarray,
        threshold_percentile: float = 90
    ) -> np.ndarray:
        """
        Identify regions with high epistemic uncertainty.
        
        These are regions where the model is uncertain and would benefit
        most from additional training data.
        
        Args:
            epistemic_uncertainty: Uncertainty values
            threshold_percentile: Percentile threshold
            
        Returns:
            Boolean mask indicating high uncertainty regions
        """
        threshold = np.percentile(epistemic_uncertainty, threshold_percentile)
        high_uncertainty_mask = epistemic_uncertainty > threshold
        
        n_high = high_uncertainty_mask.sum()
        logger.info(
            f"Identified {n_high} ({100*n_high/len(epistemic_uncertainty):.1f}%) "
            f"high epistemic uncertainty regions"
        )
        
        return high_uncertainty_mask
    
    @staticmethod
    def uncertainty_decomposition(
        ensemble_predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose predictive uncertainty into epistemic and aleatoric components.
        
        For binary classification with ensemble:
        Total variance = Epistemic variance + Mean aleatoric variance
        
        Args:
            ensemble_predictions: Predictions from ensemble
            
        Returns:
            Dictionary with decomposed uncertainties
        """
        # Compute epistemic uncertainty (model disagreement)
        epistemic_var = ensemble_predictions.var(dim=0)
        
        # For binary classification, aleatoric uncertainty is p(1-p)
        # Compute for each model and average
        aleatoric_vars = []
        for i in range(ensemble_predictions.shape[0]):
            p = ensemble_predictions[i]
            aleatoric_var = p * (1 - p)
            aleatoric_vars.append(aleatoric_var)
        
        mean_aleatoric_var = torch.stack(aleatoric_vars).mean(dim=0)
        
        # Total uncertainty
        total_var = epistemic_var + mean_aleatoric_var
        
        return {
            'epistemic_uncertainty': torch.sqrt(epistemic_var),
            'aleatoric_uncertainty': torch.sqrt(mean_aleatoric_var),
            'total_uncertainty': torch.sqrt(total_var),
            'epistemic_fraction': epistemic_var / (total_var + 1e-10),
            'aleatoric_fraction': mean_aleatoric_var / (total_var + 1e-10)
        }
