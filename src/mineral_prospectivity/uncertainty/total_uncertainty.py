"""
Total uncertainty estimation combining aleatoric and epistemic components.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from .aleatoric import AleatoricUncertainty
from .epistemic import EpistemicUncertainty

logger = logging.getLogger(__name__)


class TotalUncertainty:
    """
    Combined uncertainty estimation for mineral prospectivity predictions.
    
    Total uncertainty = sqrt(Aleatoric variance + Epistemic variance)
    
    This class combines both types of uncertainty and provides methods
    for comprehensive uncertainty analysis and visualization.
    """
    
    @staticmethod
    def combine_uncertainties(
        aleatoric_uncertainty: torch.Tensor,
        epistemic_uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine aleatoric and epistemic uncertainties.
        
        Args:
            aleatoric_uncertainty: Aleatoric uncertainty (std)
            epistemic_uncertainty: Epistemic uncertainty (std)
            
        Returns:
            Total uncertainty
        """
        # Combine variances and take square root
        total_var = aleatoric_uncertainty ** 2 + epistemic_uncertainty ** 2
        total_uncertainty = torch.sqrt(total_var)
        
        return total_uncertainty
    
    @staticmethod
    def decompose_uncertainty(
        total_uncertainty: torch.Tensor,
        aleatoric_uncertainty: torch.Tensor,
        epistemic_uncertainty: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose total uncertainty into components.
        
        Args:
            total_uncertainty: Total uncertainty
            aleatoric_uncertainty: Aleatoric component
            epistemic_uncertainty: Epistemic component
            
        Returns:
            Dictionary with uncertainty components and fractions
        """
        # Compute variances
        total_var = total_uncertainty ** 2
        aleatoric_var = aleatoric_uncertainty ** 2
        epistemic_var = epistemic_uncertainty ** 2
        
        # Compute fractions
        aleatoric_fraction = aleatoric_var / (total_var + 1e-10)
        epistemic_fraction = epistemic_var / (total_var + 1e-10)
        
        return {
            'total_uncertainty': total_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_fraction': aleatoric_fraction,
            'epistemic_fraction': epistemic_fraction
        }
    
    @staticmethod
    def confidence_bounds(
        predictions: torch.Tensor,
        total_uncertainty: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence bounds using total uncertainty.
        
        Args:
            predictions: Mean predictions
            total_uncertainty: Total uncertainty (std)
            confidence_level: Confidence level
            
        Returns:
            Dictionary with bounds
        """
        from scipy import stats
        
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower = predictions - z_score * total_uncertainty
        upper = predictions + z_score * total_uncertainty
        
        # Clip to [0, 1] for probabilities
        lower = torch.clamp(lower, 0, 1)
        upper = torch.clamp(upper, 0, 1)
        
        return {
            'lower': lower,
            'upper': upper,
            'width': upper - lower
        }
    
    @staticmethod
    def uncertainty_calibration_analysis(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze calibration of uncertainty estimates.
        
        Well-calibrated uncertainties should have prediction errors
        that match the predicted uncertainty magnitude.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            targets: Ground truth labels
            num_bins: Number of bins for calibration plot
            
        Returns:
            Dictionary with calibration statistics
        """
        # Compute prediction errors
        errors = np.abs(predictions - targets)
        
        # Bin by uncertainty
        uncertainty_bins = np.percentile(
            uncertainties,
            np.linspace(0, 100, num_bins + 1)
        )
        
        bin_indices = np.digitize(uncertainties, uncertainty_bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Compute statistics per bin
        mean_uncertainty_per_bin = []
        mean_error_per_bin = []
        count_per_bin = []
        
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_uncertainty_per_bin.append(uncertainties[mask].mean())
                mean_error_per_bin.append(errors[mask].mean())
                count_per_bin.append(mask.sum())
            else:
                mean_uncertainty_per_bin.append(0)
                mean_error_per_bin.append(0)
                count_per_bin.append(0)
        
        # Calibration error (how well uncertainty matches error)
        calibration_error = np.abs(
            np.array(mean_uncertainty_per_bin) - np.array(mean_error_per_bin)
        ).mean()
        
        return {
            'mean_uncertainty_per_bin': np.array(mean_uncertainty_per_bin),
            'mean_error_per_bin': np.array(mean_error_per_bin),
            'count_per_bin': np.array(count_per_bin),
            'calibration_error': calibration_error
        }
    
    @staticmethod
    def uncertainty_risk_analysis(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        exploration_cost: float = 1.0,
        discovery_value: float = 100.0,
        confidence_threshold: float = 0.9,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Perform risk-return analysis using uncertainty estimates.
        
        This helps prioritize exploration targets by considering both
        prospectivity and uncertainty.
        
        Args:
            predictions: Prospectivity predictions
            uncertainties: Uncertainty estimates
            exploration_cost: Cost of exploring a location
            discovery_value: Value of a successful discovery
            confidence_threshold: Minimum prediction threshold to recommend exploration
            confidence_level: Confidence level for computing lower bound (e.g. 0.95 for 95% CI)
            
        Returns:
            Dictionary with risk-return metrics
        """
        # Expected value
        expected_value = predictions * discovery_value
        
        # Risk (uncertainty-adjusted)
        # Higher uncertainty increases risk
        risk = uncertainties * discovery_value
        
        # Net expected value
        net_expected_value = expected_value - exploration_cost
        
        # Risk-adjusted return
        # Penalize high uncertainty
        risk_adjusted_return = net_expected_value - risk
        
        # Confidence-based decision
        # Only recommend if lower confidence bound > threshold
        from scipy import stats as sp_stats
        z_score = sp_stats.norm.ppf((1 + confidence_level) / 2)
        lower_bound = predictions - z_score * uncertainties
        recommend = lower_bound > confidence_threshold
        
        return {
            'expected_value': expected_value,
            'risk': risk,
            'net_expected_value': net_expected_value,
            'risk_adjusted_return': risk_adjusted_return,
            'lower_confidence_bound': lower_bound,
            'recommend_exploration': recommend
        }
    
    @staticmethod
    def uncertainty_based_sampling(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        n_samples: int,
        strategy: str = 'highest_uncertainty'
    ) -> np.ndarray:
        """
        Select samples for further investigation based on uncertainty.
        
        Useful for active learning or prioritizing exploration targets.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            n_samples: Number of samples to select
            strategy: Sampling strategy ('highest_uncertainty', 'highest_value', 'balanced')
            
        Returns:
            Indices of selected samples
        """
        if strategy == 'highest_uncertainty':
            # Select samples with highest uncertainty
            indices = np.argsort(uncertainties)[-n_samples:]
        
        elif strategy == 'highest_value':
            # Select samples with highest predicted value
            indices = np.argsort(predictions)[-n_samples:]
        
        elif strategy == 'balanced':
            # Balance between high prediction and moderate uncertainty
            score = predictions * (1 + 0.5 * uncertainties)
            indices = np.argsort(score)[-n_samples:]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Selected {n_samples} samples using {strategy} strategy")
        return indices
    
    @staticmethod
    def comprehensive_uncertainty_report(
        predictions: np.ndarray,
        aleatoric: np.ndarray,
        epistemic: np.ndarray,
        targets: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive uncertainty analysis report.
        
        Args:
            predictions: Model predictions
            aleatoric: Aleatoric uncertainty
            epistemic: Epistemic uncertainty
            targets: Optional ground truth for calibration
            
        Returns:
            Comprehensive report dictionary
        """
        total = np.sqrt(aleatoric ** 2 + epistemic ** 2)
        
        report = {
            'summary': {
                'n_samples': len(predictions),
                'mean_prediction': float(predictions.mean()),
                'mean_total_uncertainty': float(total.mean()),
                'mean_aleatoric': float(aleatoric.mean()),
                'mean_epistemic': float(epistemic.mean())
            },
            'aleatoric_stats': AleatoricUncertainty.analyze_uncertainty_distribution(aleatoric),
            'epistemic_stats': EpistemicUncertainty.analyze_uncertainty_distribution(epistemic),
            'total_stats': {
                'mean': float(total.mean()),
                'median': float(np.median(total)),
                'std': float(total.std()),
                'min': float(total.min()),
                'max': float(total.max())
            },
            'uncertainty_fractions': {
                'mean_aleatoric_fraction': float((aleatoric ** 2 / (total ** 2 + 1e-10)).mean()),
                'mean_epistemic_fraction': float((epistemic ** 2 / (total ** 2 + 1e-10)).mean())
            }
        }
        
        if targets is not None:
            calibration = TotalUncertainty.uncertainty_calibration_analysis(
                predictions, total, targets
            )
            report['calibration'] = {
                'calibration_error': float(calibration['calibration_error']),
                'mean_uncertainty_per_bin': calibration['mean_uncertainty_per_bin'].tolist(),
                'mean_error_per_bin': calibration['mean_error_per_bin'].tolist()
            }
        
        return report
