"""
Data preprocessing for mineral prospectivity features.

This module handles normalization, feature engineering, and data quality checks.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for mineral prospectivity geospatial features.
    
    Handles:
    - Missing value imputation
    - Feature scaling/normalization
    - Outlier handling
    - Feature engineering
    
    Args:
        scaler_type: Type of scaler ('standard' or 'robust')
        imputation_strategy: Strategy for missing values ('mean', 'median', 'most_frequent')
        handle_outliers: Whether to clip outliers
    """
    
    def __init__(
        self,
        scaler_type: str = 'robust',
        imputation_strategy: str = 'median',
        handle_outliers: bool = True
    ):
        self.scaler_type = scaler_type
        self.imputation_strategy = imputation_strategy
        self.handle_outliers = handle_outliers
        
        # Initialize components
        self.imputer = SimpleImputer(strategy=imputation_strategy)
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
        self.feature_stats = {}
        
        logger.info(
            f"Initialized DataPreprocessor with scaler={scaler_type}, "
            f"imputation={imputation_strategy}"
        )
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Optional labels (not used, for sklearn compatibility)
            
        Returns:
            Self
        """
        logger.info("Fitting preprocessor...")
        
        # Compute statistics before preprocessing
        self.feature_stats['raw_mean'] = np.nanmean(X, axis=0)
        self.feature_stats['raw_std'] = np.nanstd(X, axis=0)
        self.feature_stats['missing_fraction'] = np.isnan(X).mean(axis=0)
        
        # Fit imputer
        X_imputed = self.imputer.fit_transform(X)
        
        # Handle outliers if requested
        if self.handle_outliers:
            # Compute percentiles for clipping
            q1 = np.percentile(X_imputed, 1, axis=0)
            q99 = np.percentile(X_imputed, 99, axis=0)
            self.feature_stats['clip_min'] = q1
            self.feature_stats['clip_max'] = q99
            
            X_clipped = np.clip(X_imputed, q1, q99)
        else:
            X_clipped = X_imputed
        
        # Fit scaler
        self.scaler.fit(X_clipped)
        
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Impute missing values
        X_imputed = self.imputer.transform(X)
        
        # Handle outliers if requested
        if self.handle_outliers:
            X_clipped = np.clip(
                X_imputed,
                self.feature_stats['clip_min'],
                self.feature_stats['clip_max']
            )
        else:
            X_clipped = X_imputed
        
        # Scale features
        X_scaled = self.scaler.transform(X_clipped)
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Optional labels
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            X: Scaled features
            
        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        return self.scaler.inverse_transform(X)
    
    def get_feature_importance_weights(self) -> np.ndarray:
        """
        Get feature importance weights based on variance.
        
        Returns:
            Array of feature weights
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        # Use inverse of standard deviation as importance
        # Features with more variance are more informative
        if hasattr(self.scaler, 'scale_'):
            weights = 1.0 / (self.scaler.scale_ + 1e-8)
            weights = weights / weights.sum()
            return weights
        else:
            return np.ones(len(self.feature_stats['raw_mean']))
    
    def check_data_quality(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Check data quality and return report.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with data quality metrics
        """
        report = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'missing_values': {
                'total': np.isnan(X).sum(),
                'per_feature': np.isnan(X).sum(axis=0).tolist(),
                'fraction': np.isnan(X).mean()
            },
            'infinite_values': {
                'total': np.isinf(X).sum(),
                'fraction': np.isinf(X).mean()
            },
            'constant_features': [],
            'highly_correlated_pairs': []
        }
        
        # Check for constant features
        for i in range(X.shape[1]):
            if np.nanstd(X[:, i]) < 1e-8:
                report['constant_features'].append(i)
        
        logger.info(f"Data quality report: {report}")
        return report
    
    def save(self, filepath: str):
        """Save preprocessor state."""
        import pickle
        
        state = {
            'scaler_type': self.scaler_type,
            'imputation_strategy': self.imputation_strategy,
            'handle_outliers': self.handle_outliers,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_stats': self.feature_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved preprocessor to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load preprocessor from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            scaler_type=state['scaler_type'],
            imputation_strategy=state['imputation_strategy'],
            handle_outliers=state['handle_outliers']
        )
        
        preprocessor.imputer = state['imputer']
        preprocessor.scaler = state['scaler']
        preprocessor.is_fitted = state['is_fitted']
        preprocessor.feature_stats = state['feature_stats']
        
        logger.info(f"Loaded preprocessor from {filepath}")
        return preprocessor
