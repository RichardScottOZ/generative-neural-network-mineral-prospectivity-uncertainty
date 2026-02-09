"""
Configuration management for mineral prospectivity experiments.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for experiments.
    
    Handles loading, validation, and access to configuration parameters.
    """
    
    DEFAULT_CONFIG = {
        # Model architecture
        'input_dim': 50,
        'latent_dim': 32,
        'encoder_hidden_dims': [256, 128, 64],
        'decoder_hidden_dims': [64, 128, 256],
        'output_dim': 1,
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'beta': 1.0,  # Beta-VAE parameter
        'early_stopping_patience': 10,
        'base_seed': 42,
        
        # Ensemble parameters
        'num_models': 10,
        'use_bootstrap': True,
        'num_mc_samples': 100,
        
        # Data parameters
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        
        # AWS parameters
        'region_name': 'us-east-1',
        'job_queue': 'mineral-prospectivity-queue',
        'job_definition': 'mineral-prospectivity-training',
        's3_bucket': 'mineral-prospectivity-data',
        
        # Uncertainty parameters
        'confidence_level': 0.95,
        'uncertainty_threshold_percentile': 90,
        
        # Economic parameters
        'exploration_cost': 1.0,
        'discovery_value': 100.0,
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_dict:
            self.update(config_dict)
    
    def update(self, config_dict: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            config_dict: Dictionary with configuration updates
        """
        self.config.update(config_dict)
        logger.info(f"Updated configuration with {len(config_dict)} parameters")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict syntax."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set configuration value using dict syntax."""
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def save_json(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved configuration to {filepath}")
    
    def save_yaml(self, filepath: str):
        """
        Save configuration to YAML file.
        
        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {filepath}")
    
    @classmethod
    def load_json(cls, filepath: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Config instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"Loaded configuration from {filepath}")
        return cls(config_dict)
    
    @classmethod
    def load_yaml(cls, filepath: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Config instance
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {filepath}")
        return cls(config_dict)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check required parameters
        required = ['input_dim', 'num_models', 's3_bucket']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required configuration: {key}")
        
        # Validate ranges
        if self.config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.config['num_models'] <= 0:
            raise ValueError("num_models must be positive")
        
        # Validate splits sum to 1
        splits_sum = (
            self.config['train_split'] +
            self.config['val_split'] +
            self.config['test_split']
        )
        if abs(splits_sum - 1.0) > 0.01:
            raise ValueError("Data splits must sum to 1.0")
        
        logger.info("Configuration validated successfully")
        return True
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({json.dumps(self.config, indent=2)})"
