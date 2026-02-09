"""
S3 data loader for mineral prospectivity data.

This module handles loading training/validation/test data from S3,
including geospatial features, labels, and metadata.
"""

import boto3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class MineralProspectivityDataset(Dataset):
    """
    PyTorch Dataset for mineral prospectivity data.
    
    Args:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,) or (n_samples, 1)
        transform: Optional transform to apply to features
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform=None
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).reshape(-1, 1)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class S3DataLoader:
    """
    Handles loading and preprocessing data from S3 for mineral prospectivity modeling.
    
    Expected S3 structure:
    s3://bucket/
        data/
            training/
                features.npy
                labels.npy
            validation/
                features.npy
                labels.npy
            test/
                features.npy
                labels.npy
            features_metadata.json
        experiments/
            {experiment_name}/
                config_{timestamp}.json
                models/{timestamp}/
                    model_0.pt
                    model_1.pt
                    ...
                results/{timestamp}/
                    history_0.json
                    ...
    
    Args:
        bucket_name: Name of S3 bucket
        experiment_name: Name of experiment
        region_name: AWS region
    """
    
    def __init__(
        self,
        bucket_name: str,
        experiment_name: str,
        region_name: str = 'us-east-1'
    ):
        self.bucket_name = bucket_name
        self.experiment_name = experiment_name
        self.region_name = region_name
        
        self.s3_client = boto3.client('s3', region_name=region_name)
        
        # Cache for loaded data
        self._cache = {}
        
        logger.info(
            f"Initialized S3DataLoader with bucket={bucket_name}, "
            f"experiment={experiment_name}"
        )
    
    def _download_file(self, s3_key: str, local_path: Optional[str] = None) -> str:
        """
        Download file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local path to save file (temp file if None)
            
        Returns:
            Local file path
        """
        if local_path is None:
            # Create temporary file
            fd, local_path = tempfile.mkstemp()
            import os
            os.close(fd)
        
        logger.debug(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")
        
        try:
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_path
            )
            return local_path
        
        except Exception as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            raise
    
    def upload_file(self, local_path: str, s3_key: str):
        """
        Upload file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key
        """
        logger.debug(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
        
        try:
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key
            )
        except Exception as e:
            logger.error(f"Error uploading {s3_key}: {e}")
            raise
    
    def load_config(self, config_key: str) -> Dict[str, Any]:
        """
        Load configuration from S3.
        
        Args:
            config_key: S3 key for config file
            
        Returns:
            Configuration dictionary
        """
        local_path = self._download_file(config_key)
        
        with open(local_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def load_features_metadata(self) -> Dict[str, Any]:
        """
        Load features metadata from S3.
        
        Returns:
            Metadata dictionary with feature names, types, etc.
        """
        if 'metadata' in self._cache:
            return self._cache['metadata']
        
        metadata_key = 'data/features_metadata.json'
        local_path = self._download_file(metadata_key)
        
        with open(local_path, 'r') as f:
            metadata = json.load(f)
        
        self._cache['metadata'] = metadata
        return metadata
    
    def _load_numpy_data(
        self,
        data_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and labels from S3.
        
        Args:
            data_type: One of 'training', 'validation', 'test'
            
        Returns:
            Tuple of (features, labels)
        """
        cache_key = f'{data_type}_data'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Download features
        features_key = f'data/{data_type}/features.npy'
        features_path = self._download_file(features_key)
        features = np.load(features_path)
        
        # Download labels
        labels_key = f'data/{data_type}/labels.npy'
        labels_path = self._download_file(labels_key)
        labels = np.load(labels_path)
        
        logger.info(
            f"Loaded {data_type} data: "
            f"features shape={features.shape}, labels shape={labels.shape}"
        )
        
        self._cache[cache_key] = (features, labels)
        return features, labels
    
    def load_training_data(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        bootstrap: bool = False,
        random_seed: Optional[int] = None
    ) -> DataLoader:
        """
        Load training data and create DataLoader.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            bootstrap: Whether to use bootstrap sampling
            random_seed: Random seed for reproducibility
            
        Returns:
            PyTorch DataLoader
        """
        features, labels = self._load_numpy_data('training')
        
        # Bootstrap sampling if requested
        if bootstrap:
            if random_seed is not None:
                np.random.seed(random_seed)
            
            n_samples = len(features)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            features = features[indices]
            labels = labels[indices]
            
            logger.info("Applied bootstrap sampling")
        
        dataset = MineralProspectivityDataset(features, labels)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Use 0 for compatibility
            pin_memory=torch.cuda.is_available()
        )
        
        return dataloader
    
    def load_validation_data(
        self,
        batch_size: int = 32
    ) -> DataLoader:
        """
        Load validation data and create DataLoader.
        
        Args:
            batch_size: Batch size
            
        Returns:
            PyTorch DataLoader
        """
        features, labels = self._load_numpy_data('validation')
        
        dataset = MineralProspectivityDataset(features, labels)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        return dataloader
    
    def load_test_data(
        self,
        batch_size: int = 32
    ) -> DataLoader:
        """
        Load test data and create DataLoader.
        
        Args:
            batch_size: Batch size
            
        Returns:
            PyTorch DataLoader
        """
        features, labels = self._load_numpy_data('test')
        
        dataset = MineralProspectivityDataset(features, labels)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        return dataloader
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        s3_key: str
    ):
        """
        Save predictions and uncertainties to S3.
        
        Args:
            predictions: Prediction array
            uncertainties: Uncertainty array
            s3_key: S3 key to save to
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.npz', delete=False) as f:
            temp_path = f.name
        
        np.savez(
            temp_path,
            predictions=predictions,
            uncertainties=uncertainties
        )
        
        self.upload_file(temp_path, s3_key)
        logger.info(f"Saved predictions to s3://{self.bucket_name}/{s3_key}")
        
        # Clean up
        import os
        os.remove(temp_path)
