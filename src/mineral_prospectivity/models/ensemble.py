"""
Ensemble model for combining multiple VAE models to estimate epistemic uncertainty.

The ensemble approach captures model uncertainty by training multiple models
with different initializations and/or on different data subsets (bootstrap).
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import logging

from .vae_model import VAEProspectivityModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble of VAE models for robust prospectivity prediction with epistemic uncertainty.
    
    This class manages multiple VAE models and provides methods to:
    - Train ensemble members independently
    - Aggregate predictions from all models
    - Estimate epistemic uncertainty from model disagreement
    - Combine with aleatoric uncertainty for total uncertainty
    
    Args:
        num_models: Number of models in the ensemble
        input_dim: Input feature dimension
        latent_dim: Latent space dimension
        model_kwargs: Additional arguments for VAE model initialization
    """
    
    def __init__(
        self,
        num_models: int,
        input_dim: int,
        latent_dim: int = 32,
        **model_kwargs
    ):
        self.num_models = num_models
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model_kwargs = model_kwargs
        
        # Initialize ensemble members
        self.models: List[VAEProspectivityModel] = []
        for i in range(num_models):
            model = VAEProspectivityModel(
                input_dim=input_dim,
                latent_dim=latent_dim,
                **model_kwargs
            )
            self.models.append(model)
        
        logger.info(f"Initialized ensemble with {num_models} models")
    
    def train_model(
        self,
        model_idx: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        beta: float = 1.0,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train a single model in the ensemble.
        
        Args:
            model_idx: Index of model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to train on
            beta: Beta parameter for beta-VAE
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with training history
        """
        model = self.models[model_idx].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                loss_dict = model.compute_loss(batch_x, batch_y, beta=beta)
                loss = loss_dict['total_loss']
                
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            if val_loader is not None:
                model.eval()
                epoch_val_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        
                        loss_dict = model.compute_loss(batch_x, batch_y, beta=beta)
                        epoch_val_loss += loss_dict['total_loss'].item()
                
                epoch_val_loss /= len(val_loader)
                val_losses.append(epoch_val_loss)
                
                # Early stopping
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Model {model_idx}: Early stopping at epoch {epoch}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Model {model_idx}, Epoch {epoch+1}/{epochs}: "
                        f"Train Loss = {epoch_train_loss:.4f}, "
                        f"Val Loss = {epoch_val_loss:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Model {model_idx}, Epoch {epoch+1}/{epochs}: "
                        f"Train Loss = {epoch_train_loss:.4f}"
                    )
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def predict(
        self,
        x: torch.Tensor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_mc_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the full ensemble with uncertainty quantification.
        
        Args:
            x: Input features
            device: Device for computation
            num_mc_samples: Number of Monte Carlo samples per model
            
        Returns:
            Dictionary containing:
                - mean: Ensemble mean prediction
                - aleatoric_uncertainty: Mean aleatoric uncertainty across models
                - epistemic_uncertainty: Model disagreement (epistemic)
                - total_uncertainty: Combined uncertainty
                - individual_predictions: Predictions from each model
        """
        x = x.to(device)
        
        all_predictions = []
        all_aleatoric = []
        
        for model in self.models:
            model.to(device)
            model.eval()
            
            # Get predictions with uncertainty from each model
            pred_dict = model.predict_with_uncertainty(x, num_samples=num_mc_samples)
            
            all_predictions.append(pred_dict['mean'])
            all_aleatoric.append(pred_dict['aleatoric_uncertainty'] ** 2)  # Variance
        
        # Stack predictions
        predictions = torch.stack(all_predictions)  # [num_models, batch_size, output_dim]
        aleatoric_vars = torch.stack(all_aleatoric)
        
        # Ensemble mean
        ensemble_mean = predictions.mean(dim=0)
        
        # Mean aleatoric uncertainty
        mean_aleatoric_var = aleatoric_vars.mean(dim=0)
        
        # Epistemic uncertainty (variance across models)
        epistemic_var = predictions.var(dim=0)
        
        # Total uncertainty
        total_var = mean_aleatoric_var + epistemic_var
        
        return {
            'mean': ensemble_mean,
            'aleatoric_uncertainty': torch.sqrt(mean_aleatoric_var),
            'epistemic_uncertainty': torch.sqrt(epistemic_var),
            'total_uncertainty': torch.sqrt(total_var),
            'individual_predictions': predictions
        }
    
    def save_ensemble(self, directory: str):
        """
        Save all models in the ensemble.
        
        Args:
            directory: Directory to save models
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = save_dir / f"model_{i}.pt"
            model.save_model(str(model_path))
        
        # Save ensemble metadata
        metadata = {
            'num_models': self.num_models,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'model_kwargs': self.model_kwargs
        }
        torch.save(metadata, save_dir / "ensemble_metadata.pt")
        
        logger.info(f"Saved ensemble to {directory}")
    
    @classmethod
    def load_ensemble(cls, directory: str, device: str = 'cpu'):
        """
        Load ensemble from saved models.
        
        Args:
            directory: Directory containing saved models
            device: Device to load models on
            
        Returns:
            EnsembleModel instance
        """
        load_dir = Path(directory)
        
        # Load metadata
        metadata = torch.load(load_dir / "ensemble_metadata.pt", map_location=device)
        
        # Create ensemble
        ensemble = cls(
            num_models=metadata['num_models'],
            input_dim=metadata['input_dim'],
            latent_dim=metadata['latent_dim'],
            **metadata['model_kwargs']
        )
        
        # Load individual models
        for i in range(metadata['num_models']):
            model_path = load_dir / f"model_{i}.pt"
            ensemble.models[i] = VAEProspectivityModel.load_model(
                str(model_path), 
                device=device
            )
        
        logger.info(f"Loaded ensemble from {directory}")
        return ensemble
