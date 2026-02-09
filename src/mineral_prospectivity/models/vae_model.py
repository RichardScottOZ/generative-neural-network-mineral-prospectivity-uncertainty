"""
Variational Autoencoder (VAE) model for mineral prospectivity prediction.

This model implements a VAE architecture that can:
1. Learn latent representations of geospatial features
2. Generate prospectivity maps
3. Estimate aleatoric uncertainty through the probabilistic decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class Encoder(nn.Module):
    """Encoder network for VAE that maps input features to latent space."""
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE that maps latent space to prospectivity predictions."""
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        # Output both mean and log variance for aleatoric uncertainty
        self.fc_mean = nn.Linear(prev_dim, output_dim)
        self.fc_logvar = nn.Linear(prev_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(z)
        mean = torch.sigmoid(self.fc_mean(h))  # Prospectivity probability
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)  # Aleatoric uncertainty
        return mean, logvar


class VAEProspectivityModel(nn.Module):
    """
    Variational Autoencoder for mineral prospectivity prediction with uncertainty.
    
    This model learns to:
    - Encode geospatial features into a latent representation
    - Decode latent variables to prospectivity predictions
    - Estimate aleatoric uncertainty in predictions
    
    Args:
        input_dim: Number of input features (geophysical, geochemical, etc.)
        latent_dim: Dimension of latent space
        encoder_hidden_dims: List of hidden layer dimensions for encoder
        decoder_hidden_dims: List of hidden layer dimensions for decoder
        output_dim: Output dimension (typically 1 for binary prospectivity)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        encoder_hidden_dims: Optional[list] = None,
        decoder_hidden_dims: Optional[list] = None,
        output_dim: int = 1
    ):
        super().__init__()
        
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 128, 64]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 128, 256]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        
        self.encoder = Encoder(input_dim, encoder_hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_hidden_dims, output_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Returns:
            Dictionary containing:
                - prediction_mean: Mean prospectivity prediction
                - prediction_logvar: Log variance (aleatoric uncertainty)
                - latent_mu: Mean of latent distribution
                - latent_logvar: Log variance of latent distribution
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        pred_mean, pred_logvar = self.decoder(z)
        
        output = {
            'prediction_mean': pred_mean,
            'prediction_logvar': pred_logvar,
            'latent_mu': mu,
            'latent_logvar': logvar
        }
        
        if return_latent:
            output['latent_z'] = z
        
        return output
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates using Monte Carlo sampling.
        
        Args:
            x: Input features
            num_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            Dictionary with mean predictions and uncertainties
        """
        self.eval()
        
        with torch.no_grad():
            # Get latent distribution
            mu, logvar = self.encoder(x)
            
            predictions = []
            aleatoric_vars = []
            for _ in range(num_samples):
                # Sample from latent distribution
                z = self.reparameterize(mu, logvar)
                # Decode
                pred_mean, pred_logvar = self.decoder(z)
                predictions.append(pred_mean)
                aleatoric_vars.append(torch.exp(pred_logvar))
            
            predictions = torch.stack(predictions)
            
            # Compute statistics
            pred_mean = predictions.mean(dim=0)
            pred_std = predictions.std(dim=0)
            
            # Average aleatoric uncertainty across all MC samples
            aleatoric_var = torch.stack(aleatoric_vars).mean(dim=0)
            
            # Epistemic uncertainty (variance of predictions)
            epistemic_var = pred_std ** 2
            
            # Total uncertainty
            total_var = aleatoric_var + epistemic_var
        
        return {
            'mean': pred_mean,
            'std': torch.sqrt(total_var),
            'aleatoric_uncertainty': torch.sqrt(aleatoric_var),
            'epistemic_uncertainty': torch.sqrt(epistemic_var),
            'total_uncertainty': torch.sqrt(total_var)
        }
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss with KL divergence and reconstruction loss.
        
        Args:
            x: Input features
            y: Target labels
            beta: Weight for KL divergence term (beta-VAE)
            
        Returns:
            Dictionary with total loss and components
        """
        outputs = self.forward(x)
        
        pred_mean = outputs['prediction_mean']
        pred_logvar = outputs['prediction_logvar']
        latent_mu = outputs['latent_mu']
        latent_logvar = outputs['latent_logvar']
        
        # Reconstruction loss with aleatoric uncertainty
        # Negative log likelihood assuming Gaussian distribution
        reconstruction_loss = 0.5 * (
            pred_logvar + 
            (y - pred_mean) ** 2 / torch.exp(pred_logvar)
        ).mean()
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp()
        ) / x.size(0)
        
        # Total loss
        total_loss = reconstruction_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss
        }
    
    def save_model(self, path: str):
        """Save model state to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'output_dim': self.output_dim,
            'encoder_hidden_dims': self.encoder_hidden_dims,
            'decoder_hidden_dims': self.decoder_hidden_dims
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            output_dim=checkpoint['output_dim'],
            encoder_hidden_dims=checkpoint.get('encoder_hidden_dims', [256, 128, 64]),
            decoder_hidden_dims=checkpoint.get('decoder_hidden_dims', [64, 128, 256])
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
