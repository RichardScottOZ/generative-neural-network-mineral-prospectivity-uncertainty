"""
Neural network models for mineral prospectivity prediction.
"""

from .vae_model import VAEProspectivityModel
from .ensemble import EnsembleModel

__all__ = ["VAEProspectivityModel", "EnsembleModel"]
