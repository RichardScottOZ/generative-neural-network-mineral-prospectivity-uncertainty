"""
Generative Neural Network Approach to Uncertainty and Risk-Return Analysis
in Mineral Prospectivity Modelling

This package implements a framework for mineral prospectivity modeling using
generative neural networks with comprehensive uncertainty quantification,
designed to run on AWS Batch infrastructure.
"""

__version__ = "0.1.0"

from . import models
from . import aws_batch
from . import data
from . import uncertainty
from . import metrics
from . import utils

__all__ = [
    "models",
    "aws_batch",
    "data",
    "uncertainty",
    "metrics",
    "utils",
]
