# Generative Neural Network for Mineral Prospectivity with Uncertainty Quantification

A comprehensive framework for mineral prospectivity modeling using Variational Autoencoders (VAE) with ensemble-based uncertainty quantification, designed to run on AWS Batch infrastructure.

This implementation is based on the paper: [A generative neural network approach to uncertainty and risk-return analysis in mineral prospectivity modelling](https://www.sciencedirect.com/science/article/pii/S0169136825004044)

## Features

### 🧠 Advanced Deep Learning
- **Variational Autoencoder (VAE)** architecture for mineral prospectivity prediction
- Learns latent representations of complex geospatial features
- Handles high-dimensional geophysical, geochemical, and geological data

### 📊 Comprehensive Uncertainty Quantification
- **Aleatoric Uncertainty**: Data-inherent noise and measurement errors
- **Epistemic Uncertainty**: Model uncertainty from limited training data
- **Total Uncertainty**: Combined uncertainty for robust decision-making
- Multiple uncertainty metrics (calibration, sharpness, correlation)

### 🔄 Ensemble Learning
- Train multiple models with different initializations
- Bootstrap sampling for diverse ensemble members
- Model disagreement quantifies epistemic uncertainty
- Improved robustness and reliability

### ☁️ AWS Batch Infrastructure
- Distributed training across multiple GPU instances
- Automatic job scheduling and monitoring
- Scalable to large ensembles (10-100+ models)
- Cost-effective with Spot instance support

### 📈 Risk-Return Analysis
- Economic value calculations for exploration decisions
- Uncertainty-based target prioritization
- Capture efficiency metrics
- Active learning for optimal data collection

### 🎯 Production-Ready
- S3 integration for data and results
- Comprehensive logging and monitoring
- Infrastructure-as-code with CloudFormation
- Easy deployment and scaling

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS Batch Infrastructure                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Training Job │  │ Training Job │  │ Training Job │      │
│  │   Model 0    │  │   Model 1    │  │   Model N    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │   S3 Storage    │
                    ├─────────────────┤
                    │ Training Data   │
                    │ Validation Data │
                    │ Test Data       │
                    │ Trained Models  │
                    │ Results/Metrics │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Ensemble Model │
                    ├─────────────────┤
                    │ VAE Model 0     │
                    │ VAE Model 1     │
                    │ ...             │
                    │ VAE Model N     │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Predictions    │
                    ├─────────────────┤
                    │ Mean            │
                    │ Aleatoric Unc.  │
                    │ Epistemic Unc.  │
                    │ Total Unc.      │
                    └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- AWS account with appropriate permissions
- Docker (for building container images)
- AWS CLI configured

### Install Package

```bash
# Clone repository
git clone https://github.com/RichardScottOZ/generative-neural-network-mineral-prospectivity-uncertainty.git
cd generative-neural-network-mineral-prospectivity-uncertainty

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Optional Dependencies

```bash
# For development
pip install -e ".[dev]"

# For visualization
pip install -e ".[viz]"
```

## Quick Start

### 1. Setup AWS Infrastructure

See [AWS Setup Guide](docs/AWS_SETUP.md) for detailed instructions.

```python
from mineral_prospectivity.aws_batch.infrastructure import setup_batch_infrastructure

resources = setup_batch_infrastructure(
    region_name='us-east-1',
    vpc_id='vpc-xxxxx',
    subnet_ids=['subnet-xxxxx', 'subnet-yyyyy'],
    s3_bucket='mineral-prospectivity-data'
)
```

### 2. Prepare and Upload Data

See [S3 Data Requirements](docs/S3_DATA_REQUIREMENTS.md) for data format details.

```python
from mineral_prospectivity.data.s3_loader import S3DataLoader
import numpy as np

# Prepare your data
features = np.load('features.npy')  # Shape: (n_samples, n_features)
labels = np.load('labels.npy')      # Shape: (n_samples,)

# Upload to S3
s3_loader = S3DataLoader(bucket_name='mineral-prospectivity-data', experiment_name='exp-1')
s3_loader.upload_file('features.npy', 'data/training/features.npy')
s3_loader.upload_file('labels.npy', 'data/training/labels.npy')
```

### 3. Launch Ensemble Training

```python
from mineral_prospectivity.aws_batch.job_launcher import BatchJobLauncher
from mineral_prospectivity.utils.config import Config

# Configure experiment
config = Config()
config.update({
    'input_dim': 50,
    'num_models': 10,
    'epochs': 100,
    'batch_size': 32
})

# Launch training jobs
launcher = BatchJobLauncher(
    job_queue='mineral-prospectivity-queue',
    job_definition='mineral-prospectivity-training',
    s3_bucket='mineral-prospectivity-data'
)

job_ids = launcher.launch_ensemble_training(
    num_models=10,
    config=config.to_dict(),
    experiment_name='exp-1'
)

# Monitor progress
status = launcher.monitor_jobs(job_ids)
```

### 4. Make Predictions with Uncertainty

```python
from mineral_prospectivity.models.ensemble import EnsembleModel
import torch

# Load trained ensemble
ensemble = EnsembleModel.load_ensemble('./models/exp-1')

# Predict on new data
test_data = torch.FloatTensor(test_features)
results = ensemble.predict(test_data)

# Access predictions and uncertainties
predictions = results['mean']
aleatoric_uncertainty = results['aleatoric_uncertainty']
epistemic_uncertainty = results['epistemic_uncertainty']
total_uncertainty = results['total_uncertainty']
```

## Documentation

- [AWS Setup Guide](docs/AWS_SETUP.md) - Complete AWS infrastructure setup
- [S3 Data Requirements](docs/S3_DATA_REQUIREMENTS.md) - Data format and structure
- [Usage Guide](docs/USAGE.md) - Detailed usage examples and workflows

## Key Concepts

### Uncertainty Types

1. **Aleatoric Uncertainty** (Data Uncertainty)
   - Represents inherent noise in measurements
   - Cannot be reduced by collecting more data
   - Estimated through VAE's probabilistic decoder
   - Important for understanding data quality limitations

2. **Epistemic Uncertainty** (Model Uncertainty)
   - Represents uncertainty in model parameters
   - Can be reduced with more training data
   - Estimated through ensemble disagreement
   - Indicates where the model needs more information

3. **Total Uncertainty**
   - Combined measure: `Total² = Aleatoric² + Epistemic²`
   - Used for robust decision-making
   - Guides active learning and exploration

### Metrics

**Performance Metrics:**
- ROC-AUC and Precision-Recall curves
- Capture efficiency at various percentiles
- Confusion matrix metrics
- Economic value analysis

**Uncertainty Metrics:**
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Negative Log Likelihood (NLL)
- Coverage and sharpness
- Uncertainty-error correlation

## Project Structure

```
.
├── src/
│   └── mineral_prospectivity/
│       ├── models/              # Neural network models
│       │   ├── vae_model.py    # VAE architecture
│       │   └── ensemble.py     # Ensemble management
│       ├── aws_batch/           # AWS Batch integration
│       │   ├── job_launcher.py # Job submission
│       │   ├── job_definitions.py  # Infrastructure templates
│       │   ├── infrastructure.py   # Setup utilities
│       │   └── train_worker.py     # Training worker script
│       ├── data/                # Data handling
│       │   ├── s3_loader.py    # S3 data loading
│       │   └── preprocessor.py # Data preprocessing
│       ├── uncertainty/         # Uncertainty quantification
│       │   ├── aleatoric.py    # Aleatoric uncertainty
│       │   ├── epistemic.py    # Epistemic uncertainty
│       │   └── total_uncertainty.py  # Combined analysis
│       ├── metrics/             # Evaluation metrics
│       │   ├── performance_metrics.py
│       │   └── uncertainty_metrics.py
│       └── utils/               # Utilities
│           ├── config.py       # Configuration management
│           └── logging.py      # Logging setup
├── docs/                        # Documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Example Use Case

**Copper Exploration in Australia**

```python
# 1. Prepare geospatial features
features = extract_features([
    'magnetic_intensity',
    'gravity_anomaly', 
    'cu_concentration',
    'au_concentration',
    'distance_to_fault',
    'rock_type_encoding'
])  # Shape: (100000, 50)

# 2. Train ensemble
job_ids = launcher.launch_ensemble_training(
    num_models=20,
    config=config.to_dict(),
    experiment_name='copper-exploration-2024'
)

# 3. Generate prospectivity map with uncertainty
ensemble = EnsembleModel.load_ensemble('./trained_models')
results = ensemble.predict(grid_features)

prospectivity_map = results['mean'].reshape(grid_shape)
uncertainty_map = results['total_uncertainty'].reshape(grid_shape)

# 4. Identify high-priority targets
from mineral_prospectivity.uncertainty.total_uncertainty import TotalUncertainty

risk_analysis = TotalUncertainty.uncertainty_risk_analysis(
    predictions=results['mean'].numpy().flatten(),
    uncertainties=results['total_uncertainty'].numpy().flatten(),
    exploration_cost=100_000,  # $100k per drill hole
    discovery_value=50_000_000,  # $50M for discovery
    confidence_threshold=0.8
)

top_targets = np.argsort(risk_analysis['risk_adjusted_return'])[-100:]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mineral_prospectivity_2025,
  title={A generative neural network approach to uncertainty and risk-return analysis in mineral prospectivity modelling},
  journal={Ore Geology Reviews},
  year={2025},
  doi={10.1016/j.oregeorev.2025.004044}
}
```

## Acknowledgments

- Based on the research paper on generative neural networks for mineral prospectivity
- Built with PyTorch, AWS Batch, and scikit-learn
- Inspired by advances in uncertainty quantification and Bayesian deep learning

## Support

For questions and issues:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the examples in `docs/USAGE.md`
