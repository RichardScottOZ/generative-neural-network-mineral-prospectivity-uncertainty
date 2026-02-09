# Usage Guide

This guide demonstrates how to use the mineral prospectivity framework for training ensemble models with uncertainty quantification.

## Quick Start

### 1. Install the Package

```bash
pip install -e .
```

### 2. Prepare Your Data

Upload your data to S3 following the [S3 Data Requirements](S3_DATA_REQUIREMENTS.md).

### 3. Launch Ensemble Training

```python
from mineral_prospectivity.aws_batch.job_launcher import BatchJobLauncher
from mineral_prospectivity.utils.config import Config

# Create configuration
config = Config()
config.update({
    'input_dim': 50,
    'num_models': 10,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001
})

# Initialize job launcher
launcher = BatchJobLauncher(
    job_queue='mineral-prospectivity-queue',
    job_definition='mineral-prospectivity-training',
    s3_bucket='your-bucket',
    region_name='us-east-1'
)

# Launch ensemble training
job_ids = launcher.launch_ensemble_training(
    num_models=10,
    config=config.to_dict(),
    experiment_name='experiment-1'
)

print(f"Launched {len(job_ids)} training jobs")

# Monitor jobs
final_status = launcher.monitor_jobs(job_ids, poll_interval=30)
print(f"Final status: {final_status}")
```

### 4. Download and Evaluate Results

```python
# Download trained models
models_dir = './trained_models'
launcher.download_ensemble_results(
    experiment_name='experiment-1',
    local_directory=models_dir
)

# Load ensemble
from mineral_prospectivity.models.ensemble import EnsembleModel

ensemble = EnsembleModel.load_ensemble(models_dir)

# Make predictions with uncertainty
import torch
import numpy as np

# Load test data
test_data = np.load('test_features.npy')
test_data_tensor = torch.FloatTensor(test_data)

# Predict with uncertainty
results = ensemble.predict(test_data_tensor)

print(f"Mean prediction shape: {results['mean'].shape}")
print(f"Aleatoric uncertainty shape: {results['aleatoric_uncertainty'].shape}")
print(f"Epistemic uncertainty shape: {results['epistemic_uncertainty'].shape}")
print(f"Total uncertainty shape: {results['total_uncertainty'].shape}")
```

## Detailed Workflows

### Local Training (Single Model)

Train a single VAE model locally:

```python
from mineral_prospectivity.models.vae_model import VAEProspectivityModel
from mineral_prospectivity.data.s3_loader import S3DataLoader
import torch

# Load data
s3_loader = S3DataLoader(
    bucket_name='your-bucket',
    experiment_name='local-test'
)

train_loader = s3_loader.load_training_data(batch_size=32)
val_loader = s3_loader.load_validation_data(batch_size=32)

# Create model
model = VAEProspectivityModel(
    input_dim=50,
    latent_dim=32
)

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        loss_dict = model.compute_loss(batch_x, batch_y)
        loss = loss_dict['total_loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

# Save model
model.save_model('local_model.pt')
```

### Local Ensemble Training

Train an ensemble locally:

```python
from mineral_prospectivity.models.ensemble import EnsembleModel

# Create ensemble
ensemble = EnsembleModel(
    num_models=5,
    input_dim=50,
    latent_dim=32
)

# Train each model
for i in range(5):
    print(f"Training model {i+1}/5...")
    
    history = ensemble.train_model(
        model_idx=i,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        learning_rate=0.001,
        device=device
    )
    
    print(f"Model {i} best val loss: {min(history['val_losses']):.4f}")

# Save ensemble
ensemble.save_ensemble('./ensemble_models')
```

### Making Predictions

```python
import numpy as np
import torch

# Load test data
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')

test_data = torch.FloatTensor(test_features)

# Predict with ensemble
results = ensemble.predict(test_data, num_mc_samples=100)

# Extract predictions and uncertainties
predictions = results['mean'].cpu().numpy()
aleatoric = results['aleatoric_uncertainty'].cpu().numpy()
epistemic = results['epistemic_uncertainty'].cpu().numpy()
total = results['total_uncertainty'].cpu().numpy()

# Compute metrics
from mineral_prospectivity.metrics.performance_metrics import PerformanceMetrics
from mineral_prospectivity.metrics.uncertainty_metrics import UncertaintyMetrics

# Performance metrics
perf_metrics = PerformanceMetrics.compute_all_metrics(
    test_labels,
    predictions.flatten(),
    total.flatten()
)

print(f"ROC-AUC: {perf_metrics['roc_auc']:.4f}")
print(f"Average Precision: {perf_metrics['average_precision']:.4f}")

# Uncertainty metrics
unc_metrics = UncertaintyMetrics.compute_all_uncertainty_metrics(
    test_labels,
    predictions.flatten(),
    aleatoric.flatten(),
    epistemic.flatten(),
    total.flatten()
)

print(f"Calibration Error: {unc_metrics['total']['expected_calibration_error']:.4f}")
print(f"Uncertainty-Error Correlation: {unc_metrics['total']['uncertainty_error_correlation']:.4f}")
```

### Risk-Return Analysis

```python
from mineral_prospectivity.uncertainty.total_uncertainty import TotalUncertainty

# Perform risk-return analysis
risk_analysis = TotalUncertainty.uncertainty_risk_analysis(
    predictions=predictions.flatten(),
    uncertainties=total.flatten(),
    exploration_cost=1.0,
    discovery_value=100.0,
    confidence_threshold=0.7
)

# Get recommended exploration targets
recommended = risk_analysis['recommend_exploration']
print(f"Recommended exploration targets: {recommended.sum()}")

# Sort by risk-adjusted return
sorted_indices = np.argsort(risk_analysis['risk_adjusted_return'])[::-1]
top_targets = sorted_indices[:100]  # Top 100 targets

print(f"Top target indices: {top_targets}")
print(f"Expected values: {risk_analysis['expected_value'][top_targets]}")
print(f"Risks: {risk_analysis['risk'][top_targets]}")
```

### Uncertainty Decomposition

```python
from mineral_prospectivity.uncertainty.total_uncertainty import TotalUncertainty

# Generate comprehensive uncertainty report
report = TotalUncertainty.comprehensive_uncertainty_report(
    predictions=predictions.flatten(),
    aleatoric=aleatoric.flatten(),
    epistemic=epistemic.flatten(),
    targets=test_labels
)

print("\nUncertainty Report:")
print(f"Mean total uncertainty: {report['summary']['mean_total_uncertainty']:.4f}")
print(f"Mean aleatoric: {report['summary']['mean_aleatoric']:.4f}")
print(f"Mean epistemic: {report['summary']['mean_epistemic']:.4f}")

print(f"\nAleatoric fraction: {report['uncertainty_fractions']['mean_aleatoric_fraction']:.2%}")
print(f"Epistemic fraction: {report['uncertainty_fractions']['mean_epistemic_fraction']:.2%}")

if 'calibration' in report:
    print(f"\nCalibration error: {report['calibration']['calibration_error']:.4f}")
```

### Active Learning / Uncertainty Sampling

```python
# Select samples with highest uncertainty for further investigation
high_uncertainty_indices = TotalUncertainty.uncertainty_based_sampling(
    predictions=predictions.flatten(),
    uncertainties=total.flatten(),
    n_samples=50,
    strategy='highest_uncertainty'
)

print(f"Samples with highest uncertainty: {high_uncertainty_indices}")

# Or balance high prediction with moderate uncertainty
balanced_indices = TotalUncertainty.uncertainty_based_sampling(
    predictions=predictions.flatten(),
    uncertainties=total.flatten(),
    n_samples=50,
    strategy='balanced'
)

print(f"Balanced selection: {balanced_indices}")
```

## Configuration Management

### Create Custom Configuration

```python
from mineral_prospectivity.utils.config import Config

# Start with defaults
config = Config()

# Customize
config.update({
    # Model
    'input_dim': 75,
    'latent_dim': 64,
    'encoder_hidden_dims': [512, 256, 128],
    'decoder_hidden_dims': [128, 256, 512],
    
    # Training
    'batch_size': 64,
    'learning_rate': 0.0005,
    'epochs': 200,
    'beta': 1.5,
    
    # Ensemble
    'num_models': 20,
    'use_bootstrap': True,
    
    # AWS
    's3_bucket': 'my-custom-bucket'
})

# Validate
config.validate()

# Save
config.save_json('my_config.json')
config.save_yaml('my_config.yaml')
```

### Load Configuration

```python
# Load from JSON
config = Config.load_json('my_config.json')

# Or from YAML
config = Config.load_yaml('my_config.yaml')
```

## Monitoring and Logging

```python
from mineral_prospectivity.utils.logging import setup_logging

# Setup logging
logger = setup_logging(
    level=logging.INFO,
    log_file='experiment.log'
)

# Use throughout your code
logger.info("Starting experiment...")
logger.error("Something went wrong!")
```

## Infrastructure Management

### Check Infrastructure Status

```python
from mineral_prospectivity.aws_batch.infrastructure import check_infrastructure_status

status = check_infrastructure_status(region_name='us-east-1')

for component, state in status.items():
    print(f"{component}: {state}")
```

### Generate Infrastructure Templates

```python
from mineral_prospectivity.aws_batch.job_definitions import save_infrastructure_templates

# Save all templates to infrastructure/ directory
save_infrastructure_templates(output_dir='infrastructure')
```

## Tips and Best Practices

### 1. Start Small

Begin with a small ensemble (3-5 models) to verify everything works:

```python
# Test configuration
config.update({
    'num_models': 3,
    'epochs': 20
})
```

### 2. Monitor Costs

Use AWS Cost Explorer to track spending. Set up billing alerts.

### 3. Use Spot Instances

Configure compute environment to use Spot instances for 70-90% cost savings.

### 4. Preprocess Data Properly

Always preprocess and validate data before uploading to S3.

### 5. Version Your Experiments

Use meaningful experiment names with timestamps:

```python
from datetime import datetime

experiment_name = f"copper-exploration-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
```

### 6. Save Intermediate Results

Enable checkpointing during training to resume interrupted jobs.

### 7. Validate Uncertainty Calibration

Always check that uncertainty estimates are well-calibrated:

```python
from mineral_prospectivity.metrics.uncertainty_metrics import UncertaintyMetrics

ece = UncertaintyMetrics.expected_calibration_error(
    test_labels, predictions, uncertainties
)

if ece > 0.1:
    logger.warning(f"Poor calibration detected: ECE = {ece:.4f}")
```

## Troubleshooting

### Common Issues

1. **Job fails immediately**: Check CloudWatch logs for detailed error messages
2. **Out of memory**: Reduce batch size or model dimensions
3. **Data not found**: Verify S3 bucket permissions and data paths
4. **Slow training**: Use GPU instances (p3, g4dn families)
5. **Poor uncertainty quality**: Increase ensemble size or MC samples

### Debug Mode

Run with verbose logging:

```python
from mineral_prospectivity.utils.logging import setup_logging
import logging

setup_logging(level=logging.DEBUG)
```
