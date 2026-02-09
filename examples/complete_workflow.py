"""
Example script demonstrating the complete workflow for mineral prospectivity modeling
with uncertainty quantification using AWS Batch.
"""

import numpy as np
from pathlib import Path
import logging

from mineral_prospectivity.utils.logging import setup_logging
from mineral_prospectivity.utils.config import Config
from mineral_prospectivity.data.s3_loader import S3DataLoader
from mineral_prospectivity.data.preprocessor import DataPreprocessor
from mineral_prospectivity.aws_batch.job_launcher import BatchJobLauncher
from mineral_prospectivity.models.ensemble import EnsembleModel
from mineral_prospectivity.metrics.performance_metrics import PerformanceMetrics
from mineral_prospectivity.metrics.uncertainty_metrics import UncertaintyMetrics
from mineral_prospectivity.uncertainty.total_uncertainty import TotalUncertainty

# Setup logging
logger = setup_logging(level=logging.INFO)


def prepare_synthetic_data(n_samples=10000, n_features=50):
    """
    Generate synthetic mineral prospectivity data for demonstration.
    
    In practice, this would load real geospatial data.
    """
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features")
    
    # Generate features
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate labels (5% positive class - typical for mineral deposits)
    # Create some structure: deposits more likely in certain feature ranges
    deposit_likelihood = 1 / (1 + np.exp(-features[:, 0] - 0.5 * features[:, 1]))
    labels = np.random.binomial(1, deposit_likelihood * 0.1).astype(np.float32)
    
    logger.info(f"Positive class fraction: {labels.mean():.2%}")
    
    return features, labels


def split_and_preprocess_data(features, labels):
    """Split and preprocess the data."""
    from sklearn.model_selection import train_test_split
    
    logger.info("Splitting data into train/val/test")
    
    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=0.15, stratify=labels, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Preprocess
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor(
        scaler_type='robust',
        imputation_strategy='median',
        handle_outliers=True
    )
    
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessor


def upload_data_to_s3(train_data, val_data, test_data, s3_bucket, experiment_name):
    """Upload training data to S3."""
    logger.info("Uploading data to S3")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Save locally first
    data_dir = Path('./data_temp')
    data_dir.mkdir(exist_ok=True)
    
    for split, (X, y) in [('training', train_data), ('validation', val_data), ('test', test_data)]:
        np.save(data_dir / f'{split}_features.npy', X)
        np.save(data_dir / f'{split}_labels.npy', y)
    
    # Upload to S3
    s3_loader = S3DataLoader(
        bucket_name=s3_bucket,
        experiment_name=experiment_name
    )
    
    for split in ['training', 'validation', 'test']:
        s3_loader.upload_file(
            str(data_dir / f'{split}_features.npy'),
            f'data/{split}/features.npy'
        )
        s3_loader.upload_file(
            str(data_dir / f'{split}_labels.npy'),
            f'data/{split}/labels.npy'
        )
    
    logger.info("Data uploaded successfully")


def main():
    """Main workflow for mineral prospectivity modeling."""
    
    logger.info("=" * 60)
    logger.info("Mineral Prospectivity Modeling with Uncertainty")
    logger.info("=" * 60)
    
    # Configuration
    s3_bucket = 'mineral-prospectivity-demo'  # Change to your bucket
    experiment_name = 'demo-experiment'
    
    # Step 1: Prepare data
    logger.info("\nStep 1: Data Preparation")
    features, labels = prepare_synthetic_data(n_samples=10000, n_features=50)
    
    train_data, val_data, test_data, preprocessor = split_and_preprocess_data(
        features, labels
    )
    
    # Step 2: Upload to S3 (optional - comment out if running locally)
    # upload_data_to_s3(train_data, val_data, test_data, s3_bucket, experiment_name)
    
    # Step 3: Configure experiment
    logger.info("\nStep 2: Configuration")
    config = Config()
    config.update({
        'input_dim': 50,
        'latent_dim': 32,
        'num_models': 5,  # Small ensemble for demo
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        's3_bucket': s3_bucket
    })
    
    config.validate()
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Step 4: Train ensemble (locally for demo)
    logger.info("\nStep 3: Training Ensemble")
    logger.info("Training locally (use AWS Batch for production)")
    
    # Create data loaders
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data[0]),
        torch.FloatTensor(train_data[1]).reshape(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_data[0]),
        torch.FloatTensor(val_data[1]).reshape(-1, 1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create and train ensemble
    ensemble = EnsembleModel(
        num_models=config['num_models'],
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim']
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    for i in range(config['num_models']):
        logger.info(f"\nTraining model {i+1}/{config['num_models']}")
        history = ensemble.train_model(
            model_idx=i,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            device=device
        )
    
    # Save ensemble
    ensemble_dir = Path('./trained_ensemble')
    ensemble.save_ensemble(str(ensemble_dir))
    logger.info(f"Ensemble saved to {ensemble_dir}")
    
    # Step 5: Make predictions with uncertainty
    logger.info("\nStep 4: Predictions and Uncertainty Quantification")
    
    X_test, y_test = test_data
    test_tensor = torch.FloatTensor(X_test)
    
    results = ensemble.predict(test_tensor, device=device, num_mc_samples=100)
    
    predictions = results['mean'].cpu().numpy().flatten()
    aleatoric = results['aleatoric_uncertainty'].cpu().numpy().flatten()
    epistemic = results['epistemic_uncertainty'].cpu().numpy().flatten()
    total = results['total_uncertainty'].cpu().numpy().flatten()
    
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Mean prediction: {predictions.mean():.4f}")
    logger.info(f"Mean total uncertainty: {total.mean():.4f}")
    
    # Step 6: Evaluate performance
    logger.info("\nStep 5: Performance Evaluation")
    
    perf_metrics = PerformanceMetrics.compute_all_metrics(
        y_test, predictions, total
    )
    
    logger.info(f"ROC-AUC: {perf_metrics['roc_auc']:.4f}")
    logger.info(f"Average Precision: {perf_metrics['average_precision']:.4f}")
    logger.info(f"Best F1: {perf_metrics['best_f1']:.4f}")
    
    # Step 7: Uncertainty analysis
    logger.info("\nStep 6: Uncertainty Analysis")
    
    unc_metrics = UncertaintyMetrics.compute_all_uncertainty_metrics(
        y_test, predictions, aleatoric, epistemic, total
    )
    
    logger.info(f"Expected Calibration Error: {unc_metrics['total']['expected_calibration_error']:.4f}")
    logger.info(f"Uncertainty-Error Correlation: {unc_metrics['total']['uncertainty_error_correlation']:.4f}")
    logger.info(f"Coverage (95%): {unc_metrics['total']['coverage_95']:.2%}")
    
    # Step 8: Risk-return analysis
    logger.info("\nStep 7: Risk-Return Analysis")
    
    risk_analysis = TotalUncertainty.uncertainty_risk_analysis(
        predictions=predictions,
        uncertainties=total,
        exploration_cost=1.0,
        discovery_value=100.0,
        confidence_threshold=0.7
    )
    
    n_recommended = risk_analysis['recommend_exploration'].sum()
    logger.info(f"Recommended exploration targets: {n_recommended}")
    logger.info(f"Expected net value: ${risk_analysis['net_expected_value'].sum():.2f}")
    
    # Generate comprehensive report
    logger.info("\nStep 8: Comprehensive Uncertainty Report")
    
    report = TotalUncertainty.comprehensive_uncertainty_report(
        predictions=predictions,
        aleatoric=aleatoric,
        epistemic=epistemic,
        targets=y_test
    )
    
    logger.info("\nUncertainty Decomposition:")
    logger.info(f"  Mean aleatoric: {report['summary']['mean_aleatoric']:.4f}")
    logger.info(f"  Mean epistemic: {report['summary']['mean_epistemic']:.4f}")
    logger.info(f"  Mean total: {report['summary']['mean_total_uncertainty']:.4f}")
    
    logger.info("\nUncertainty Fractions:")
    logger.info(f"  Aleatoric: {report['uncertainty_fractions']['mean_aleatoric_fraction']:.2%}")
    logger.info(f"  Epistemic: {report['uncertainty_fractions']['mean_epistemic_fraction']:.2%}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Workflow completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
