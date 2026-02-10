"""
Training worker script for AWS Batch jobs.

This script runs inside a batch job container and trains a single model
in the ensemble. It reads configuration from S3, loads data, trains the model,
and saves results back to S3.
"""

import os
import sys
import argparse
import logging
import json
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mineral_prospectivity.models.vae_model import VAEProspectivityModel
from mineral_prospectivity.data.s3_loader import S3DataLoader
from mineral_prospectivity.utils.logging import setup_logging
from mineral_prospectivity.metrics.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a single model in the ensemble'
    )
    
    parser.add_argument(
        '--model-index',
        type=int,
        required=True,
        help='Index of this model in the ensemble'
    )
    
    parser.add_argument(
        '--config-key',
        type=str,
        required=True,
        help='S3 key for configuration file'
    )
    
    parser.add_argument(
        '--s3-bucket',
        type=str,
        default=os.environ.get('S3_BUCKET'),
        help='S3 bucket name'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=os.environ.get('EXPERIMENT_NAME'),
        help='Experiment name'
    )
    
    parser.add_argument(
        '--timestamp',
        type=str,
        default=os.environ.get('TIMESTAMP'),
        help='Timestamp for this run'
    )
    
    parser.add_argument(
        '--local-dir',
        type=str,
        default='/tmp/mineral_prospectivity',
        help='Local directory for temporary files'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    logger.info(f"Starting training worker for model {args.model_index}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Config: s3://{args.s3_bucket}/{args.config_key}")
    
    # Create local working directory
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize S3 data loader
        s3_loader = S3DataLoader(
            bucket_name=args.s3_bucket,
            experiment_name=args.experiment_name
        )
        
        # Load configuration
        logger.info("Loading configuration...")
        config = s3_loader.load_config(args.config_key)
        logger.info(f"Configuration loaded: {json.dumps(config, indent=2)}")
        
        # Load training data
        logger.info("Loading training data from S3...")
        train_loader = s3_loader.load_training_data(
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            bootstrap=config.get('use_bootstrap', False),
            random_seed=args.model_index  # Different seed for each model
        )
        
        # Load validation data
        logger.info("Loading validation data from S3...")
        val_loader = s3_loader.load_validation_data(
            batch_size=config.get('batch_size', 32)
        )
        
        # Initialize model
        logger.info("Initializing model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Set random seed BEFORE model initialization for reproducible weights
        torch.manual_seed(args.model_index + config.get('base_seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.model_index + config.get('base_seed', 42))
        
        model = VAEProspectivityModel(
            input_dim=config['input_dim'],
            latent_dim=config.get('latent_dim', 32),
            encoder_hidden_dims=config.get('encoder_hidden_dims'),
            decoder_hidden_dims=config.get('decoder_hidden_dims'),
            output_dim=config.get('output_dim', 1)
        ).to(device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        logger.info("Starting training...")
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = config.get('early_stopping_patience', 10)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(config.get('epochs', 100)):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                loss_dict = model.compute_loss(
                    batch_x,
                    batch_y,
                    beta=config.get('beta', 1.0)
                )
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    loss_dict = model.compute_loss(
                        batch_x,
                        batch_y,
                        beta=config.get('beta', 1.0)
                    )
                    epoch_val_loss += loss_dict['total_loss'].item()
            
            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
            
            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                
                # Save best model checkpoint
                best_model_path = local_dir / f"model_{args.model_index}_best.pt"
                model.save_model(str(best_model_path))
            else:
                patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{config.get('epochs', 100)}: "
                    f"Train Loss = {epoch_train_loss:.4f}, "
                    f"Val Loss = {epoch_val_loss:.4f}, "
                    f"Best Val Loss = {best_val_loss:.4f}"
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loader = s3_loader.load_test_data(batch_size=config.get('batch_size', 32))
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                
                pred_dict = model.predict_with_uncertainty(
                    batch_x,
                    num_samples=config.get('num_mc_samples', 100)
                )
                
                all_predictions.append(pred_dict['mean'].cpu())
                all_targets.append(batch_y)
                all_uncertainties.append(pred_dict['total_uncertainty'].cpu())
        
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        uncertainties = torch.cat(all_uncertainties)
        
        # Calculate metrics
        metrics = PerformanceMetrics.compute_all_metrics(
            predictions.numpy(),
            targets.numpy(),
            uncertainties.numpy()
        )
        
        logger.info(f"Test metrics: {json.dumps(metrics, indent=2)}")
        
        # Save results to S3
        logger.info("Saving results to S3...")
        
        # Save best model
        model_s3_key = f"experiments/{args.experiment_name}/models/{args.timestamp}/model_{args.model_index}.pt"
        s3_loader.upload_file(
            str(best_model_path),
            model_s3_key
        )
        logger.info(f"Saved model to s3://{args.s3_bucket}/{model_s3_key}")
        
        # Save training history
        history = {
            'model_index': args.model_index,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'test_metrics': metrics,
            'config': config
        }
        
        history_path = local_dir / f"history_{args.model_index}.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        history_s3_key = f"experiments/{args.experiment_name}/results/{args.timestamp}/history_{args.model_index}.json"
        s3_loader.upload_file(
            str(history_path),
            history_s3_key
        )
        logger.info(f"Saved training history to s3://{args.s3_bucket}/{history_s3_key}")
        
        logger.info("Training worker completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Training worker failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
