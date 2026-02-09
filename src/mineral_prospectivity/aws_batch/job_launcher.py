"""
AWS Batch job launcher for distributed ensemble training.

This module provides functionality to launch and manage AWS Batch jobs
for training ensemble models in parallel.
"""

import boto3
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class BatchJobLauncher:
    """
    Manages AWS Batch job submission and monitoring for ensemble training.
    
    This class handles:
    - Submitting batch jobs for individual ensemble members
    - Monitoring job status
    - Retrieving job logs and results
    - Managing S3 data locations
    
    Args:
        job_queue: Name of the AWS Batch job queue
        job_definition: Name/ARN of the job definition
        s3_bucket: S3 bucket for data and results
        region_name: AWS region
    """
    
    def __init__(
        self,
        job_queue: str,
        job_definition: str,
        s3_bucket: str,
        region_name: str = 'us-east-1'
    ):
        self.job_queue = job_queue
        self.job_definition = job_definition
        self.s3_bucket = s3_bucket
        self.region_name = region_name
        
        # Initialize AWS clients
        self.batch_client = boto3.client('batch', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.logs_client = boto3.client('logs', region_name=region_name)
        
        logger.info(
            f"Initialized BatchJobLauncher with queue={job_queue}, "
            f"definition={job_definition}, bucket={s3_bucket}"
        )
    
    def launch_ensemble_training(
        self,
        num_models: int,
        config: Dict[str, Any],
        experiment_name: str,
        job_name_prefix: str = "mineral-prospectivity"
    ) -> List[str]:
        """
        Launch AWS Batch jobs to train an ensemble of models.
        
        Args:
            num_models: Number of models in ensemble
            config: Configuration dictionary for training
            experiment_name: Name for this experiment
            job_name_prefix: Prefix for job names
            
        Returns:
            List of job IDs
        """
        job_ids = []
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Upload config to S3
        config_key = f"experiments/{experiment_name}/config_{timestamp}.json"
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=config_key,
            Body=json.dumps(config, indent=2)
        )
        
        logger.info(f"Uploaded config to s3://{self.s3_bucket}/{config_key}")
        
        # Submit jobs for each model in ensemble
        for model_idx in range(num_models):
            job_name = f"{job_name_prefix}-{experiment_name}-model-{model_idx}-{timestamp}"
            
            # Prepare environment variables
            environment = [
                {'name': 'MODEL_INDEX', 'value': str(model_idx)},
                {'name': 'NUM_MODELS', 'value': str(num_models)},
                {'name': 'EXPERIMENT_NAME', 'value': experiment_name},
                {'name': 'S3_BUCKET', 'value': self.s3_bucket},
                {'name': 'CONFIG_KEY', 'value': config_key},
                {'name': 'TIMESTAMP', 'value': timestamp},
            ]
            
            # Submit job
            try:
                response = self.batch_client.submit_job(
                    jobName=job_name,
                    jobQueue=self.job_queue,
                    jobDefinition=self.job_definition,
                    containerOverrides={
                        'environment': environment,
                        'command': [
                            'python',
                            '-m',
                            'mineral_prospectivity.aws_batch.train_worker',
                            '--model-index', str(model_idx),
                            '--config-key', config_key
                        ]
                    }
                )
                
                job_id = response['jobId']
                job_ids.append(job_id)
                
                logger.info(f"Submitted job {job_name} with ID {job_id}")
                
            except Exception as e:
                logger.error(f"Failed to submit job for model {model_idx}: {e}")
                raise
        
        logger.info(f"Successfully launched {len(job_ids)} training jobs")
        return job_ids
    
    def monitor_jobs(
        self,
        job_ids: List[str],
        poll_interval: int = 30,
        timeout: int = 7200
    ) -> Dict[str, str]:
        """
        Monitor the status of submitted jobs.
        
        Args:
            job_ids: List of job IDs to monitor
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait (seconds)
            
        Returns:
            Dictionary mapping job IDs to final status
        """
        start_time = time.time()
        job_statuses = {job_id: 'SUBMITTED' for job_id in job_ids}
        
        terminal_statuses = {'SUCCEEDED', 'FAILED'}
        
        while True:
            elapsed_time = time.time() - start_time
            
            if elapsed_time > timeout:
                logger.warning(f"Monitoring timeout after {timeout} seconds")
                break
            
            # Check if all jobs are complete
            if all(status in terminal_statuses for status in job_statuses.values()):
                logger.info("All jobs completed")
                break
            
            # Update job statuses
            for job_id in job_ids:
                if job_statuses[job_id] not in terminal_statuses:
                    try:
                        response = self.batch_client.describe_jobs(jobs=[job_id])
                        if response['jobs']:
                            job = response['jobs'][0]
                            new_status = job['status']
                            
                            if new_status != job_statuses[job_id]:
                                logger.info(
                                    f"Job {job_id} status changed: "
                                    f"{job_statuses[job_id]} -> {new_status}"
                                )
                                job_statuses[job_id] = new_status
                    
                    except Exception as e:
                        logger.error(f"Error checking status for job {job_id}: {e}")
            
            # Log summary
            status_counts = {}
            for status in job_statuses.values():
                status_counts[status] = status_counts.get(status, 0) + 1
            
            logger.info(f"Job status summary: {status_counts}")
            
            # Wait before next poll
            time.sleep(poll_interval)
        
        return job_statuses
    
    def get_job_logs(self, job_id: str) -> Optional[List[str]]:
        """
        Retrieve CloudWatch logs for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            List of log messages or None if unavailable
        """
        try:
            # Get job details
            response = self.batch_client.describe_jobs(jobs=[job_id])
            if not response['jobs']:
                logger.error(f"Job {job_id} not found")
                return None
            
            job = response['jobs'][0]
            
            # Get log stream name
            if 'container' in job and 'logStreamName' in job['container']:
                log_stream = job['container']['logStreamName']
                log_group = '/aws/batch/job'
                
                # Retrieve logs
                log_response = self.logs_client.get_log_events(
                    logGroupName=log_group,
                    logStreamName=log_stream
                )
                
                messages = [event['message'] for event in log_response['events']]
                return messages
            else:
                logger.warning(f"Log stream not available for job {job_id}")
                return None
        
        except Exception as e:
            logger.error(f"Error retrieving logs for job {job_id}: {e}")
            return None
    
    def check_s3_data_requirements(self) -> Dict[str, bool]:
        """
        Check if required data exists in S3.
        
        Returns:
            Dictionary indicating which data components are present
        """
        required_paths = {
            'training_data': 'data/training/',
            'validation_data': 'data/validation/',
            'test_data': 'data/test/',
            'features_metadata': 'data/features_metadata.json'
        }
        
        results = {}
        
        for name, prefix in required_paths.items():
            try:
                # Check if path exists
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=prefix,
                    MaxKeys=1
                )
                
                exists = 'Contents' in response and len(response['Contents']) > 0
                results[name] = exists
                
                if exists:
                    logger.info(f"✓ Found {name} at s3://{self.s3_bucket}/{prefix}")
                else:
                    logger.warning(f"✗ Missing {name} at s3://{self.s3_bucket}/{prefix}")
            
            except Exception as e:
                logger.error(f"Error checking {name}: {e}")
                results[name] = False
        
        return results
    
    def download_ensemble_results(
        self,
        experiment_name: str,
        local_directory: str,
        timestamp: Optional[str] = None
    ) -> List[str]:
        """
        Download trained ensemble models from S3.
        
        Args:
            experiment_name: Name of experiment
            local_directory: Local directory to save models
            timestamp: Specific timestamp to download (latest if None)
            
        Returns:
            List of downloaded file paths
        """
        import os
        
        # Determine S3 prefix
        if timestamp:
            prefix = f"experiments/{experiment_name}/models/{timestamp}/"
        else:
            prefix = f"experiments/{experiment_name}/models/"
        
        try:
            # List objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No results found at s3://{self.s3_bucket}/{prefix}")
                return []
            
            downloaded_files = []
            
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                local_path = os.path.join(local_directory, filename)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Download file
                self.s3_client.download_file(
                    self.s3_bucket,
                    key,
                    local_path
                )
                
                downloaded_files.append(local_path)
                logger.info(f"Downloaded {key} to {local_path}")
            
            logger.info(f"Downloaded {len(downloaded_files)} files")
            return downloaded_files
        
        except Exception as e:
            logger.error(f"Error downloading results: {e}")
            raise
