"""
AWS Batch infrastructure setup and management.
"""

import boto3
import json
import logging
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError

from .job_definitions import (
    create_job_definitions,
    get_iam_policies,
    get_compute_environment_spec,
    get_job_queue_spec
)

logger = logging.getLogger(__name__)


def setup_batch_infrastructure(
    region_name: str = 'us-east-1',
    vpc_id: Optional[str] = None,
    subnet_ids: Optional[list] = None,
    s3_bucket: Optional[str] = None
) -> Dict[str, str]:
    """
    Set up AWS Batch infrastructure for mineral prospectivity modeling.
    
    This function creates:
    - IAM roles for batch jobs
    - Compute environment
    - Job queue
    - Job definitions
    
    Args:
        region_name: AWS region
        vpc_id: VPC ID for compute environment
        subnet_ids: List of subnet IDs
        s3_bucket: S3 bucket name
        
    Returns:
        Dictionary with created resource ARNs/names
    """
    
    # Initialize clients
    batch_client = boto3.client('batch', region_name=region_name)
    iam_client = boto3.client('iam', region_name=region_name)
    
    resources = {}
    
    try:
        # Create IAM roles
        logger.info("Creating IAM roles...")
        iam_policies = get_iam_policies()
        
        # Job role
        try:
            job_role_response = iam_client.create_role(
                RoleName='MineralProspectivityBatchJobRole',
                AssumeRolePolicyDocument=json.dumps(
                    iam_policies['job_role_trust_policy']
                ),
                Description='Role for mineral prospectivity batch jobs'
            )
            
            # Attach policy
            iam_client.put_role_policy(
                RoleName='MineralProspectivityBatchJobRole',
                PolicyName='JobPolicy',
                PolicyDocument=json.dumps(iam_policies['job_role_policy'])
            )
            
            resources['job_role_arn'] = job_role_response['Role']['Arn']
            logger.info(f"Created job role: {resources['job_role_arn']}")
        
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                logger.info("Job role already exists")
                role = iam_client.get_role(
                    RoleName='MineralProspectivityBatchJobRole'
                )
                resources['job_role_arn'] = role['Role']['Arn']
            else:
                raise
        
        # Execution role
        try:
            exec_role_response = iam_client.create_role(
                RoleName='MineralProspectivityBatchExecutionRole',
                AssumeRolePolicyDocument=json.dumps(
                    iam_policies['execution_role_trust_policy']
                ),
                Description='Execution role for mineral prospectivity batch jobs'
            )
            
            # Attach AWS managed policy
            iam_client.attach_role_policy(
                RoleName='MineralProspectivityBatchExecutionRole',
                PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
            )
            
            resources['execution_role_arn'] = exec_role_response['Role']['Arn']
            logger.info(f"Created execution role: {resources['execution_role_arn']}")
        
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                logger.info("Execution role already exists")
                role = iam_client.get_role(
                    RoleName='MineralProspectivityBatchExecutionRole'
                )
                resources['execution_role_arn'] = role['Role']['Arn']
            else:
                raise
        
        # Create compute environment
        if vpc_id and subnet_ids:
            logger.info("Creating compute environment...")
            
            compute_env_spec = get_compute_environment_spec()
            compute_env_spec['computeResources']['subnets'] = subnet_ids
            
            try:
                compute_env_response = batch_client.create_compute_environment(
                    **compute_env_spec
                )
                resources['compute_environment_arn'] = compute_env_response['computeEnvironmentArn']
                logger.info(f"Created compute environment: {resources['compute_environment_arn']}")
            
            except ClientError as e:
                if 'already exists' in str(e):
                    logger.info("Compute environment already exists")
                    envs = batch_client.describe_compute_environments(
                        computeEnvironments=['mineral-prospectivity-compute-env']
                    )
                    if envs['computeEnvironments']:
                        resources['compute_environment_arn'] = envs['computeEnvironments'][0]['computeEnvironmentArn']
                else:
                    raise
        
        # Create job queue
        logger.info("Creating job queue...")
        
        job_queue_spec = get_job_queue_spec()
        if 'compute_environment_arn' in resources:
            job_queue_spec['computeEnvironmentOrder'][0]['computeEnvironment'] = \
                resources['compute_environment_arn']
        
        try:
            job_queue_response = batch_client.create_job_queue(**job_queue_spec)
            resources['job_queue_arn'] = job_queue_response['jobQueueArn']
            resources['job_queue_name'] = job_queue_response['jobQueueName']
            logger.info(f"Created job queue: {resources['job_queue_arn']}")
        
        except ClientError as e:
            if 'already exists' in str(e):
                logger.info("Job queue already exists")
                queues = batch_client.describe_job_queues(
                    jobQueues=['mineral-prospectivity-queue']
                )
                if queues['jobQueues']:
                    resources['job_queue_arn'] = queues['jobQueues'][0]['jobQueueArn']
                    resources['job_queue_name'] = queues['jobQueues'][0]['jobQueueName']
            else:
                raise
        
        # Register job definitions
        logger.info("Registering job definitions...")
        
        job_defs = create_job_definitions()
        
        for job_type, job_def in job_defs.items():
            # Update with created role ARNs
            if 'job_role_arn' in resources:
                job_def['containerProperties']['jobRoleArn'] = resources['job_role_arn']
            if 'execution_role_arn' in resources:
                job_def['containerProperties']['executionRoleArn'] = resources['execution_role_arn']
            
            try:
                response = batch_client.register_job_definition(**job_def)
                resources[f'{job_type}_job_definition_arn'] = response['jobDefinitionArn']
                logger.info(f"Registered {job_type} job definition: {response['jobDefinitionArn']}")
            
            except ClientError as e:
                logger.error(f"Error registering {job_type} job definition: {e}")
        
        logger.info("Infrastructure setup complete!")
        return resources
    
    except Exception as e:
        logger.error(f"Error setting up infrastructure: {e}")
        raise


def teardown_batch_infrastructure(region_name: str = 'us-east-1'):
    """
    Tear down AWS Batch infrastructure.
    
    Warning: This will delete all resources. Use with caution!
    
    Args:
        region_name: AWS region
    """
    
    batch_client = boto3.client('batch', region_name=region_name)
    iam_client = boto3.client('iam', region_name=region_name)
    
    logger.warning("Starting infrastructure teardown...")
    
    try:
        # Disable and delete job queue
        try:
            batch_client.update_job_queue(
                jobQueue='mineral-prospectivity-queue',
                state='DISABLED'
            )
            logger.info("Disabled job queue")
            
            # Wait for queue to be disabled
            import time
            time.sleep(10)
            
            batch_client.delete_job_queue(
                jobQueue='mineral-prospectivity-queue'
            )
            logger.info("Deleted job queue")
        except ClientError as e:
            logger.warning(f"Error deleting job queue: {e}")
        
        # Disable and delete compute environment
        try:
            batch_client.update_compute_environment(
                computeEnvironment='mineral-prospectivity-compute-env',
                state='DISABLED'
            )
            logger.info("Disabled compute environment")
            
            # Wait for environment to be disabled
            time.sleep(10)
            
            batch_client.delete_compute_environment(
                computeEnvironment='mineral-prospectivity-compute-env'
            )
            logger.info("Deleted compute environment")
        except ClientError as e:
            logger.warning(f"Error deleting compute environment: {e}")
        
        # Delete IAM roles
        for role_name in [
            'MineralProspectivityBatchJobRole',
            'MineralProspectivityBatchExecutionRole'
        ]:
            try:
                # Detach policies first
                policies = iam_client.list_attached_role_policies(RoleName=role_name)
                for policy in policies['AttachedPolicies']:
                    iam_client.detach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy['PolicyArn']
                    )
                
                # Delete inline policies
                inline_policies = iam_client.list_role_policies(RoleName=role_name)
                for policy_name in inline_policies['PolicyNames']:
                    iam_client.delete_role_policy(
                        RoleName=role_name,
                        PolicyName=policy_name
                    )
                
                # Delete role
                iam_client.delete_role(RoleName=role_name)
                logger.info(f"Deleted IAM role: {role_name}")
            
            except ClientError as e:
                logger.warning(f"Error deleting role {role_name}: {e}")
        
        logger.info("Infrastructure teardown complete!")
    
    except Exception as e:
        logger.error(f"Error during teardown: {e}")
        raise


def check_infrastructure_status(region_name: str = 'us-east-1') -> Dict[str, str]:
    """
    Check status of AWS Batch infrastructure.
    
    Args:
        region_name: AWS region
        
    Returns:
        Dictionary with status of each component
    """
    
    batch_client = boto3.client('batch', region_name=region_name)
    iam_client = boto3.client('iam', region_name=region_name)
    
    status = {}
    
    # Check compute environment
    try:
        envs = batch_client.describe_compute_environments(
            computeEnvironments=['mineral-prospectivity-compute-env']
        )
        if envs['computeEnvironments']:
            status['compute_environment'] = envs['computeEnvironments'][0]['status']
        else:
            status['compute_environment'] = 'NOT_FOUND'
    except Exception as e:
        status['compute_environment'] = f'ERROR: {e}'
    
    # Check job queue
    try:
        queues = batch_client.describe_job_queues(
            jobQueues=['mineral-prospectivity-queue']
        )
        if queues['jobQueues']:
            status['job_queue'] = queues['jobQueues'][0]['status']
        else:
            status['job_queue'] = 'NOT_FOUND'
    except Exception as e:
        status['job_queue'] = f'ERROR: {e}'
    
    # Check IAM roles
    for role_name in [
        'MineralProspectivityBatchJobRole',
        'MineralProspectivityBatchExecutionRole'
    ]:
        try:
            iam_client.get_role(RoleName=role_name)
            status[role_name] = 'EXISTS'
        except ClientError:
            status[role_name] = 'NOT_FOUND'
    
    return status
