# AWS Batch Setup Guide

This guide explains how to set up the AWS infrastructure for running mineral prospectivity ensemble training.

## Prerequisites

1. AWS Account with appropriate permissions
2. AWS CLI configured
3. Docker installed locally
4. ECR repository created

## Infrastructure Components

### 1. S3 Bucket

Create an S3 bucket to store:
- Training/validation/test data
- Experiment configurations
- Trained models
- Results and metrics

```bash
aws s3 mb s3://mineral-prospectivity-data --region us-east-1
```

### 2. IAM Roles

The framework requires two IAM roles:

#### Batch Job Role
Allows batch jobs to access S3 and write logs.

```bash
aws iam create-role \
    --role-name MineralProspectivityBatchJobRole \
    --assume-role-policy-document file://infrastructure/job_role_trust_policy.json

aws iam put-role-policy \
    --role-name MineralProspectivityBatchJobRole \
    --policy-name JobPolicy \
    --policy-document file://infrastructure/job_role_policy.json
```

#### Batch Execution Role
Allows ECS to pull container images and write logs.

```bash
aws iam create-role \
    --role-name MineralProspectivityBatchExecutionRole \
    --assume-role-policy-document file://infrastructure/execution_role_trust_policy.json

aws iam attach-role-policy \
    --role-name MineralProspectivityBatchExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

### 3. Docker Container

Build and push the Docker container:

```bash
# Build container
docker build -t mineral-prospectivity:latest .

# Tag for ECR
docker tag mineral-prospectivity:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mineral-prospectivity:latest

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Push to ECR
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mineral-prospectivity:latest
```

### 4. Compute Environment

Create a compute environment for running batch jobs:

```bash
aws batch create-compute-environment --cli-input-json file://infrastructure/compute_environment.json
```

### 5. Job Queue

Create a job queue:

```bash
aws batch create-job-queue --cli-input-json file://infrastructure/job_queue.json
```

### 6. Job Definitions

Register job definitions:

```bash
aws batch register-job-definition --cli-input-json file://infrastructure/job_definitions.json
```

## Alternative: CloudFormation

Use the provided CloudFormation template to create all resources at once:

```bash
aws cloudformation create-stack \
    --stack-name mineral-prospectivity-infrastructure \
    --template-body file://infrastructure/cloudformation_template.json \
    --parameters \
        ParameterKey=S3BucketName,ParameterValue=mineral-prospectivity-data \
        ParameterKey=VpcId,ParameterValue=vpc-xxxxx \
        ParameterKey=SubnetIds,ParameterValue=subnet-xxxxx\\,subnet-yyyyy \
        ParameterKey=ECRRepositoryUri,ParameterValue=YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mineral-prospectivity \
    --capabilities CAPABILITY_IAM
```

## Programmatic Setup

Use the Python API to set up infrastructure:

```python
from mineral_prospectivity.aws_batch.infrastructure import setup_batch_infrastructure

resources = setup_batch_infrastructure(
    region_name='us-east-1',
    vpc_id='vpc-xxxxx',
    subnet_ids=['subnet-xxxxx', 'subnet-yyyyy'],
    s3_bucket='mineral-prospectivity-data'
)

print(f"Created resources: {resources}")
```

## Verification

Check that all resources are created:

```python
from mineral_prospectivity.aws_batch.infrastructure import check_infrastructure_status

status = check_infrastructure_status(region_name='us-east-1')
print(status)
```

## Cost Considerations

- **Compute**: Use Spot instances to reduce costs by up to 90%
- **Storage**: Use S3 Intelligent-Tiering for automatic cost optimization
- **Logs**: Set log retention policies to avoid excessive CloudWatch costs
- **GPU Instances**: p3.2xlarge typically costs ~$3/hour, consider g4dn instances for lower cost

## Cleanup

To tear down infrastructure:

```python
from mineral_prospectivity.aws_batch.infrastructure import teardown_batch_infrastructure

teardown_batch_infrastructure(region_name='us-east-1')
```

Or delete the CloudFormation stack:

```bash
aws cloudformation delete-stack --stack-name mineral-prospectivity-infrastructure
```
