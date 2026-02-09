"""
AWS Batch job definitions and infrastructure templates.

This module provides functions to create and manage AWS Batch job definitions
for the mineral prospectivity ensemble training pipeline.
"""

import json
from typing import Dict, Any, Optional


def create_job_definitions() -> Dict[str, Any]:
    """
    Create AWS Batch job definition templates.
    
    Returns:
        Dictionary containing job definition specifications
    """
    
    # Main training job definition
    training_job_def = {
        "jobDefinitionName": "mineral-prospectivity-training",
        "type": "container",
        "containerProperties": {
            "image": "your-ecr-registry/mineral-prospectivity:latest",
            "vcpus": 4,
            "memory": 16384,  # 16 GB
            "command": [
                "python",
                "-m",
                "mineral_prospectivity.aws_batch.train_worker"
            ],
            "jobRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/BatchJobRole",
            "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/BatchExecutionRole",
            "resourceRequirements": [
                {
                    "type": "GPU",
                    "value": "1"
                }
            ],
            "environment": [
                {
                    "name": "PYTHONUNBUFFERED",
                    "value": "1"
                }
            ],
            "mountPoints": [],
            "volumes": [],
            "ulimits": [],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/aws/batch/job",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "mineral-prospectivity"
                }
            }
        },
        "retryStrategy": {
            "attempts": 3,
            "evaluateOnExit": [
                {
                    "action": "RETRY",
                    "onStatusReason": "Task failed to start"
                },
                {
                    "action": "EXIT",
                    "onExitCode": "0"
                }
            ]
        },
        "timeout": {
            "attemptDurationSeconds": 14400  # 4 hours
        },
        "platformCapabilities": ["EC2"]
    }
    
    # Inference job definition (for making predictions with trained ensemble)
    inference_job_def = {
        "jobDefinitionName": "mineral-prospectivity-inference",
        "type": "container",
        "containerProperties": {
            "image": "your-ecr-registry/mineral-prospectivity:latest",
            "vcpus": 2,
            "memory": 8192,  # 8 GB
            "command": [
                "python",
                "-m",
                "mineral_prospectivity.aws_batch.inference_worker"
            ],
            "jobRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/BatchJobRole",
            "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/BatchExecutionRole",
            "resourceRequirements": [
                {
                    "type": "GPU",
                    "value": "1"
                }
            ],
            "environment": [
                {
                    "name": "PYTHONUNBUFFERED",
                    "value": "1"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/aws/batch/job",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "mineral-prospectivity-inference"
                }
            }
        },
        "retryStrategy": {
            "attempts": 2
        },
        "timeout": {
            "attemptDurationSeconds": 3600  # 1 hour
        },
        "platformCapabilities": ["EC2"]
    }
    
    return {
        "training": training_job_def,
        "inference": inference_job_def
    }


def get_iam_policies() -> Dict[str, Any]:
    """
    Get IAM policy documents required for AWS Batch jobs.
    
    Returns:
        Dictionary containing IAM policies
    """
    
    # Job role policy (permissions jobs need to access AWS resources)
    job_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::YOUR_BUCKET_NAME/*",
                    "arn:aws:s3:::YOUR_BUCKET_NAME"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage"
                ],
                "Resource": "*"
            }
        ]
    }
    
    # Execution role trust policy
    execution_role_trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # Job role trust policy
    job_role_trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    return {
        "job_role_policy": job_role_policy,
        "execution_role_trust_policy": execution_role_trust_policy,
        "job_role_trust_policy": job_role_trust_policy
    }


def get_compute_environment_spec() -> Dict[str, Any]:
    """
    Get compute environment specification for AWS Batch.
    
    Returns:
        Compute environment configuration
    """
    
    compute_environment = {
        "computeEnvironmentName": "mineral-prospectivity-compute-env",
        "type": "MANAGED",
        "state": "ENABLED",
        "computeResources": {
            "type": "EC2",
            "allocationStrategy": "BEST_FIT_PROGRESSIVE",
            "minvCpus": 0,
            "maxvCpus": 256,
            "desiredvCpus": 0,
            "instanceTypes": [
                "p3.2xlarge",  # GPU instances for training
                "p3.8xlarge"
            ],
            "subnets": [
                "subnet-xxxxx",  # Replace with your subnet IDs
                "subnet-yyyyy"
            ],
            "securityGroupIds": [
                "sg-xxxxx"  # Replace with your security group ID
            ],
            "instanceRole": "arn:aws:iam::YOUR_ACCOUNT:instance-profile/ecsInstanceRole",
            "tags": {
                "Name": "mineral-prospectivity-batch",
                "Project": "mineral-prospectivity"
            }
        }
    }
    
    return compute_environment


def get_job_queue_spec() -> Dict[str, Any]:
    """
    Get job queue specification for AWS Batch.
    
    Returns:
        Job queue configuration
    """
    
    job_queue = {
        "jobQueueName": "mineral-prospectivity-queue",
        "state": "ENABLED",
        "priority": 1,
        "computeEnvironmentOrder": [
            {
                "order": 1,
                "computeEnvironment": "mineral-prospectivity-compute-env"
            }
        ],
        "tags": {
            "Project": "mineral-prospectivity"
        }
    }
    
    return job_queue


def generate_cloudformation_template() -> Dict[str, Any]:
    """
    Generate AWS CloudFormation template for complete infrastructure.
    
    Returns:
        CloudFormation template as dictionary
    """
    
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "AWS Batch infrastructure for Mineral Prospectivity Modeling",
        "Parameters": {
            "S3BucketName": {
                "Type": "String",
                "Description": "S3 bucket name for data and results",
                "Default": "mineral-prospectivity-data"
            },
            "VpcId": {
                "Type": "AWS::EC2::VPC::Id",
                "Description": "VPC ID for compute environment"
            },
            "SubnetIds": {
                "Type": "List<AWS::EC2::Subnet::Id>",
                "Description": "Subnet IDs for compute environment"
            },
            "ECRRepositoryUri": {
                "Type": "String",
                "Description": "ECR repository URI for Docker image"
            }
        },
        "Resources": {
            "DataBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {
                    "BucketName": {"Ref": "S3BucketName"},
                    "VersioningConfiguration": {
                        "Status": "Enabled"
                    },
                    "Tags": [
                        {
                            "Key": "Project",
                            "Value": "mineral-prospectivity"
                        }
                    ]
                }
            },
            "BatchJobRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "ecs-tasks.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole"
                            }
                        ]
                    },
                    "ManagedPolicyArns": [
                        "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
                    ],
                    "Policies": [
                        {
                            "PolicyName": "S3Access",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Effect": "Allow",
                                        "Action": [
                                            "s3:GetObject",
                                            "s3:PutObject",
                                            "s3:ListBucket"
                                        ],
                                        "Resource": [
                                            {"Fn::Sub": "arn:aws:s3:::${S3BucketName}/*"},
                                            {"Fn::Sub": "arn:aws:s3:::${S3BucketName}"}
                                        ]
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            "ComputeEnvironment": {
                "Type": "AWS::Batch::ComputeEnvironment",
                "Properties": {
                    "ComputeEnvironmentName": "mineral-prospectivity-compute-env",
                    "Type": "MANAGED",
                    "State": "ENABLED",
                    "ComputeResources": {
                        "Type": "EC2",
                        "AllocationStrategy": "BEST_FIT_PROGRESSIVE",
                        "MinvCpus": 0,
                        "MaxvCpus": 256,
                        "DesiredvCpus": 0,
                        "InstanceTypes": ["p3.2xlarge", "p3.8xlarge"],
                        "Subnets": {"Ref": "SubnetIds"},
                        "InstanceRole": {"Fn::GetAtt": ["ECSInstanceRole", "Arn"]}
                    }
                }
            },
            "JobQueue": {
                "Type": "AWS::Batch::JobQueue",
                "Properties": {
                    "JobQueueName": "mineral-prospectivity-queue",
                    "State": "ENABLED",
                    "Priority": 1,
                    "ComputeEnvironmentOrder": [
                        {
                            "Order": 1,
                            "ComputeEnvironment": {"Ref": "ComputeEnvironment"}
                        }
                    ]
                }
            },
            "ECSInstanceRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "ec2.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole"
                            }
                        ]
                    },
                    "ManagedPolicyArns": [
                        "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
                    ]
                }
            }
        },
        "Outputs": {
            "S3BucketName": {
                "Description": "S3 bucket for data and results",
                "Value": {"Ref": "DataBucket"}
            },
            "JobQueueArn": {
                "Description": "ARN of the job queue",
                "Value": {"Ref": "JobQueue"}
            },
            "ComputeEnvironmentArn": {
                "Description": "ARN of the compute environment",
                "Value": {"Ref": "ComputeEnvironment"}
            }
        }
    }
    
    return template


def save_infrastructure_templates(output_dir: str = "infrastructure"):
    """
    Save all infrastructure templates to files.
    
    Args:
        output_dir: Directory to save templates
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save job definitions
    job_defs = create_job_definitions()
    with open(os.path.join(output_dir, "job_definitions.json"), "w") as f:
        json.dump(job_defs, f, indent=2)
    
    # Save IAM policies
    iam_policies = get_iam_policies()
    with open(os.path.join(output_dir, "iam_policies.json"), "w") as f:
        json.dump(iam_policies, f, indent=2)
    
    # Save compute environment spec
    compute_env = get_compute_environment_spec()
    with open(os.path.join(output_dir, "compute_environment.json"), "w") as f:
        json.dump(compute_env, f, indent=2)
    
    # Save job queue spec
    job_queue = get_job_queue_spec()
    with open(os.path.join(output_dir, "job_queue.json"), "w") as f:
        json.dump(job_queue, f, indent=2)
    
    # Save CloudFormation template
    cf_template = generate_cloudformation_template()
    with open(os.path.join(output_dir, "cloudformation_template.json"), "w") as f:
        json.dump(cf_template, f, indent=2)
    
    print(f"Infrastructure templates saved to {output_dir}/")
