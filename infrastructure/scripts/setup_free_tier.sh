#!/bin/bash

# This script sets up the necessary AWS services within the free tier limits.

# Set AWS region
AWS_REGION="us-east-1"
$REGION="us-east-1"

# Create S3 bucket for document storage
BUCKET_NAME="document-qa-aws-project-bucket-$(date +%s)"
aws s3api create-bucket --bucket $BUCKET_NAME --region $AWS_REGION --create-bucket-configuration LocationConstraint=$AWS_REGION

# Create IAM role for Lambda function
ROLE_NAME="lambda-execution-role"
aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://lambda_trust_policy.json

# Attach policies to the role
aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonTextractFullAccess
aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Create Lambda function
LAMBDA_FUNCTION_NAME="DocumentQALambda"
aws lambda create-function --function-name $LAMBDA_FUNCTION_NAME --runtime python3.8 --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/$ROLE_NAME --handler api.lambda_handler --zip-file fileb://path_to_your_lambda_zip_file.zip

# Create API Gateway
API_NAME="DocumentQAAPI"
aws apigateway create-rest-api --name $API_NAME

$EMBEDDING_MODEL_NAME="document-qa-embedding-model"
$EMBEDDING_INSTANCE_TYPE="ml.t2.medium"  # Free tier compatible
aws sagemaker create-model `
  --model-name $EMBEDDING_MODEL_NAME `
  --region $REGION `
  --primary-container '{
      "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
      "ModelDataUrl": "s3://aws-ml-blog/artifacts/pytorch-transformers-embedding/model.tar.gz",
      "Environment": {
          "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
          "HF_MODEL_ID": "sentence-transformers/all-MiniLM-L6-v2",
          "HF_TASK": "feature-extraction"
      }
  }' `
  --execution-role-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role"

# Create endpoint configuration
aws sagemaker create-endpoint-config `
  --endpoint-config-name "$EMBEDDING_MODEL_NAME-config" `
  --region $REGION `
  --production-variants '[{
      "VariantName": "AllTraffic",
      "ModelName": "'$EMBEDDING_MODEL_NAME'",
      "InstanceType": "'$EMBEDDING_INSTANCE_TYPE'",
      "InitialInstanceCount": 1
  }]'

# Create endpoint (this takes ~10 minutes)
aws sagemaker create-endpoint `
  --endpoint-name "$EMBEDDING_MODEL_NAME-endpoint" `
  --region $REGION `
  --endpoint-config-name "$EMBEDDING_MODEL_NAME-config"

# Save the endpoint name to a variable 
$EMBEDDING_MODEL_ENDPOINT="$EMBEDDING_MODEL_NAME-endpoint"
Write-Output "Embedding Model Endpoint: $EMBEDDING_MODEL_ENDPOINT"

# Output the created resources
echo "S3 Bucket: $BUCKET_NAME"
echo "Lambda Function: $LAMBDA_FUNCTION_NAME"
echo "API Gateway: $API_NAME"