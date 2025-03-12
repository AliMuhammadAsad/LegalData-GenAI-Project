#!/bin/bash

# This script automates the deployment of the AWS infrastructure for the Document Q&A project.

# Set the AWS region
AWS_REGION="us-east-1"

# Deploy S3 bucket for document storage
echo "Deploying S3 bucket..."
aws cloudformation deploy --template-file ../templates/s3_bucket.yaml --stack-name document-qa-s3 --region $AWS_REGION

# Deploy Lambda function
echo "Deploying Lambda function..."
aws cloudformation deploy --template-file ../templates/lambda.yaml --stack-name document-qa-lambda --region $AWS_REGION

# Deploy API Gateway
echo "Deploying API Gateway..."
aws cloudformation deploy --template-file ../templates/api_gateway.yaml --stack-name document-qa-api --region $AWS_REGION

# Deploy SageMaker resources
echo "Deploying SageMaker resources..."
aws cloudformation deploy --template-file ../templates/sagemaker.yaml --stack-name document-qa-sagemaker --region $AWS_REGION

echo "Deployment completed successfully!"