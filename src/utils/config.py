import os

class Config:
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'genaiprojectawsbucket')
    TEXTRACT_ROLE_ARN = os.getenv('TEXTRACT_ROLE_ARN', 'your-textract-role-arn')
    SAGEMAKER_ENDPOINT = os.getenv(
        'SAGEMAKER_ENDPOINT', 'document-qa-embedding-model-endpoint')
    VECTOR_DB_URL = os.getenv('VECTOR_DB_URL', 'your-vector-db-url')
    API_GATEWAY_URL = os.getenv('API_GATEWAY_URL', 'your-api-gateway-url')