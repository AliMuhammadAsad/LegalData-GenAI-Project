import os
import boto3
from botocore.exceptions import ClientError
import logging
import time
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_aws_client(service_name):
    """
    Create and return an AWS client for the specified service.
    
    Args:
        service_name (str): The name of the AWS service (e.g., 's3', 'textract')
        
    Returns:
        boto3.client: The AWS client
    """
    try:
        client = boto3.client(
            service_name,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        return client
    except Exception as e:
        logger.error(f"Error creating AWS client for {service_name}: {str(e)}")
        raise

def upload_file_to_s3(file_path, object_name=None):
    """
    Upload a file to an S3 bucket
    
    Args:
        file_path (str): Path to the file to upload
        object_name (str, optional): S3 object name. If not specified, the file name is used
        
    Returns:
        bool: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    # Upload the file
    s3_client = get_aws_client('s3')
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, object_name)
        logger.info(f"Uploaded {file_path} to {S3_BUCKET_NAME}/{object_name}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        return False

def download_file_from_s3(object_name, file_path):
    """
    Download a file from an S3 bucket
    
    Args:
        object_name (str): S3 object name
        file_path (str): Path where the file should be saved
        
    Returns:
        bool: True if file was downloaded, else False
    """
    s3_client = get_aws_client('s3')
    try:
        s3_client.download_file(S3_BUCKET_NAME, object_name, file_path)
        logger.info(f"Downloaded {S3_BUCKET_NAME}/{object_name} to {file_path}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading file from S3: {str(e)}")
        return False

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it doesn't exist
    
    Args:
        directory_path (str): Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")