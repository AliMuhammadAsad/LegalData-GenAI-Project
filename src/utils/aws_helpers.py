import boto3
import json
# from src.utils.aws_helpers import extract_text_from_document, invoke_sagemaker_endpoint

def upload_to_s3(file_name, bucket_name='genaiprojectawsbucket', object_name=None):
    s3_client = boto3.client('s3')
    if object_name is None:
        object_name = file_name
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False
    return True

def extract_text_from_document(bucket_name, document_name):
    textract_client = boto3.client('textract')
    response = textract_client.detect_document_text(
        Document={'S3Object': {'Bucket': bucket_name, 'Name': document_name}}
    )
    return response


def extract_text_from_document(bucket_name, document_name):
    textract = boto3.client('textract')

    response = textract.detect_document_text(
        Document={'S3Object': {'Bucket': bucket_name, 'Name': document_name}}
    )

    text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            text += item['Text'] + '\n'

    return text


def invoke_sagemaker_endpoint(endpoint_name, payload):
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    return json.loads(response['Body'].read().decode())

    # Create the functions needed by lambda_handler.py


def get_relevant_documents(query, s3_bucket):
    # Implement document retrieval logic here
    # For now, return a placeholder
    return ["This is a placeholder for retrieved documents"]


def invoke_llm_endpoint(prompt):
    # Implement LLM invocation here
    # For now, return a placeholder
    return f"This is a placeholder answer for: {prompt}"

def create_presigned_url(bucket_name, object_name, expiration=3600):
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
    return response