import boto3
import json
import os

def extract_text_from_document(document):
    textract = boto3.client('textract')
    
    response = textract.detect_document_text(
        Document={'S3Object': {'Bucket': os.environ['S3_BUCKET'], 'Name': document}}
    )
    
    text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            text += item['Text'] + '\n'
    
    return text

def lambda_handler(event, context):
    document_name = event['document_name']
    
    extracted_text = extract_text_from_document(document_name)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'extracted_text': extracted_text})
    }