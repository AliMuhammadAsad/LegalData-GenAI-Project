import json
import boto3
from src.utils.aws_helpers import get_relevant_documents, invoke_llm_endpoint

def lambda_handler(event, context):
    question = event["queryStringParameters"]["q"]
    
    # Retrieve relevant documents from the vector database
    relevant_docs = get_relevant_documents(question)
    
    # Run the question through the LLM (SageMaker endpoint)
    answer = invoke_llm_endpoint(relevant_docs, question)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'answer': answer})
    }


def handle_query(question):
    # Get relevant documents from vector DB
    relevant_docs = get_relevant_documents(question, "genaiprojectawsbucket")

    # Construct prompt with retrieved context
    prompt = f"Question: {question}\n\nContext: {' '.join(relevant_docs)}\n\nAnswer:"

    # Get answer from LLM
    answer = invoke_llm_endpoint(prompt)

    return answer
