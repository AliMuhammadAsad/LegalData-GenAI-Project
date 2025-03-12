import json
import boto3

def generate_answer(question, context):
    # Initialize the SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')

    # Prepare the payload for the LLM
    payload = {
        "question": question,
        "context": context
    }

    # Call the SageMaker endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='your-sagemaker-endpoint-name',
        ContentType='application/json',
        Body=json.dumps(payload)
    )

    # Parse the response
    result = json.loads(response['Body'].read().decode())
    return result['answer']

def llm_pipeline(question, retrieved_docs):
    # Combine retrieved documents into a single context
    context = " ".join(retrieved_docs)

    # Generate an answer using the LLM
    answer = generate_answer(question, context)
    return answer