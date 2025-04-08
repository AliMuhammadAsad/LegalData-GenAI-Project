import boto3
import numpy as np

class EmbeddingGenerator:
    def __init__(self, sagemaker_endpoint):
        self.sagemaker_endpoint = sagemaker_endpoint
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')

    def generate_embeddings(self, text):
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.sagemaker_endpoint,
            ContentType='application/json',
            Body=json.dumps({"text": text})
        )
        embeddings = json.loads(response['Body'].read().decode())
        return np.array(embeddings['embeddings'])  # Assuming the response contains an 'embeddings' key

def main():
    # Example usage
    sagemaker_endpoint = 'your-sagemaker-endpoint-name'
    generator = EmbeddingGenerator(sagemaker_endpoint)
    text = "Sample text for embedding generation."
    embeddings = generator.generate_embeddings(text)
    print(embeddings)

if __name__ == "__main__":
    main()