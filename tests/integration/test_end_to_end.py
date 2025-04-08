import json
import requests

def test_end_to_end():
    # Step 1: Upload a document to S3
    document_path = 'path/to/test/document.pdf'
    upload_response = requests.post('http://localhost:3000/api/upload', files={'file': open(document_path, 'rb')})
    assert upload_response.status_code == 200

    # Step 2: Query the document
    query = "What are the compliance requirements?"
    query_response = requests.post('http://localhost:3000/api/query', json={'query': query})
    assert query_response.status_code == 200

    # Step 3: Validate the response
    response_data = json.loads(query_response.text)
    assert 'answer' in response_data
    assert response_data['answer'] is not None

    print("End-to-end test passed!")