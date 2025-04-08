from flask import Flask, request, jsonify
from src.backend.api.lambda_handler import handle_query

app = Flask(__name__)

@app.route('/api/query', methods=['POST'])
def query_document():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    answer = handle_query(question)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)