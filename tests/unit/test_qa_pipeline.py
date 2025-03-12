import pytest
from src.backend.qa.llm_pipeline import generate_answer
from src.backend.qa.retriever import retrieve_documents

def test_retrieve_documents(mocker):
    mock_query = "What are the compliance regulations?"
    mock_documents = ["Document 1 content", "Document 2 content"]
    
    mock_retrieve = mocker.patch('src.backend.qa.retriever.retrieve_documents', return_value=mock_documents)
    
    documents = retrieve_documents(mock_query)
    
    assert documents == mock_documents
    mock_retrieve.assert_called_once_with(mock_query)

def test_generate_answer(mocker):
    mock_documents = ["Document 1 content", "Document 2 content"]
    mock_answer = "The compliance regulations are..."
    
    mock_generate = mocker.patch('src.backend.qa.llm_pipeline.generate_answer', return_value=mock_answer)
    
    answer = generate_answer(mock_documents)
    
    assert answer == mock_answer
    mock_generate.assert_called_once_with(mock_documents)