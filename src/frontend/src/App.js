import React, { useState } from 'react';
import DocumentUpload from './components/DocumentUpload';
import QueryInterface from './components/QueryInterface';
import './App.css';

function App() {
    const [documents, setDocuments] = useState([]);
    const [query, setQuery] = useState('');
    const [answer, setAnswer] = useState('');

    const handleDocumentUpload = (uploadedDocuments) => {
        setDocuments(uploadedDocuments);
    };

    const handleQueryChange = (newQuery) => {
        setQuery(newQuery);
    };

    const handleAnswerReceived = (newAnswer) => {
        setAnswer(newAnswer);
    };

    return (
        <div className="App">
            <h1>Document Q&A System</h1>
            <DocumentUpload onUpload={handleDocumentUpload} />
            <QueryInterface 
                documents={documents} 
                query={query} 
                onQueryChange={handleQueryChange} 
                onAnswerReceived={handleAnswerReceived} 
            />
            {answer && <div className="answer">Answer: {answer}</div>}
        </div>
    );
}

export default App;