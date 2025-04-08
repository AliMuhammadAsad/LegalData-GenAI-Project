import React, { useState } from 'react';
import axios from 'axios';

const QueryInterface = () => {
    const [query, setQuery] = useState('');
    const [answer, setAnswer] = useState('');
    const [loading, setLoading] = useState(false);

    const handleQueryChange = (event) => {
        setQuery(event.target.value);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setLoading(true);
        setAnswer('');

        try {
            const response = await axios.post('/api/query', { query });
            setAnswer(response.data.answer);
        } catch (error) {
            console.error('Error fetching answer:', error);
            setAnswer('Error fetching answer. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>Document Query Interface</h2>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={query}
                    onChange={handleQueryChange}
                    placeholder="Enter your query"
                    required
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Loading...' : 'Submit'}
                </button>
            </form>
            {answer && (
                <div>
                    <h3>Answer:</h3>
                    <p>{answer}</p>
                </div>
            )}
        </div>
    );
};

export default QueryInterface;