import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api'; // Replace with your actual API Gateway URL

export const uploadDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error uploading document:', error);
        throw error;
    }
};

export const queryDocument = async (query) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/query`, { query });
        return response.data.answer;
    } catch (error) {
        console.error('Error querying document:', error);
        throw error;
    }
};