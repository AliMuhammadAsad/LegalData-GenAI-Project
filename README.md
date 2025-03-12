# Document QA System on AWS

## Overview
This project implements a real-time Question-Answering (Q&A) system using Large Language Models (LLMs) on AWS. The system is designed to efficiently retrieve answers from large document repositories, making it ideal for industries such as legal, compliance, and customer support.

## Problem Statement
Organizations often face challenges in quickly accessing information from lengthy documents. This project addresses these challenges by providing a scalable and low-latency Q&A system that can process various document formats.

## Architecture
The architecture leverages several AWS services:
- **Amazon S3**: For storing documents.
- **Amazon Textract**: For extracting text from documents.
- **Amazon SageMaker**: For generating embeddings and running the LLM.
- **AWS Lambda**: For executing serverless functions to handle API requests.
- **Amazon API Gateway**: For managing API endpoints.

## Objectives
- Implement a scalable Q&A architecture on AWS.
- Optimize performance for latency and retrieval accuracy.
- Demonstrate real-world applications, such as legal or compliance Q&A.

## Methodology
1. **Document Storage**: Store documents in Amazon S3.
2. **Text Extraction**: Use Amazon Textract to extract text from documents.
3. **Embedding Generation**: Generate embeddings using Amazon SageMaker and store them in a vector database (e.g., FAISS).
4. **API Integration**: Use Amazon API Gateway for user interactions and AWS Lambda for processing queries.

## Usage
1. **Upload Documents**: Users can upload documents through the frontend interface.
2. **Query Documents**: Users can input queries, and the system retrieves relevant answers in real-time.
3. **View Results**: The answers are displayed on the frontend for user review.

## Performance Metrics
- **Latency**: Measure the average query response time.
- **Accuracy**: Evaluate the precision and relevance of the generated answers.

## Concrete Usage Scenario
A compliance team uploads legal documents to S3. Employees can query specific regulations, and the system provides direct answers, significantly reducing the time spent on manual searches.

## Getting Started
1. Clone the repository.
2. Set up your AWS account and configure the necessary services.
3. Run the setup script to initialize the AWS infrastructure.
4. Deploy the application using the provided deployment scripts.

## Requirements
- AWS account (preferably using the free tier)
- Python 3.x
- Node.js (for the frontend)

## Acknowledgments
This project utilizes various AWS services and open-source libraries to facilitate document processing and Q&A capabilities.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.