# Document QA System on AWS

## Overview
This project implements a real-time Question-Answering (Q&A) system using Large Language Models (LLMs) on AWS. The system is designed to efficiently retrieve answers from large document repositories, making it ideal for industries such as legal, compliance, and customer support.

## Problem Statement
Organizations often face challenges in quickly accessing information from lengthy documents. This project addresses these challenges by providing a scalable and low-latency Q&A system that can process various document formats.

## Architecture
The architecture leverages several AWS services:
- **Amazon S3**: For storing documents.
- **Amazon Textract**: For extracting text from documents.

## Objectives
- Implement a scalable Q&A architecture on AWS.
- Optimize performance for latency and retrieval accuracy.
- Demonstrate real-world applications, such as legal or compliance Q&A.

## Methodology
1. **Document Processing**: Cleaning and structuring documents. 
2. **Embedding Generation**: Creating vector representations, and storing them in S3.
3. **Hybrid Retrieval**: Combinging retrieval methods to create a hybrid retrieval method.
4. **Response Generation**: Assessing system effectiveness on various metrics.

## Usage
1. **Upload Documents**: Users can upload documents through the frontend interface.
2. **Query Documents**: Users can input queries, and the system retrieves relevant answers in real-time.
3. **View Results**: The answers are displayed on the frontend for user review.

## Performance Metrics
We used Llama3 as a judge which performed evaluation on the following metrics.
- **Average Processing Time**: Measure the average query response time.
- **Faithfullness**: Measures if the answer is supported
- **Answer Relevance**: Assesses how well the answer addresses the question
- **Completeness**: Checks if the answer fully covers the question's scope
- **Processing Time**: Tracks retrieval time, generation time, and total time per query

## Concrete Usage Scenario
A compliance team uploads legal documents to S3. Employees can query specific regulations, and the system provides direct answers, significantly reducing the time spent on manual searches.

## Acknowledgments
This project utilizes various AWS services and open-source libraries to facilitate document processing and Q&A capabilities.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
