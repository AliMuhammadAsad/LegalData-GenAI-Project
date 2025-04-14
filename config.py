import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# S3 Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Legal Document Processing Configuration
LEGAL_CHUNK_SIZE = 1200  # Larger chunks for legal context
LEGAL_CHUNK_OVERLAP = 250  # More overlap for legal documents

# Embedding Configuration
EMBEDDING_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
EMBEDDING_DIMENSION = 768  # BERT base models have 768 dimensions

# LLM Configuration
MODEL_NAME = "gpt-3.5-turbo"  # Can upgrade to gpt-4 for better legal understanding
TEMPERATURE = 0.2  # Lower temperature for more precise legal responses
MAX_TOKENS = 800   # Increased for more comprehensive legal explanations

# Application Configuration
APP_TITLE = "LexiSearch: Legal Document Q&A System"
APP_DESCRIPTION = """
Upload legal documents and ask questions about their content. 
This system uses retrieval-augmented generation to provide accurate answers based on the legal documents you've provided.
"""
DISCLAIMER = """
**Disclaimer**: This tool provides information based on the documents you upload, but it is not a substitute for professional legal advice. 
The information provided should not be construed as legal advice and no attorney-client relationship is created by using this tool.
"""
