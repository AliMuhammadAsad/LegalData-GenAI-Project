import os
import sys
import logging
import json
from src.document_processing import process_documents
from src.embedding_creation import create_embeddings_for_documents, DocumentIndexManager

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_embedding_creation():
    """
    Test the embedding creation process
    """
    # Path to test document (make sure this file exists)
    test_documents = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'data', 'raw', 'test_document.pdf')
    ]

    # Check if test document exists
    if not os.path.exists(test_documents[0]):
        logger.error(f"Test document not found: {test_documents[0]}")
        logger.info(
            "Please place a test PDF document in the 'data/raw' directory")
        return

    # Process document
    try:
        # Process documents to extract text
        processed_files = process_documents(test_documents)
        if not processed_files:
            logger.error("No documents were processed successfully")
            return

        # Create embeddings for processed documents
        embedding_results = create_embeddings_for_documents(processed_files)
        if not embedding_results:
            logger.error("No embeddings were created successfully")
            return

        for doc_id, embeddings_file, index_file in embedding_results:
            logger.info(f"Created embeddings for document {doc_id}")
            logger.info(f"Embeddings saved to {embeddings_file}")
            logger.info(f"FAISS index saved to {index_file}")

            # Load embeddings to verify
            with open(embeddings_file, 'r') as f:
                embeddings_data = json.load(f)

            logger.info(f"Number of chunks: {embeddings_data['num_chunks']}")
            logger.info(
                f"Embedding dimension: {embeddings_data['embedding_dimension']}")

        # Create master index
        index_manager = DocumentIndexManager()
        master_index_path, metadata_path = index_manager.create_or_update_master_index()

        logger.info("Embedding creation test completed successfully")

    except Exception as e:
        logger.error(f"Error during embedding creation test: {str(e)}")


if __name__ == "__main__":
    test_embedding_creation()
