import os
import sys
import logging
from src.document_processing import process_documents

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_document_processing():
    """
    Test document processing functionality
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
        processed_files = process_documents(test_documents)
        if processed_files:
            logger.info(
                f"Successfully processed {len(processed_files)} documents")
            for file_path in processed_files:
                logger.info(f"Processed file saved at: {file_path}")
        else:
            logger.warning("No documents were processed successfully")
    except Exception as e:
        logger.error(f"Error during document processing test: {str(e)}")


if __name__ == "__main__":
    test_document_processing()
