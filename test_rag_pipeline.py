import os
import sys
import logging
import json
from src.generation import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_rag_pipeline():
    """
    Test the complete RAG pipeline
    """
    # Sample queries
    queries = [
        "What is the main topic of the document?",
        "Summarize the key points of the document.",
        # Add more test queries relevant to your test document
    ]

    try:
        # Initialize the RAG pipeline
        rag = RAGPipeline()

        # Process each query
        for query in queries:
            logger.info(f"\n\n--- Testing query: {query} ---")

            # Process the query
            result = rag.process_query(query)

            # Log the results
            logger.info(f"Query: {query}")
            logger.info(f"Response: {result['response']}")
            logger.info(f"Retrieved {len(result['retrieved_chunks'])} chunks")
            logger.info(
                f"Retrieval time: {result['retrieval_time']:.2f} seconds")
            logger.info(
                f"Generation time: {result['generation_time']:.2f} seconds")
            logger.info(f"Total time: {result['total_time']:.2f} seconds")

            # Print top retrieved chunks for inspection
            logger.info("\nTop retrieved chunks:")
            for i, chunk in enumerate(result['retrieved_chunks']):
                logger.info(f"Chunk {i+1} (Score: {chunk['score']:.2f}):")
                logger.info(f"Document: {chunk['document_id']}")
                # First 100 chars
                logger.info(f"Text: {chunk['text'][:100]}...")

            logger.info("=" * 80)

        logger.info("RAG pipeline test completed successfully")

    except Exception as e:
        logger.error(f"Error during RAG pipeline test: {str(e)}")


if __name__ == "__main__":
    test_rag_pipeline()
