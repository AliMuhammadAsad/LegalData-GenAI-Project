from src.utils import create_directory_if_not_exists, upload_file_to_s3
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION
import os
import json
import logging
import numpy as np
import faiss
import pickle
import time
from sentence_transformers import SentenceTransformer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingCreator:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        """
        Initialize the embedding creator with the specified model
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = EMBEDDING_DIMENSION

    def create_embeddings(self, processed_file_path):
        """
        Create embeddings for text chunks in the processed file
        
        Args:
            processed_file_path (str): Path to the processed text file
            
        Returns:
            tuple: (document_id, embeddings_file_path, index_file_path)
        """
        # Load processed text chunks
        with open(processed_file_path, 'r') as f:
            processed_data = json.load(f)

        # Extract document_id and chunks from the processed data
        document_id = processed_data['document_id']
        chunks_data = processed_data['chunks']

        if not document_id:
            raise ValueError(f"No document_id found in {processed_file_path}")

        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks_data]
        chunk_ids = [chunk['chunk_id'] for chunk in chunks_data]

        logger.info(
            f"Generating embeddings for {len(texts)} text chunks from document {document_id}")

        # Generate embeddings
        start_time = time.time()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embedding_time = time.time() - start_time
        logger.info(f"Embeddings generated in {embedding_time:.2f} seconds")

        # Create FAISS index
        index = self._create_faiss_index(embeddings)

        # Save embeddings and index
        embeddings_file, index_file = self._save_embeddings_and_index(
            document_id, embeddings, index, chunks_data
        )

        return document_id, embeddings_file, index_file

    def _create_faiss_index(self, embeddings, dim=768):
        """
        Create a FAISS index for the embeddings
        
        Args:
            embeddings (numpy.ndarray): Array of embeddings
            
        Returns:
            faiss.Index: FAISS index
        """
        # Normalize embeddings for cosine similarity
        index = faiss.IndexHNSWFlat(dim, 32)  # 32 connections per node
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def _save_embeddings_and_index(self, document_id, embeddings, index, chunks_data):
        """
        Save embeddings and FAISS index to files
        
        Args:
            document_id (str): Document identifier
            embeddings (numpy.ndarray): Array of embeddings
            index (faiss.Index): FAISS index
            chunks_data (list): Original chunks data with metadata
            
        Returns:
            tuple: (embeddings_file_path, index_file_path)
        """
        # Create embeddings directory if it doesn't exist
        embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'data', 'embeddings')
        create_directory_if_not_exists(embeddings_dir)

        # Save embeddings with metadata
        embeddings_with_metadata = {
            'document_id': document_id,
            'model_name': EMBEDDING_MODEL_NAME,
            'embedding_dimension': self.embedding_dim,
            'num_chunks': len(chunks_data),
            'chunks': chunks_data,
            'embeddings': embeddings.tolist()
        }

        embeddings_file = os.path.join(
            embeddings_dir, f"{document_id}_embeddings.json")
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings_with_metadata, f)

        # Save FAISS index
        index_file = os.path.join(embeddings_dir, f"{document_id}_index.faiss")
        faiss.write_index(index, index_file)

        logger.info(f"Saved embeddings to {embeddings_file}")
        logger.info(f"Saved FAISS index to {index_file}")

        # Upload to S3
        s3_embeddings_key = f"embeddings/{document_id}_embeddings.json"
        s3_index_key = f"embeddings/{document_id}_index.faiss"

        upload_file_to_s3(embeddings_file, s3_embeddings_key)
        upload_file_to_s3(index_file, s3_index_key)

        return embeddings_file, index_file


def create_embeddings_for_documents(processed_file_paths):
    """
    Create embeddings for multiple processed documents
    
    Args:
        processed_file_paths (list): List of paths to processed text files
        
    Returns:
        list: List of tuples (document_id, embeddings_file_path, index_file_path)
    """
    embedding_creator = EmbeddingCreator()
    results = []

    for file_path in processed_file_paths:
        try:
            result = embedding_creator.create_embeddings(file_path)
            results.append(result)
        except Exception as e:
            logger.error(
                f"Error creating embeddings for {file_path}: {str(e)}")

    return results

# Class to manage the document index across multiple documents


class DocumentIndexManager:
    def __init__(self):
        """
        Initialize the document index manager
        """
        self.index_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'data', 'embeddings')
        create_directory_if_not_exists(self.index_dir)
        self.master_index_path = os.path.join(
            self.index_dir, 'master_index.faiss')
        self.metadata_path = os.path.join(
            self.index_dir, 'master_metadata.pkl')

    def create_or_update_master_index(self, document_ids=None):
        """
        Create or update the master index containing embeddings from all documents
        
        Args:
            document_ids (list, optional): List of document IDs to include.
                                           If None, all available documents will be included.
        
        Returns:
            tuple: (master_index_path, metadata_path)
        """
        # Get all embedding files if document_ids is not specified
        if document_ids is None:
            embedding_files = [f for f in os.listdir(self.index_dir)
                               if f.endswith('_embeddings.json')]
            document_ids = [f.split('_embeddings.json')[0]
                            for f in embedding_files]

        logger.info(f"Creating master index for {len(document_ids)} documents")

        all_embeddings = []
        all_metadata = []

        # Load embeddings and metadata for each document
        for doc_id in document_ids:
            embeddings_file = os.path.join(
                self.index_dir, f"{doc_id}_embeddings.json")

            if not os.path.exists(embeddings_file):
                logger.warning(
                    f"Embeddings file not found for document {doc_id}")
                continue

            with open(embeddings_file, 'r') as f:
                embeddings_data = json.load(f)

            # Get embeddings and metadata
            embeddings = np.array(
                embeddings_data['embeddings'], dtype=np.float32)
            faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

            # Add document embeddings to the collection
            all_embeddings.append(embeddings)

            # Add metadata for each chunk
            for i, chunk in enumerate(embeddings_data['chunks']):
                metadata = {
                    'document_id': doc_id,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': i,
                    'embedding_index': len(all_metadata),
                    'text': chunk['text']
                }
                all_metadata.append(metadata)

        # If no embeddings were found, return
        if not all_embeddings:
            logger.warning("No embeddings found for any document")
            return None, None

        # Concatenate all embeddings
        combined_embeddings = np.vstack(all_embeddings)

        # Create master index
        master_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        master_index.add(combined_embeddings)

        # Save master index
        faiss.write_index(master_index, self.master_index_path)

        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)

        logger.info(
            f"Created master index with {len(all_metadata)} chunks from {len(document_ids)} documents")
        logger.info(f"Master index saved to {self.master_index_path}")
        logger.info(f"Metadata saved to {self.metadata_path}")

        # Upload to S3
        upload_file_to_s3(self.master_index_path,
                          "embeddings/master_index.faiss")
        upload_file_to_s3(self.metadata_path, "embeddings/master_metadata.pkl")

        return self.master_index_path, self.metadata_path
