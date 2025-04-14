from src.utils import create_directory_if_not_exists
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION
import os
import json
import logging
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import sys
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentRetriever:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        """
        Initialize the document retriever
        
        Args:
            model_name (str): Name of the embedding model to use
        """
        # Base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Paths
        self.index_dir = os.path.join(base_dir, 'data', 'embeddings')
        self.master_index_path = os.path.join(
            self.index_dir, 'master_index.faiss')
        self.metadata_path = os.path.join(
            self.index_dir, 'master_metadata.pkl')

        # Check if master index exists
        if not os.path.exists(self.master_index_path) or not os.path.exists(self.metadata_path):
            logger.warning(
                "Master index or metadata not found. Creating new index...")
            from src.embedding_creation import DocumentIndexManager
            index_manager = DocumentIndexManager()
            index_manager.create_or_update_master_index()

        # Load the master index
        self.index = faiss.read_index(self.master_index_path)

        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Initialize embedding model
        logger.info(f"Loading legal domain embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self._build_bm25_index()
        self._init_cross_encoder()

        logger.info(
            f"Document retriever initialized with {len(self.metadata)} chunks")
        
    def _init_cross_encoder(self):
        """Initialize cross-encoder model for reranking"""
        cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading cross-encoder model: {cross_encoder_name}")
        
        self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_name)
        self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_name)
        self.cross_encoder.to(self.device)
        
    def _rerank_with_cross_encoder(self, query, results, top_k=5):
        """Rerank results using cross-encoder"""
        if not results:
            return results
            
        texts = [result['text'] for result in results]
        
        # Prepare cross-encoder inputs
        cross_inp = [[query, text] for text in texts]
        
        # Tokenize
        cross_features = self.cross_encoder_tokenizer(
            cross_inp,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        # Score
        with torch.no_grad():
            cross_scores = self.cross_encoder(**cross_features).logits.flatten()
            
        # Convert to numpy array
        cross_scores = cross_scores.cpu().numpy()
        
        # Add cross-encoder scores to results
        for i, result in enumerate(results):
            result['cross_score'] = float(cross_scores[i])
        
        # Re-sort based on cross-encoder scores
        reranked_results = sorted(results, key=lambda x: x['cross_score'], reverse=True)[:top_k]
        
        return reranked_results
        
    def search(self, query, top_k=5, alpha=0.7, rerank=True):
        # First get hybrid search results
        hybrid_results = self._hybrid_search(query, top_k=top_k*2, alpha=alpha)  # Get more results for reranking
        
        # Then rerank with cross-encoder
        if rerank and hybrid_results:
            final_results = self._rerank_with_cross_encoder(query, hybrid_results, top_k=top_k)
        else:
            final_results = hybrid_results[:top_k]
            
        return final_results
    
    def _build_bm25_index(self):
        """Build BM25 index from document chunks"""
        # Tokenize all documents for BM25
        self.bm25_corpus = []
        self.doc_mapping = []  # Maps BM25 index to metadata index

        if not hasattr(self, 'metadata') or not self.metadata:
            logger.warning("No metadata available for BM25 indexing")
            return

        for idx, meta in enumerate(self.metadata):
            # Tokenize text (simple whitespace tokenization for example)
            tokenized_doc = meta['text'].lower().split()
            self.bm25_corpus.append(tokenized_doc)
            self.doc_mapping.append(idx)

        # Create BM25 index
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        logger.info(f"BM25 index built with {len(self.bm25_corpus)} documents")

    def _encode_query(self, query):
        """Extract embeddings for the query using the BERT model"""
        # Tokenize query
        encoded_input = self.tokenizer([query], padding=True, truncation=True,
                                       max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(self.device)
                         for k, v in encoded_input.items()}

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean pooling
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / \
            torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return embeddings.cpu().numpy()

    def _hybrid_search(self, query, top_k=10, alpha=0.7):
        """
        Hybrid search combining semantic and lexical retrieval
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            alpha (float): Weight for semantic search (0-1)
                          1.0 = only semantic, 0.0 = only BM25
        
        Returns:
            list: List of dictionaries containing retrieved chunks and metadata
        """
        # 1. Semantic search with FAISS
        query_embedding = self._encode_query(query)
        faiss.normalize_L2(query_embedding)
        semantic_scores, semantic_indices = self.index.search(
            query_embedding, top_k * 2)  # Get more results for fusion

        # 2. BM25 lexical search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Get top BM25 results
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        top_bm25_scores = bm25_scores[top_bm25_indices]

        # Map BM25 indices to metadata indices
        top_bm25_meta_indices = [self.doc_mapping[i] for i in top_bm25_indices]

        # 3. Score normalization and fusion
        # Create dictionary to hold combined scores for each document
        combined_scores = {}

        # Normalize semantic scores to 0-1 range
        if len(semantic_scores[0]) > 0:
            min_sem_score = min(semantic_scores[0])
            max_sem_score = max(semantic_scores[0])
            range_sem = max_sem_score - min_sem_score if max_sem_score > min_sem_score else 1

            for i, idx in enumerate(semantic_indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    norm_score = (
                        semantic_scores[0][i] - min_sem_score) / range_sem
                    combined_scores[idx] = alpha * norm_score

        # Normalize BM25 scores to 0-1 range
        if len(top_bm25_scores) > 0:
            min_bm25_score = min(top_bm25_scores)
            max_bm25_score = max(top_bm25_scores)
            range_bm25 = max_bm25_score - min_bm25_score if max_bm25_score > min_bm25_score else 1

            for i, idx in enumerate(top_bm25_meta_indices):
                if idx in combined_scores:
                    norm_score = (
                        top_bm25_scores[i] - min_bm25_score) / range_bm25
                    combined_scores[idx] += (1 - alpha) * norm_score
                else:
                    norm_score = (
                        top_bm25_scores[i] - min_bm25_score) / range_bm25
                    combined_scores[idx] = (1 - alpha) * norm_score

        # Sort documents by combined score
        sorted_docs = sorted(combined_scores.items(),
                             key=lambda x: x[1], reverse=True)[:top_k]

        # Format results
        results = []
        for idx, score in sorted_docs:
            result = {
                'score': float(score),
                'text': self.metadata[idx]['text'],
                'document_id': self.metadata[idx]['document_id'],
                'chunk_id': self.metadata[idx]['chunk_id']
            }
            results.append(result)

        return results

    def format_context(self, results):
        """
        Format retrieved results into context for the LLM
        
        Args:
            results (list): List of retrieved results
            
        Returns:
            str: Formatted context string
        """
        if not results:
            return "No relevant information found."

        context_parts = []

        for i, result in enumerate(results):
            doc_id = result['document_id']
            score = result['score']
            text = result['text']

            # Format the content
            context_part = f"[Document: {doc_id}, Relevance: {score:.2f}]\n{text}\n"
            context_parts.append(context_part)

        return "\n".join(context_parts)
