import faiss
import numpy as np

class DocumentRetriever:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def retrieve(self, query_embedding, top_k=5):
        distances, indices = self.vector_db.search(query_embedding, top_k)
        return indices, distances

    def load_embeddings(self, embeddings):
        self.vector_db.add(embeddings)

    def create_query_embedding(self, query, model):
        # Assuming model is a pre-trained embedding model
        return model.encode(query).reshape(1, -1)  # Reshape for FAISS

# Example usage:
# vector_db = faiss.IndexFlatL2(embedding_dimension)
# retriever = DocumentRetriever(vector_db)
# query_embedding = retriever.create_query_embedding("What is the compliance regulation?", model)
# indices, distances = retriever.retrieve(query_embedding)