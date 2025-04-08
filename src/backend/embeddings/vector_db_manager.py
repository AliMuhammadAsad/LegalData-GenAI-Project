import faiss
import numpy as np

class VectorDBManager:
    def __init__(self, dimension, index_path):
        self.dimension = dimension
        self.index_path = index_path
        self.index = self._initialize_index()

    def _initialize_index(self):
        index = faiss.IndexFlatL2(self.dimension)
        return index

    def add_embeddings(self, embeddings):
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save_index(self):
        faiss.write_index(self.index, self.index_path)

    def load_index(self):
        self.index = faiss.read_index(self.index_path)