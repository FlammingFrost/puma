import sys
import os
import torch
import faiss
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    losses
)
from tools.logger import logger


class Embedder:
    def __init__(self, base_model="jinaai/jina-embeddings-v2-base-code", fine_tuned_model=None):
        """
        Initialize the embedder with a pre-trained model or a fine-tuned model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            fine_tuned_model if fine_tuned_model else base_model, 
            trust_remote_code=True).to(self.device)
        logger.info(f"Loaded model: {fine_tuned_model if fine_tuned_model else base_model}")
        self.index = None

    def encode(self, texts):
        """
        Encode a list of code snippets or queries.
        """
        return np.array(self.model.encode(texts, convert_to_numpy=True))
    
    def calculate_similarity(self, test_data):
        """
        Calculate cosine similarity between queries and positive examples in the test data.
        """
        queries = [item["query"] for item in test_data]
        positives = [item["positive"] for item in test_data]
        query_embeddings = self.encode(queries)
        positive_embeddings = self.encode(positives)
        similarities = np.dot(query_embeddings, positive_embeddings.T) / (np.linalg.norm(query_embeddings) * np.linalg.norm(positive_embeddings))
        
        return similarities
    
    def transform_query(self, query_embeddings):
        """
        Transform queries using a simple neural network to make query and code in the same space.
        """
        
        
        
        return query_embeddings
    
    # def build_faiss_index(self, code_snippets):
    #     """
    #     Build a FAISS index for fast retrieval of code snippets.
    #     """
    #     code_embeddings = self.encode(code_snippets)
    #     self.index = faiss.IndexFlatL2(code_embeddings.shape[1])
    #     self.index.add(code_embeddings)
    #     self.snippets = code_snippets
    #     logger.info("FAISS index built successfully.")

    # def retrieve_similar_code(self, query, top_k=2):
    #     """
    #     Retrieve top_k similar code snippets given a query.
    #     """
    #     if self.index is None:
    #         raise ValueError("FAISS index not built. Call `build_faiss_index()` first.")

    #     query_embedding = self.encode([query])
    #     D, I = self.index.search(query_embedding, k=top_k)

    #     return [self.snippets[idx] for idx in I[0]]
    

def test_encoding():
    embedder = Embedder(fine_tuned_model="models/fine_tuned_embedder")
    texts = ["def add(a, b): return a + b", "def subtract(a, b): return a - b"]
    embeddings = embedder.encode(texts)
    print(embeddings.shape)
    print(embeddings)
    
    assert isinstance(embeddings, np.ndarray), "Encoding should return a NumPy array"
    assert embeddings.shape[0] == len(texts), "Embeddings should match the number of input texts"
    assert embeddings.shape[1] > 0, "Embeddings should have a valid dimension"

    logger.info("Encoding test passed!")
    
if __name__ == "__main__":
    test_encoding()
