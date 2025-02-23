import sys
import os
import torch
# import faiss
import torch.nn as nn
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    losses
)
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset
from tools.logger import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.logger import logger

class MLPEmbedder(nn.Module):
    """
    Class for defining the MLP transformation layer to map query embedding closer to code embedding.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768, base_model="jinaai/jina-embeddings-v2-base-code", fine_tuned_model=None, device="cuda"):
        super(MLPEmbedder, self).__init__()
        self.embedder = Embedder(base_model, fine_tuned_model)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, query):
        embedding = self.embedder(query)
        embedding = torch.tensor(embedding).to(self.device)
        return self.fc2(self.relu(self.fc1(embedding)))
    
    def to(self, device):
        self.device = device
        self.embedder.to(device)
        return self
    
    def train(self)-> None:
        for param in self.embedder.parameters():
            param.requires_grad = False
        for layer in [self.fc1, self.fc2]:
            for param in layer.parameters():
                param.requires_grad = True
                
    def eval(self):
        for param in self.parameters():
            param.requires_grad = False

class Embedder(nn.Module):
    """
    Class for loading a pre-trained Sentence Transformer model.
    """
    def __init__(self, base_model="jinaai/jina-embeddings-v2-base-code", fine_tuned_model=None):
        super(Embedder, self).__init__()
        self.model = SentenceTransformer(
            fine_tuned_model if fine_tuned_model else base_model, 
            trust_remote_code=True)
        logger.info(f"Loaded model: {fine_tuned_model if fine_tuned_model else base_model}")
        self.index = None

    def forward(self, query):
        return self.model.encode(query)
        
    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

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
    # def retrieve_similar_code(self, query, top_k=2):
    #     """
    #     Retrieve top_k similar code snippets given a query.
    #     """
    #     if self.index is None:
    #         raise ValueError("FAISS index not built. Call `build_faiss_index()` first.")

    #     query_embedding = self.encode([query])
    #     D, I = self.index.search(query_embedding, k=top_k)
    #     query_embedding = self.encode([query])
    #     D, I = self.index.search(query_embedding, k=top_k)

    #     return [self.snippets[idx] for idx in I[0]]
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
