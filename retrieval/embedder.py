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
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.logger import logger

class MLPEmbedder(nn.Module):
    """
    Class for defining the MLP transformation layer to map query embedding closer to code embedding.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768, base_model="jinaai/jina-embeddings-v2-base-code", fine_tuned_model=None, device="cuda"):
        super(MLPEmbedder, self).__init__()
        self.embedder = Embedder(base_model if not fine_tuned_model else fine_tuned_model)
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
    Class for loading a pre-trained model from Hugging Face and generating embeddings.
    """
    def __init__(self, model_name="jinaai/jina-embeddings-v2-base-code"):
        super(Embedder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        logger.info(f"Loaded model: {model_name}")

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

# Example usage
def test_encoding():
    embedder = Embedder(model_name="jinaai/jina-embeddings-v2-base-code")
    texts = ["def add(a, b): return a + b", "def subtract(a, b): return a - b"]
    embeddings = embedder(texts)
    print(embeddings.shape)
    print(embeddings)
    
    assert isinstance(embeddings, torch.Tensor), "Encoding should return a PyTorch tensor"
    assert embeddings.shape[0] == len(texts), "Embeddings should match the number of input texts"
    assert embeddings.shape[1] > 0, "Embeddings should have a valid dimension"

    logger.info("Encoding test passed!")
    
if __name__ == "__main__":
    test_encoding()
