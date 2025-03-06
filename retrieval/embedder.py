import sys
import os
import torch
# import faiss
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from transformers import AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.logger import logger

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class MLPEmbedder(nn.Module):
    """
    Class for defining the MLP transformation layer to map query embedding closer to code embedding.
    """
    def __init__(self, base_model, input_dim=768, hidden_dim=512, output_dim=768 ):
        super(MLPEmbedder, self).__init__()
        self.embedder = base_model
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_enc):
        base_emb = self.embedder(input_enc)
        mapped_emb = self.fc2(self.relu(self.fc1(base_emb)))
        return mapped_emb

    def train(self, mode=True):
        """
        Override the default train() to freeze the base model's layers.
        """
        super(MLPEmbedder, self).train(mode)
        self.embedder.eval()  # Freeze the base model
        for param in self.embedder.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = mode
        for param in self.fc2.parameters():
            param.requires_grad = mode
            
class MLP(nn.Module):
    """
    Defines the MLP transformation layer to map query embedding closer to code embedding.
    Now works with precomputed embeddings instead of raw token inputs.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768, residual=False):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.residual = residual

    def forward(self, input_emb):
        if self.residual:
            mapped_emb = self.layer_norm(input_emb + self.network(input_emb))
        else:
            mapped_emb = self.network(input_emb)
        mapped_emb = F.normalize(mapped_emb, p=2, dim=1)
        return mapped_emb

class FFN(nn.Module):
    """
    Defines the Feed Forward Network (FFN) layer. It's a simple 2-layer MLP with residual connection.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768, residual=True):
        super(FFN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_dim),
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.residual = residual

    def forward(self, input_emb):
        if self.residual:
            mapped_emb = self.layer_norm(input_emb + self.network(input_emb))
        else:
            mapped_emb = self.network(input_emb)
        mapped_emb = F.normalize(mapped_emb, p=2, dim=1)
        return mapped_emb

class Embedder(nn.Module):
    """
    Class for loading a pre-trained model from Hugging Face and generating embeddings.
    """
    def __init__(self, model_name="jinaai/jina-embeddings-v2-base-code"):
        super(Embedder, self).__init__()
        self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
        logger.info(f"Embedder loaded model: {model_name}")

    def forward(self, input_enc):
        outputs = self.model(**input_enc)
        embeddings = mean_pooling(outputs, input_enc['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


# Example usage
def test_encoding():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-code")
    base_model = Embedder(model_name="jinaai/jina-embeddings-v2-base-code")
    embedder = MLPEmbedder(input_dim=768, hidden_dim=512, output_dim=768, base_model=base_model)
    
    input = ["def add(a, b): return a + b", "def subtract(a, b): return a - b"]
    input_enc = tokenizer(input, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    
    with torch.no_grad():
        embeddings = embedder(input_enc)
        
    
    print(embeddings.shape)
    print(embeddings)
    assert isinstance(embeddings, torch.Tensor), "Encoding should return a PyTorch tensor"
    assert embeddings.shape[0] == len(input), "Embeddings should have the same number of samples as input"
    assert embeddings.shape[1] > 0, "Embeddings should have a valid dimension"

    logger.info("Encoding test passed!")
    
if __name__ == "__main__":
    test_encoding()
