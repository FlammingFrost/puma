import sys
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.logger import logger
from train.dataset_python import PythonDataset
from retrieval.embedder import Embedder

def get_embeddings(dataset, embedder, batch_size=8):
    """
    Get embeddings for the entire dataset using the provided embedder.
    
    Args:
        dataset (Dataset): The dataset to encode.
        embedder (nn.Module): The embedder model.
        batch_size (int): Batch size for DataLoader.
        
    Returns:
        Tensors of query embeddings and code embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder.to(device).eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    query_embeddings, code_embeddings = [], []

    with torch.no_grad():
        for batch in dataloader:
            query_enc, code_enc = batch
            query_enc = {key: value.to(device) for key, value in query_enc.items()}
            code_enc = {key: value.to(device) for key, value in code_enc.items()}
            
            query_emb = embedder(query_enc)
            code_emb = embedder(code_enc)
            
            query_embeddings.append(query_emb)
            code_embeddings.append(code_emb)

    query_embeddings = torch.cat(query_embeddings)
    code_embeddings = torch.cat(code_embeddings)

    return query_embeddings, code_embeddings

def calculate_cosine_similarities(query_embeddings, code_embeddings):
    """
    Calculate cosine similarities between each pair of query embedding and code embedding using PyTorch.
    
    Args:
        query_embeddings (Tensor): Tensor of query embeddings.
        code_embeddings (Tensor): Tensor of code embeddings.
        
    Returns:
        Tensor of cosine similarities.
    """
    similarities = torch.nn.functional.cosine_similarity(query_embeddings, code_embeddings, dim=1)
    return similarities
    
if __name__ == "__main__":
    test_folder = "data/python_dataset/test"
    tokenizer_name = "jinaai/jina-embeddings-v2-base-code"
    max_len = 512
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create dataset
    test_dataset = PythonDataset(test_folder, tokenizer, max_len)
    
    # Initialize embedder
    embedder = Embedder(model_name=tokenizer_name)
    
    # Get embeddings
    query_embeddings, code_embeddings = get_embeddings(test_dataset, embedder)
    # Save the embeddings as PyTorch tensors
    torch.save(query_embeddings, "train/results/query_embeddings.pt")
    torch.save(code_embeddings, "train/results/code_embeddings.pt")
    
    # Calculate cosine similarities
    similarities = calculate_cosine_similarities(query_embeddings, code_embeddings)
    
    # Save the cosine similarities
    np.save("train/results/cosine_similarities.npy", similarities.cpu().numpy())