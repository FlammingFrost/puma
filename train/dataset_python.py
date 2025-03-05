import gzip
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.logger import logger

class PythonDataset(Dataset):
    """
    A custom dataset for loading Python code and docstrings from a list of query-code pairs.
    
    Args:
        query_code_pairs (list of list of str): List of pairs of query and code.
        tokenizer (AutoTokenizer): Tokenizer to use for tokenizing the data.
        max_len (int): Maximum length for tokenization.
        
    Returns:
        A custom dataset object.
    """
    def __init__(self, data_path, tokenizer, max_len=512):
        self.query_code_pairs = read_gz_files(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Loaded {len(self.query_code_pairs)} query-code pairs. PythonDataset initialized.")

    def __len__(self):
        return len(self.query_code_pairs)
    
    def __getitem__(self, idx):
        query, code = self.query_code_pairs[idx]
        query_enc = self.tokenizer(query, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        code_enc = self.tokenizer(code, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        query_enc = {key: value for key, value in query_enc.items()}
        code_enc = {key: value for key, value in code_enc.items()}
        return query_enc, code_enc

class PrecomputedEmbeddingsDataset(Dataset):
    def __init__(self, query_embeddings_path, code_embeddings_path):
        self.query_embeddings = torch.load(query_embeddings_path, map_location=torch.device('cpu'))
        self.code_embeddings = torch.load(code_embeddings_path, map_location=torch.device('cpu'))
    
    def __len__(self):
        return len(self.query_embeddings)
    
    def __getitem__(self, idx):
        return self.query_embeddings[idx], self.code_embeddings[idx]

def read_gz_files(input_folder):
    """
    Read .gz files and return a list of query-code pairs.
    
    Args:
        input_folder (str): Path to the folder containing the raw JSONL files.
        
    Returns:
        List of query-code pairs.
    """
    query_code_pairs = []
    file_paths = [os.path.join(input_folder, file_name) for file_name in os.listdir(input_folder)]
    
    for file_path in file_paths:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                query_code_pairs.append([row['docstring'], row['code']])
    
    return query_code_pairs

# Example usage
if __name__ == "__main__":
    input_folder = "data/python_dataset/valid"
    tokenizer_name = "jinaai/jina-embeddings-v2-base-code"
    max_len = 512
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create dataset
    dataset = PythonDataset(input_folder, tokenizer, max_len)
    print(f"Number of records: {len(dataset)}")
    print(f'First row: \n{dataset[0]}')