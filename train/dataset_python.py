import gzip
import json
import random
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.logger import logger

KEEP_KEYS = ["docstring", "code", "func_name", "language", "file_path"]
PythonDataPath = "tests/data/python/python_train_0.jsonl.gz"

class PythonDataset(Dataset):
    """
    A custom dataset for loading Python code and docstrings from a JSONL file.
    
    Args:
        folder_path (str): Path to the folder containing the JSONL files.
        seed (int): Random seed for reproducibility.
        negative_triplets (bool): If True, generate negative triplets for training.
        negative_precomputed (bool): If True, use precomputed negative examples.
        subset (str): Subset of data to use ('train', 'test', 'valid').
        tokenizer (AutoTokenizer): Tokenizer to use for tokenizing the data.
        
    Returns:
        A custom dataset object.
    """
    def __init__(self, folder_path, seed=224, negative_triplets=False, negative_precomputed=False, subset='train', tokenizer=None):
        self.seed = seed
        self.codes = None
        self.negative_triplets = negative_triplets
        self.negative_precomputed = negative_precomputed
        self.tokenizer = tokenizer

        if negative_precomputed and not negative_triplets:
            raise ValueError("negative_triplets must be True when negative_precomputed is True")
        if subset not in ['train', 'test', 'valid']:
            raise ValueError("subset must be one of 'train', 'test', 'valid'")
        self._data = self.load_data(os.path.join(folder_path, subset))

    def load_data(self, folder_path):
        total_tokens = 0
        total_records = 0
        data = []
        file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)]
        for file_path in tqdm(file_paths, desc="Loading data"):
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)  # Parse each JSON line
                    record = [
                        row['docstring'],  # anchor
                        row['code']       # positive
                    ]
                    if self.tokenizer:
                        record[0] = self.tokenizer(row['docstring'], return_tensors='pt', padding=True, truncation=True)
                        record[1] = self.tokenizer(row['code'], return_tensors='pt', padding=True, truncation=True)
                    total_records += 1
                    total_tokens += len(row["code_tokens"])
                    data.append(record)
        codes = [row[1] for row in data]
        logger.debug(f"Total records: {len(data)}")
        logger.debug(f"Total tokens: {total_tokens}")
        self.codes = codes
        
        return data

    def __len__(self):
        return len(self._data)
    
    @property
    def data(self):
        return self._data

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.data[i] for i in idx]
        else:
            return self.data[idx]

    def to_hf_dataset(self):
        from datasets import Dataset
        # TODO: Add negative examples
        return Dataset.from_dict({
            "anchor": [self.tokenizer.decode(item[0]['input_ids'][0], skip_special_tokens=True) if self.tokenizer else item[0] for item in self._data],
            "positive": [self.tokenizer.decode(item[1]['input_ids'][0], skip_special_tokens=True) if self.tokenizer else item[1] for item in self._data],
            "negative": [self.tokenizer.decode(item[2]['input_ids'][0], skip_special_tokens=True) if len(item) > 2 and self.tokenizer else item[2] if len(item) > 2 else None for item in self._data]
        })

# Example usage
if __name__ == "__main__":
    folder_path = "tests/data/python"
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-code")
    dataset = PythonDataset(folder_path, tokenizer=tokenizer)
    print(f"Number of records: {len(dataset)}")
    print(f'First row: \n{dataset[0]}')