import gzip
import json
import random
from datasets import Dataset

# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.logger import logger
PythonDataPath = "tests/data/python/python_train_0.jsonl.gz"

class PythonDataset(Dataset):
    """
    A PyTorch Dataset for loading Python code and docstrings from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file containing the data.
        seed (int): Random seed for reproducibility.
        negative_triplets (bool): If True, generate negative triplets for training.
        negative_precomputed (bool): If True, use precomputed negative examples.
        
    Returns:
        A PyTorch Dataset object.
        
    Data format:
    
    If `negative_triplets` is False:
    {
        "anchor": "docstring",
        "positive": "code"
        "negative": None
    }
    
    If `negative_triplets` is True:
    {
        "anchor": "docstring",
        "positive": "code"
        "negative": "code"          # Randomly sampled  if `negative_precomputed` is False
    }
    """
    def __init__(self, file_path, seed=224, negative_triplets=False, negative_precomputed=False):
        self.file_path = file_path
        self.seed = seed
        self.codes = None
        self.negative_triplets = negative_triplets
        self.negative_precomputed = negative_precomputed
        self._data = self.load_data()

        if negative_precomputed and not negative_triplets:
            raise ValueError("negative_triplets must be True when negative_precomputed is True")

    def load_data(self):
        total_tokens = 0
        total_records = 0
        data = []

        with gzip.open(self.file_path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)  # Parse each JSON line
                record = {
                    'anchor': row['docstring'],
                    'positive': row['code'],
                    'negative': None
                }
                total_records += 1
                total_tokens += len(row["code_tokens"])
                data.append(record)
                
        codes = [row["positive"] for row in data]
        logger.debug(f"Total records: {len(data)}")
        logger.debug(f"Total tokens: {total_tokens}")
        
        if self.negative_precomputed:
            for i, row in enumerate(data):
                # Get the negative example from the precomputed list
                while row['negative'] is None or row['negative'] == row['positive']:
                    row['negative'] = random.choice(codes)
                data[i] = row
        self.codes = codes
        
        return data

    def __len__(self):
        return len(self.data)
    
    @property
    def data(self):
        return self._data

    def __getitem__(self, idx):
        if self.negative_triplets:
            if self.negative_precomputed:
                return self.data[idx]
            else:
                record = self.data[idx]
                while record['negative'] is None or record['negative'] == record['positive']:
                    record['negative'] = random.choice(self.codes)
                return record
        else:
            # return {"anchor": self.data[idx]['anchor'], "positive": self.data[idx]['positive']}
            return self.data[idx]
                

# Example usage
if __name__ == "__main__":
    import os, sys
    file_path = "tests/data/python/python_train_0.jsonl.gz"
    dataset = PythonDataset(file_path)
    print(f"Number of records: {len(dataset)}")
    # print(f'First row: \n{dataset[0]}')