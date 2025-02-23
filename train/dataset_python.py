import gzip
import json
import random
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.logger import logger

KEEP_KEYS = ["docstring", "code", "func_name", "language", "file_path"]
PythonDataPath = "tests/data/python/python_train_0.jsonl.gz"

class PythonDataset:
    """
    A custom dataset for loading Python code and docstrings from a JSONL file.
    
    Args:
        folder_path (str): Path to the folder containing the JSONL files.
        seed (int): Random seed for reproducibility.
        negative_triplets (bool): If True, generate negative triplets for training.
        negative_precomputed (bool): If True, use precomputed negative examples.
        subset (str): Subset of data to use ('train', 'test', 'valid').
        
    Returns:
        A custom dataset object.
        
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
        "negative": "code"          # Randomly sampled if `negative_precomputed` is False
    }
    """
    def __init__(self, folder_path, seed=224, negative_triplets=False, negative_precomputed=False, subset='train'):
        self.seed = seed
        self.codes = None
        self.negative_triplets = negative_triplets
        self.negative_precomputed = negative_precomputed

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
        return len(self._data)
    
    @property
    def data(self):
        return self._data

    def __getitem__(self, idx):
        if self.negative_triplets:
            if self.negative_precomputed:
                return self._data[idx]
            else:
                record = self._data[idx]
                while record['negative'] is None or record['negative'] == record['positive']:
                    record['negative'] = random.choice(self.codes)
                return record
        else:
            return self._data[idx]

    def to_hf_dataset(self):
        from datasets import Dataset
        return Dataset.from_dict({
            "anchor": [item["anchor"] for item in self._data],
            "positive": [item["positive"] for item in self._data],
            "negative": [item["negative"] for item in self._data]
        })

# Example usage
if __name__ == "__main__":
    folder_path = "tests/data/python"
    dataset = PythonDataset(folder_path)
    print(f"Number of records: {len(dataset)}")
    print(f'First row: \n{dataset[0]}')