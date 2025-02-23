import gzip
import json
import random

KEEP_KEYS = ["docstring", "code", "func_name", "language", "file_path"]

def load_data_python(folder_path, num_rows=None, seed=224):
    """
    Load local Python data from a JSONL file. Use as test data for RAG model.
    
    Args:
        folder_path (str): Path to the folder containing the JSONL file.
        keep_keys (list): List of keys to keep.
        num_rows (int): Number of rows to sample.
        seed (int): Random seed for reproducibility. Default is 224.
        
    Returns:
        data (list): List of dictionaries with the selected keys.
            keys:
                docstring: The docstring of the code.
                code: The code snippet.
                func_name: The function name.
                file_path: The file path. i.e. repo/path/to/file.py
    """
    total_tokens = 0
    total_records = 0
    data = []
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        with gzip.open(file_name, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)  # Parse each JSON line
                row['file_path'] = row['repo'] + '/' + row['path'][4:]
                data.append({k: v for k, v in row.items() if k in KEEP_KEYS})
                total_records += 1
                total_tokens += len(row["code_tokens"])

    if seed is not None:
        random.seed(seed)
    if num_rows is not None:
        data = random.sample(data, num_rows)

    print(f"Total records: {len(data)}")
    print(f"Total tokens: {total_tokens}")

    return data




def main():
    """
    This is an example of how to use the load_data_python function.
    """
    file_path = "tests/data/python/python_train_0.jsonl.gz"
    data = load_data_python(file_path, num_rows=100)
    print(f'First row: \n{data[0]}')

if __name__ == "__main__":
    main()