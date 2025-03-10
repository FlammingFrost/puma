import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.dataset_python import PythonDataset

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
DATASET_PATH = "data/python_dataset/test"
RESULTS_FILE = "quantization/cpu_runtime_results.txt"
NUM_QUERIES = 1000
NUM_TESTS = 10

def load_model(quantization):
    if quantization == "fp32":
        return AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cpu")
    elif quantization == "fp16":
        return AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True).to("cpu")
    elif quantization == "8bit":
        return AutoModel.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto", trust_remote_code=True)
    elif quantization == "4bit":
        return AutoModel.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map="auto", trust_remote_code=True)
    else:
        raise ValueError("Unsupported quantization type")

def test_runtime(quantization):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = load_model(quantization)
    model.eval()
    
    dataset = PythonDataset(DATASET_PATH, tokenizer, max_len=512)
    queries = [query_enc for query_enc, _ in dataset][:NUM_QUERIES]
    
    runtimes = []
    for _ in range(NUM_TESTS):
        start_time = time.time()
        for query_enc in tqdm(queries, desc=f"Testing {quantization}"):
            query_enc = {key: value.to("cpu") for key, value in query_enc.items()}
            with torch.no_grad():
                _ = model(**query_enc).pooler_output
        end_time = time.time()
        runtimes.append(end_time - start_time)
    
    mean_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)
    return mean_runtime, std_runtime

def main():
    quantizations = ["fp32", "fp16", "8bit", "4bit"]
    results = {}
    
    for quantization in quantizations:
        mean_runtime, std_runtime = test_runtime(quantization)
        results[quantization] = (mean_runtime, std_runtime)
        print(f"{quantization} - Mean: {mean_runtime:.4f}s, Std: {std_runtime:.4f}s")
    
    with open(RESULTS_FILE, "w") as f:
        for quantization, (mean_runtime, std_runtime) in results.items():
            f.write(f"{quantization} - Mean: {mean_runtime:.4f}s, Std: {std_runtime:.4f}s\n")

if __name__ == "__main__":
    main()
