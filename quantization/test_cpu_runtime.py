import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.dataset_python import PythonDataset

# Constants
MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
DATASET_PATH = "data/python_dataset/test"
RESULTS_FILE = "quantization/cpu_runtime_results.txt"
NUM_QUERIES = 1000
NUM_TESTS = 3

def load_model(quantization):
    """
    Load the model with specified quantization settings.
    """
    try:
        if quantization == "fp32":
            return AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cuda")
        elif quantization == "fp16":
            return AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
        elif quantization == "8bit":
            return AutoModel.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto", trust_remote_code=True)
        elif quantization == "4bit":
            return AutoModel.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map="auto", trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported quantization type: {quantization}")
    except Exception as e:
        print(f"Error loading model with {quantization}: {e}")
        return None

def test_runtime(quantization):
    """
    Measure inference runtime for different quantization levels.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = load_model(quantization)

    if model is None:
        return None, None  # Skip if model loading failed

    device = model.device
    print(f"Model loaded on {device}")
    model.eval()

    # Load dataset and move queries to the correct device
    dataset = PythonDataset(DATASET_PATH, tokenizer, max_len=512)
    queries = [{key: value.to(device) for key, value in query_enc.items()} for query_enc, _ in dataset][:NUM_QUERIES]

    runtimes = []
    
    for _ in range(NUM_TESTS):
        torch.cuda.synchronize() if torch.cuda.is_available() else None  # Ensure accurate timing
        start_time = time.perf_counter()
        
        for query_enc in tqdm(queries, desc=f"Testing {quantization}"):
            with torch.no_grad():
                _ = model(**query_enc).pooler_output

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        runtimes.append(end_time - start_time)

    mean_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)
    
    return mean_runtime, std_runtime

def main():
    """
    Run inference benchmarking for different quantization methods.
    """
    quantizations = ["8bit", "4bit", "fp32", "fp16"]
    results = {}

    for quantization in quantizations:
        mean_runtime, std_runtime = test_runtime(quantization)
        if mean_runtime is not None:
            results[quantization] = (mean_runtime, std_runtime)
            print(f"{quantization} - Mean: {mean_runtime:.4f}s, Std: {std_runtime:.4f}s")

    # Save results to a file
    with open(RESULTS_FILE, "w") as f:
        for quantization, (mean_runtime, std_runtime) in results.items():
            f.write(f"{quantization} - Mean: {mean_runtime:.4f}s, Std: {std_runtime:.4f}s\n")

if __name__ == "__main__":
    main()