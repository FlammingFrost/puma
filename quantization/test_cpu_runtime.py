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
NUM_QUERIES = 100
NUM_TESTS = 3

# Force PyTorch to use CPU
torch.set_num_threads(os.cpu_count())

def load_model(quantization):
    """
    Load model with the specified quantization, forced to run on CPU.
    """
    print(f"Loading model with {quantization} on CPU...")
    
    try:
        if quantization == "fp32":
            return AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cpu")
        
        elif quantization == "fp16":
            # FP16 is not natively supported on CPU, so it will fallback to FP32
            print("FP16 is not well supported on CPU. Running in FP32 mode instead.")
            return AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True).to("cpu")

        elif quantization == "8bit":
            print("Applying dynamic quantization (INT8) for CPU.")
            model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cpu")
            return torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

        elif quantization == "4bit":
            print("Applying 4-bit quantization (fallback to INT8, as 4-bit is not natively supported on CPU).")
            model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cpu")
            return torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )  # 4-bit quantization is not directly supported, so fallback to INT8.

        else:
            raise ValueError(f"Unsupported quantization type: {quantization}")
    
    except Exception as e:
        print(f"Error loading model with {quantization}: {e}")
        return None

def test_runtime(quantization):
    """
    Measure inference runtime for different quantization levels on CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = load_model(quantization)

    if model is None:
        return None, None  # Skip if model loading failed

    print(f"Model loaded on {next(model.parameters()).device} (CPU)")

    model.eval()

    # Load dataset and move queries to CPU
    dataset = PythonDataset(DATASET_PATH, tokenizer, max_len=512)
    queries = [{key: value.to("cpu") for key, value in query_enc.items()} for query_enc, _ in dataset][:NUM_QUERIES]

    runtimes = []

    for _ in range(NUM_TESTS):
        start_time = time.perf_counter()

        for query_enc in tqdm(queries, desc=f"Testing {quantization} on CPU"):
            with torch.no_grad():
                _ = model(**query_enc).pooler_output

        end_time = time.perf_counter()
        runtimes.append(end_time - start_time)

    mean_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)
    
    return mean_runtime, std_runtime

def main():
    """
    Run inference benchmarking for different quantization methods on CPU.
    """
    quantizations = ["fp32", "fp16", "8bit", "4bit"]  # Testing all modes
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