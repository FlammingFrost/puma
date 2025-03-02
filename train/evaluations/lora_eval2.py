import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.embedder import Embedder, mean_pooling
from dataset_python import PythonDataset

def load_lora_model(model_path, small_model_name, lora_config, device):
    base_model = Embedder(model_name=small_model_name)
    model = get_peft_model(base_model.model, lora_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def get_query_embeddings(dataset, embedder, device, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    query_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            query_enc = {key: value.to(device) for key, value in batch[0].items()}
            query_outputs = embedder(**query_enc)
            query_emb = mean_pooling(query_outputs, query_enc['attention_mask'])
            query_emb = F.normalize(query_emb, p=2, dim=1)
            query_embeddings.append(query_emb)
    
    return torch.cat(query_embeddings)

if __name__ == "__main__":
    model_path = "models/small_Embedder_finetune_lora.pth"
    small_model_name = "jinaai/jina-embeddings-v2-small-en"
    large_model_name = "jinaai/jina-embeddings-v2-base-code"
    
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        r=4, 
        lora_alpha=16, 
        lora_dropout=0.1, 
        target_modules=["query", "value"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(small_model_name)
    test_dataset = PythonDataset("data/python_dataset/test", tokenizer, 512)
    
    # Load the fine-tuned LoRA model for query embedding
    small_model = load_lora_model(model_path, small_model_name, lora_config, device)
    
    # Load precomputed code embeddings
    code_embeddings = torch.load("models/embeddings/test_embeddings_code.pt", map_location=device)
    
    # Get query embeddings for the test dataset
    test_query_embeddings = get_query_embeddings(test_dataset, small_model, device, batch_size=8)
    torch.save(test_query_embeddings, "train/results/small_test_embeddings_query_lora.pt")
    
    # Calculate the cosine similarity between the transformed query embeddings and the precomputed code embeddings
    cosine_similarities = F.cosine_similarity(test_query_embeddings, code_embeddings, dim=1)
    
    # Save the cosine similarities
    np.save("train/results/lora_cosine_similarities_small.npy", cosine_similarities.cpu().numpy())
