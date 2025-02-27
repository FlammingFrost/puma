import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from retrieval.embedder import Embedder, mean_pooling
from tools.logger import logger
from dataset_python import PythonDataset

def load_lora_model(model_path, base_model_name, lora_config, device):
    base_model = Embedder(model_name=base_model_name)
    model = get_peft_model(base_model.model, lora_config)
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    return model

def get_embeddings(dataset, embedder, device, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    query_embeddings, code_embeddings = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            query_enc, code_enc = batch
            query_enc = {key: value.to(device) for key, value in query_enc.items()}
            code_enc = {key: value.to(device) for key, value in code_enc.items()}
            
            query_outputs = embedder(**query_enc)
            query_emb = mean_pooling(query_outputs, query_enc['attention_mask'])
            query_emb = F.normalize(query_emb, p=2, dim=1)
            query_embeddings.append(query_emb)
            
            code_outputs = embedder(**code_enc)
            code_emb = mean_pooling(code_outputs, code_enc['attention_mask'])
            code_emb = F.normalize(code_emb, p=2, dim=1)
            code_embeddings.append(code_emb)
    
    query_embeddings = torch.cat(query_embeddings)
    code_embeddings = torch.cat(code_embeddings)
    
    return query_embeddings, code_embeddings


if __name__ == "__main__":
    model_path = "models/Embedder_finetune_lora.pth"
    
    base_model_name = "jinaai/jina-embeddings-v2-base-code"
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        r=4, 
        lora_alpha=16, 
        lora_dropout=0.1, 
        target_modules=["query", "value"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dataset = PythonDataset("data/python_dataset/test", AutoTokenizer.from_pretrained(base_model_name), 512)
    
    # Load the fine-tuned LoRA model
    model = load_lora_model(model_path, base_model_name, lora_config, device)
    
    # Get embeddings for the test dataset
    test_query_embeddings, test_code_embeddings = get_embeddings(test_dataset, model, device, batch_size=8)
    
    
    # Calculate the cosine similarity between the transformed query embeddings and the code embeddings
    cosine_similarities = F.cosine_similarity(test_query_embeddings, test_code_embeddings, dim=1)
    
    # Save the cosine similarities
    np.save("train/results/lora_cosine_similarities.npy", cosine_similarities.cpu().numpy())