import argparse
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
                
from retrieval.embedder import Embedder, mean_pooling
from dataset_python import PythonDataset

def fine_tune_with_lora(args):
    print('args:', args)
    if args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    
    tokenizer_small = AutoTokenizer.from_pretrained(args.small_model_name)
    tokenizer_large = AutoTokenizer.from_pretrained(args.large_model_name)
    
    train_dataset = PythonDataset(args.train_data, tokenizer_small, args.max_len)
    eval_dataset = PythonDataset(args.eval_data, tokenizer_small, args.max_len)
    
    small_model = Embedder(model_name=args.small_model_name)
    large_model = Embedder(model_name=args.large_model_name)
    large_model.model.to(args.device)  # Keep large model on device but unchanged
    large_model.model.eval()  # Freeze large model

    # Apply LoRA to the small model
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        r=4, 
        lora_alpha=16, 
        lora_dropout=0.1, 
        target_modules=["query", "value"]
    )
    small_model.model = get_peft_model(small_model.model, lora_config)
    small_model.model.to(args.device)
    
    optimizer = torch.optim.Adam(small_model.model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")  # Initialize AMP GradScaler
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    for epoch in range(args.epochs):
        small_model.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}"):
            query_enc, code_enc = batch
            query_enc = {key: value.to(args.device) for key, value in query_enc.items()}
            code_enc = {key: value.to(args.device) for key, value in code_enc.items()}
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):  # Enable mixed precision
                query_outputs = small_model.model(**query_enc)
                query_emb = mean_pooling(query_outputs, query_enc['attention_mask'])
                query_emb = F.normalize(query_emb, p=2, dim=1)
                
                with torch.no_grad():  # Keep large model frozen
                    code_outputs = large_model.model(**code_enc)
                    code_emb = mean_pooling(code_outputs, code_enc['attention_mask'])
                    code_emb = F.normalize(code_emb, p=2, dim=1)
                
                loss = 1.0 - torch.nn.CosineSimilarity(dim=1)(query_emb, code_emb).mean()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {avg_loss:.4f}")
        
        small_model.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}/{args.epochs}"):
                query_enc, code_enc = batch
                query_enc = {key: value.to(args.device) for key, value in query_enc.items()}
                code_enc = {key: value.to(args.device) for key, value in code_enc.items()}
                
                with torch.amp.autocast("cuda"):  # Enable mixed precision
                    query_outputs = small_model.model(**query_enc)
                    query_emb = mean_pooling(query_outputs, query_enc['attention_mask'])
                    query_emb = F.normalize(query_emb, p=2, dim=1)
                    
                    code_outputs = large_model.model(**code_enc)
                    code_emb = mean_pooling(code_outputs, code_enc['attention_mask'])
                    code_emb = F.normalize(code_emb, p=2, dim=1)
                    
                    loss = 1.0 - torch.nn.CosineSimilarity(dim=1)(query_emb, code_emb).mean()
                
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / len(eval_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Evaluation Loss: {avg_eval_loss:.4f}")
    
    torch.save(small_model.model.state_dict(), args.save_path)
    print(f"Small Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune small Embedder model with LoRA")
    parser.add_argument("--train_data", type=str, help="Path to the training data", default="data/python_dataset/train")
    parser.add_argument("--eval_data", type=str, help="Path to the evaluation data", default="data/python_dataset/valid")
    parser.add_argument("--small_model_name", type=str, default="jinaai/jina-embeddings-v2-small-en", help="Small model for query embedding")
    parser.add_argument("--large_model_name", type=str, default="jinaai/jina-embeddings-v2-base-code", help="Large model for code embedding")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cpu or cuda)")
    parser.add_argument("--save_path", type=str, default="models/small_Embedder_finetune_lora.pth", help="Path to save the trained model")
    
    args = parser.parse_args()
    fine_tune_with_lora(args)
