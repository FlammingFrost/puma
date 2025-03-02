import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_and_save_embeddings(dataset, embedder, batch_size=8, save_path="models/embeddings", subset=['query', 'code']):
    """
    Compute and save embeddings for the specified subset of the dataset using the provided embedder.
    
    Args:
        dataset (Dataset): The dataset to encode.
        embedder (nn.Module): The embedder model.
        batch_size (int): Batch size for DataLoader.
        save_path (str): Path to save the embeddings.
        subset (list): List specifying which embeddings to compute ('query', 'code').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder.to(device).eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    query_embeddings, code_embeddings = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            query_enc, code_enc = batch
            if 'query' in subset:
                query_enc = {key: value.to(device) for key, value in query_enc.items()}
                query_emb = embedder(query_enc)
                query_embeddings.append(query_emb)
            if 'code' in subset:
                code_enc = {key: value.to(device) for key, value in code_enc.items()}
                code_emb = embedder(code_enc)
                code_embeddings.append(code_emb)

    if 'query' in subset:
        query_embeddings = torch.cat(query_embeddings).cpu()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(query_embeddings, f"{save_path}_query.pt")
        print(f"Query embeddings saved to {save_path}_query.pt")
    if 'code' in subset:
        code_embeddings = torch.cat(code_embeddings).cpu()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(code_embeddings, f"{save_path}_code.pt")
        print(f"Code embeddings saved to {save_path}_code.pt")
    


class MappingBlockTrainer:
    """
    Class for training the Query Transformer MLP.
    """
    def __init__(self, model, train_dataset, eval_dataset, learning_rate=2e-5, epochs=5, batch_size=8, device='cpu'):
        self.device = device
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

    def train(self):
        """Trains the Query Transformer using Cosine Similarity Loss."""
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        eval_loader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        # Print the number of trainable and frozen parameters
        self.model.to(self.device)
        self.model.train()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Frozen parameters: {frozen_params}")
        
        # Initialize AMP Gradient Scaler
        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.epochs}"):
               # Move the input tensors to the GPU
                query_emb, code_emb = batch
                query_emb = query_emb.to(self.device)
                code_emb = code_emb.to(self.device)
                
                # Forward pass  
                with torch.amp.autocast("cuda"):
                    output = self.model(query_emb)
                    loss = 1.0 - torch.nn.CosineSimilarity(dim=1)(output, code_emb).mean()
                    
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_loss:.4f}")

            # Evaluation
            self.model.eval()
            eval_loss = 0
            with torch.no_grad():
                for batch in eval_loader:
                    query_emb, code_emb = batch
                    query_emb = query_emb.to(self.device)
                    code_emb = code_emb.to(self.device)
                    
                    with torch.amp.autocast("cuda"):
                        output = self.model(query_emb)
                        loss = 1.0 - torch.nn.CosineSimilarity(dim=1)(output, code_emb).mean()
                    
                    eval_loss += loss.item()
            avg_eval_loss = eval_loss / len(eval_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Evaluation Loss: {avg_eval_loss:.4f}")
        
            self.model.train()
            
        return self.model   

    def save_trained_model(self, path="models/MLPEMbedder_finetune.pth"):
        """Saves the trained model."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")



def test_MLPEmbedder():
    from transformers import AutoTokenizer
    from retrieval.embedder import Embedder, MLPEmbedder
    from train.dataset_python import PythonDataset
    input_folder = "data/python_dataset/valid"
    tokenizer_name = "jinaai/jina-embeddings-v2-base-code"
    base_model_name = "jinaai/jina-embeddings-v2-base-code"
    max_len = 512
    
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = PythonDataset(input_folder, tokenizer, max_len)
    base_model = Embedder(model_name=base_model_name)
    embedder = MLPEmbedder(input_dim=768, hidden_dim=512, output_dim=768, base_model=base_model)
    
    trainer = MappingBlockTrainer(
        model=embedder,
        train_dataset=dataset,
        eval_dataset=dataset,
        epochs=1,
        batch_size=8,
        learning_rate=2e-5
    )
    # trainer.train()
    trainer.save_trained_model(path="models/MLPEmbedder_finetune_test.pth")

if __name__ == "__main__":
    test_MLPEmbedder()