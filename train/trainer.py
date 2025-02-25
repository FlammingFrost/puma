import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



    
class MLPEmbedderTrainer:
    """
    Class for training the Query Transformer MLP.
    """
    def __init__(self, model, train_dataset, eval_dataset, learning_rate=2e-5, epochs=5, batch_size=8, device='cpu'):
        self.device = device
        self.model = model
        self.train_dataset = train_dataset
        if not hasattr(self.model, 'embedder'):
            raise AttributeError("The model does not have an 'embedder' method or attribute.")
        self.eval_dataset = eval_dataset
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        # self.sceduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

    def train(self):
        """Trains the Query Transformer using Cosine Similarity Loss."""
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        eval_loader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Print the number of trainable and frozen parameters
        self.model.to(self.device)
        self.model.train()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Frozen parameters: {frozen_params}")
        
        

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.epochs}"):
                # import pdb; pdb.set_trace()
                
                
                # Move the input tensors to the GPU
                query_enc, code_enc = batch
                query_input = {key: value.to(self.device) for key, value in query_enc.items()}
                code_input = {key: value.to(self.device) for key, value in code_enc.items()}
                
                # debug
                # assert
                
                # Forward pass  
                query_emb = self.model(query_input)
                code_emb = self.model.embedder(code_input)
                
                # Cosine Similarity Loss
                self.optimizer.zero_grad()
                loss = 1.0 - torch.nn.CosineSimilarity(dim=1)(query_emb, code_emb).mean()
                
                loss.backward()
                self.optimizer.step()
                
                
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_loss:.4f}")

            # Evaluation
            self.model.eval()
            eval_loss = 0
            with torch.no_grad():
                for batch in eval_loader:
                    query_enc, code_enc = batch
                    query_input = {key: value.to(self.device) for key, value in query_enc.items()}
                    code_input = {key: value.to(self.device) for key, value in code_enc.items()}
                    query_emb = self.model(query_input)
                    code_emb = self.model.embedder(code_input)
                    loss = 1.0 - torch.nn.CosineSimilarity(dim=1)(query_emb, code_emb).mean()
                    eval_loss += loss.item
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
    
    
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-code")
    dataset = PythonDataset(input_folder, tokenizer, max_len)
    base_model = Embedder(model_name=base_model_name)
    embedder = MLPEmbedder(input_dim=768, hidden_dim=512, output_dim=768, base_model=base_model)
    
    trainer = MLPEmbedderTrainer(
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