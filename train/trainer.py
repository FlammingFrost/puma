import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.embedder import *
from train.dataset_python import PythonDataset

    
class MLPEmbedderTrainer:
    """
    Class for training the Query Transformer MLP.
    """
    def __init__(self, model, train_dataset, eval_dataset, learning_rate=2e-5, epochs=5, batch_size=8, device = 'cpu', **kwargs):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        # set kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.sceduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

    def train(self):
        """Trains the Query Transformer using Cosine Similarity Loss."""
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        eval_loader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        for epoch in range(self.epochs):
            total_loss = 0
            for query, code, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.epochs}"):
                self.optimizer.zero_grad()
                query_emb = self.model(query)
                code_emb = self.model.embedder(code)
                assert query_emb.shape == (self.batch_size, 768), f"Query embedding shape is {query_emb.shape}"
                code_emb = torch.tensor(code_emb).to(self.device)
                loss = 1 - torch.nn.CosineSimilarity(dim=1)(query_emb, code_emb).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_loss:.4f}")

            # Evaluation
            self.model.eval()
            eval_loss = 0
            with torch.no_grad():
                for query, code, _ in tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}/{self.epochs}"):
                    query_emb = self.model(query)
                    code_emb = self.model.embedder(code)
                    code_emb = torch.tensor(code_emb).to(query.device)
                    loss = 1 - torch.nn.CosineSimilarity()(query_emb, code_emb).mean()
                    eval_loss += loss.item()

            avg_eval_loss = eval_loss / len(eval_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Evaluation Loss: {avg_eval_loss:.4f}")
            self.model.train()

        return self.model

    def save_model(self, path="models/MLPEMbedder_finetune.pth"):
        """Saves the trained model."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--task", type=str, default="MLP_Embedder_ft")
    args = parser.parse_args()
    # Load the embedding model
    
    if args.task == "MLP_Embedder_ft":
        trainer = MLPEmbedderTrainer(
            model=MLPEmbedder(device='cpu'),
            train_dataset=PythonDataset("data/python_dataset"
                                        , negative_triplets=True, subset='valid', negative_precomputed=True),
            eval_dataset=PythonDataset("data/python_dataset", negative_triplets=True,
                                       negative_precomputed=True,
                                       subset='valid'),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        trainer.train()
        trainer.save_model()