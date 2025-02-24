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
from retrieval.embedder import MLPEmbedder

    
class MLPEmbedderTrainer:
    """
    Class for training the Query Transformer MLP.
    """
    def __init__(self, model, train_dataset, eval_dataset, learning_rate=2e-5, epochs=5, batch_size=8):
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
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        eval_loader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        for epoch in range(self.epochs):
            total_loss = 0
            for query_token, code_token in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.epochs}"):
                self.optimizer.zero_grad()
                token_embedding = self.model(query_token)
                code_embedding = self.model.embedder(code_token)
                loss = 1-nn.functional.cosine_similarity(token_embedding, code_embedding).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_loss:.4f}")

            # Evaluation
            self.model.eval()
            eval_loss = 0
            with torch.no_grad():
                for query_token, code_token in tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}/{self.epochs}"):
                    token_embedding = self.model(query_token)
                    eval_loss += loss.item()
                    loss = 1-nn.functional.cosine_similarity(token_embedding, code_embedding).mean()
                    eval_loss += loss.item()
            avg_eval_loss = eval_loss / len(eval_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Evaluation Loss: {avg_eval_loss:.4f}")
            self.model.train()

        return self.model

    def save_trained_model(self, path="models/MLPEMbedder_finetune.pth"):
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
        from train.dataset_python import PythonDataset
        base_model = "jinaai/jina-embeddings-v2-base-code"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        trainer = MLPEmbedderTrainer(
            model=MLPEmbedder(base_model=base_model),
            train_dataset=PythonDataset("data/python_dataset", tokenizer=tokenizer, subset="valid"),
            eval_dataset=PythonDataset("data/python_dataset", tokenizer=tokenizer, subset="valid"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        trainer.train()
        trainer.save_trained_model()