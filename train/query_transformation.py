import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedding.embedder import Embedder
from train.dataset_python import PythonDataset

class QueryTransformer(nn.Module):
    """
    Class for defining the MLP tranformation layer to map query embedding closer to code embedding.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768):
        super(QueryTransformer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, query_embedding):
        return self.fc2(self.relu(self.fc1(query_embedding)))
    
class EmbeddingDataset(Dataset):
    """
    Convert PythonDataset into PyTorch Dataset.
    """
    def __init__(self, dataset, embedder):
        self.embedder = embedder
        self.dataset = dataset.data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]
        query_embedding = self.embedder.encode(record["anchor"], convert_to_tensor=True)
        code_embedding = self.embedder.encode(record["positive"], convert_to_tensor=True)
        return query_embedding, code_embedding
    
class QueryTransformerTrainer:
    """
    Class for training the Query Transformer MLP.
    """
    def __init__(self, model, embedder, dataset, batch_size=4, lr=1e-4, epochs=5):
        self.model = model
        self.embedder = embedder
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.loss_fn = nn.CosineEmbeddingLoss(margin=0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        """Trains the Query Transformer using Cosine Similarity Loss."""
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for query_emb, code_emb in self.dataloader:
                self.optimizer.zero_grad()
                
                # Transform the query embeddings
                transformed_query_emb = self.model(query_emb)
                
                # Labels: 1 since query and code should be similar
                labels = torch.ones(query_emb.shape[0]).to(query_emb.device)
                
                # Compute loss
                loss = self.loss_fn(transformed_query_emb, code_emb, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")

        print("Training complete!")
        return self.model

    def save_model(self, path="query_transformer.pth"):
        """Saves the trained model."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    # Load the embedding model
    embedder = Embedder(base_model="jinaai/jina-embeddings-v2-base-code")
    # embedder = Embedder(fine_tuned_model="models/fine_tuned_embedder")
    
    # Load dataset
    folder_path = "tests/data/python"
    train_dataset = PythonDataset(folder_path)
    
    # Convert data to embeddings
    embedding_dataset = EmbeddingDataset(train_dataset, embedder)

    # Initialize Query Transformer
    query_transformer = QueryTransformer()

    # Train the model
    trainer = QueryTransformerTrainer(query_transformer, embedder, embedding_dataset, epochs=5)
    trained_model = trainer.train()
    trainer.save_model()