import sys
import os
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset
# from torch.utils.data import DataLoader
# from train.negative_example_generator import NegativeExampleGenerator
from dataset_python import PythonDataset
from tools.logger import logger


def finetune_embedder(model_name="jinaai/jina-embeddings-v2-base-code", train_dataset=None, use_hard_negatives=True, train_epochs=3, batch_size=8, save_path="models/fine_tuned_embedder"):
    """
    Fine-tune the SentenceTransformer model using contrastive learning.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, trust_remote_code=True).to(device)

    # Define Loss functions
    loss = losses.MultipleNegativesRankingLoss(model)

    # Convert train_dataset to Dataset object if it's not already
    if isinstance(train_dataset, PythonDataset):
        train_dataset = train_dataset.to_hf_dataset()

    # Train model
    logger.info("Starting fine-tuning...")
    args = SentenceTransformerTrainingArguments(
        output_dir="models/",
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    
    # Create a trainer & train
    logger.info("Starting fine-tuning with contrastive learning...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    logger.info(f"Fine-tuned model saved at: {save_path}")

def finetune_embedder(model_name="jinaai/jina-embeddings-v2-base-code", train_dataset=None, use_hard_negatives=True, train_epochs=3, batch_size=8, save_path="models/fine_tuned_embedder"):
    """
    Fine-tune the SentenceTransformer model using contrastive learning.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, trust_remote_code=True).to(device)

    # Define Loss functions
    loss = losses.MultipleNegativesRankingLoss(model)

    # Convert train_dataset to Dataset object if it's not already
    if isinstance(train_dataset, PythonDataset):
        train_dataset = train_dataset.to_hf_dataset()

    # Train model
    logger.info("Starting fine-tuning...")
    args = SentenceTransformerTrainingArguments(
        output_dir="models/",
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    
    # Create a trainer & train
    logger.info("Starting fine-tuning with contrastive learning...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    logger.info(f"Fine-tuned model saved at: {save_path}")


# Example Usage
if __name__ == "__main__": 
    train_dataset = PythonDataset("tests/data/python", seed=224, negative_triplets=True, negative_precomputed=True)
    fine_tune_sentence_bert(train_dataset=train_dataset)
