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
from train.negative_example_generator import NegativeExampleGenerator

from tools.logger import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class SentenceBERTFineTuner:
    def __init__(self, model_name="jinaai/jina-embeddings-v2-base-code"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, trust_remote_code=True).to(self.device)

    def train(self, train_data, use_hard_negatives=True, train_epochs=3, batch_size=8, save_path="models/fine_tuned_embedder"):
        """
        Fine-tune the SentenceTransformer model using contrastive learning.
        """
        # Extract positive examples from train_data
        queries = [item["query"] for item in train_data]
        positives = [item["positive"] for item in train_data]

        # Generate negative examples
        negative_generator = NegativeExampleGenerator()
        if use_hard_negatives:
            negatives = negative_generator.get_semantic_hard_negatives(queries, positives, positives + queries, self.model)
        else:
            negatives = negative_generator.get_random_negative(queries, positives)

        # Convert data into a Hugging Face Dataset (WITHOUT using InputExample)
        train_dataset = Dataset.from_dict({
            "anchor": queries,  # Renamed to match SentenceTransformer's triplet format
            "positive": positives,
            "negative": negatives
        })

        # Define Loss functions
        # https://www.sbert.net/docs/sentence_transformer/loss_overview.html
        loss = losses.MultipleNegativesRankingLoss(self.model)

        # Train model
        logger.info("Starting fine-tuning...")
        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir="models/",
            # Optional training parameters:
            num_train_epochs=train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            # bf16=False,  # Set to True if you have a GPU that supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            # # Optional tracking/debugging parameters:
            # eval_strategy="steps",
            # eval_steps=100,
            # save_strategy="steps",
            # save_steps=100,
            # save_total_limit=2,
            # logging_steps=100,
            # run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
        )
        
        # Create a trainer & train
        logger.info("Starting fine-tuning with contrastive learning...")
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            # eval_dataset=eval_data,
            loss=loss,
            # evaluator=dev_evaluator,
        )
        trainer.train()
        
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        logger.info(f"Fine-tuned model saved at: {save_path}")

# Example Usage
if __name__ == "__main__":
    train_data = [
        {"query": "What is AI?", "positive": "Artificial Intelligence is...", "negative": "The sky is blue."},
        # Add more samples...
    ]
    
    fine_tuner = SentenceBERTFineTuner()
    fine_tuner.train(train_data)
