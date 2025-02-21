import torch
import faiss
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset
# from data_processing.negative_example_generator import NegativeExampleGenerator

class Embedder:
    def __init__(self, base_model="jinaai/jina-embeddings-v2-base-code", fine_tuned_model=None):
        """
        Initialize the embedder with a pre-trained model or a fine-tuned model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            fine_tuned_model if fine_tuned_model else base_model, 
            trust_remote_code=True).to(self.device)
        self.index = None

    def encode(self, texts):
        """
        Encode a list of code snippets or queries.
        """
        return np.array(self.model.encode(texts, convert_to_numpy=True))

    def train_embedder(self, train_data, use_hard_negatives=True, train_epochs=3, batch_size=8, save_path="models/fine_tuned_jina_embeddings"):
        """
        Fine-tune the embedder using contrastive learning.
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

        # 3. Convert data into a Hugging Face Dataset (WITHOUT using InputExample)
        train_dataset = Dataset.from_dict({
            "anchor": queries,  # Renamed to match SentenceTransformer's triplet format
            "positive": positives,
            "negative": negatives
        })

        
        # Loss functions:
        # https://www.sbert.net/docs/sentence_transformer/loss_overview.html
        loss = losses.MultipleNegativesRankingLoss(self.model)
        
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
        print("Starting fine-tuning with contrastive learning...")
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            # eval_dataset=eval_data,
            loss=loss,
            # evaluator=dev_evaluator,
        )
        trainer.train()

        self.model.save_pretrained(save_path)
        print(f"Fine-tuned model saved at: {save_path}")

    def load_fine_tuned_model(self, model_path="fine_tuned_jina_embeddings"):
        """
        Load a fine-tuned embedding model.
        """
        self.model = SentenceTransformer(model_path).to(self.device)
        print(f"Loaded fine-tuned model from {model_path}")

    def build_faiss_index(self, code_snippets):
        """
        Build a FAISS index for fast retrieval of code snippets.
        """
        code_embeddings = self.encode(code_snippets)
        self.index = faiss.IndexFlatL2(code_embeddings.shape[1])
        self.index.add(code_embeddings)
        self.snippets = code_snippets
        print("FAISS index built successfully.")

    def retrieve_similar_code(self, query, top_k=2):
        """
        Retrieve top_k similar code snippets given a query.
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call `build_faiss_index()` first.")

        query_embedding = self.encode([query])
        D, I = self.index.search(query_embedding, k=top_k)

        return [self.snippets[idx] for idx in I[0]]
    

def test_encoding():
    embedder = Embedder()
    texts = ["def add(a, b): return a + b", "def subtract(a, b): return a - b"]
    embeddings = embedder.encode(texts)
    print(embeddings.shape)
    print(embeddings)
    
    assert isinstance(embeddings, np.ndarray), "Encoding should return a NumPy array"
    assert embeddings.shape[0] == len(texts), "Embeddings should match the number of input texts"
    assert embeddings.shape[1] > 0, "Embeddings should have a valid dimension"

    print("Encoding test passed!")
    
if __name__ == "__main__":
    test_encoding()
