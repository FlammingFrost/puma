import argparse
from transformers import AutoTokenizer
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
                
from retrieval.embedder import Embedder, MLPEmbedder
from dataset_python import PythonDataset
from trainer import MLPEmbedderTrainer, compute_and_save_embeddings, PrecomputedEmbeddingsDataset


def main(args):
    print('args:', args)
    if args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    train_dataset = PythonDataset(args.train_data, tokenizer, args.max_len)
    eval_dataset = PythonDataset(args.eval_data, tokenizer, args.max_len)
    
    base_model = Embedder(model_name=args.base_model_name)
    
    # Compute and save embeddings
    compute_and_save_embeddings(train_dataset, base_model, batch_size=args.batch_size, save_path="train_embeddings")
    compute_and_save_embeddings(eval_dataset, base_model, batch_size=args.batch_size, save_path="eval_embeddings")
    
    # Load precomputed embeddings
    train_dataset = PrecomputedEmbeddingsDataset("train_embeddings_query.pt", "train_embeddings_code.pt")
    eval_dataset = PrecomputedEmbeddingsDataset("eval_embeddings_query.pt", "eval_embeddings_code.pt")
    
    embedder = MLPEmbedder(input_dim=768, hidden_dim=512, output_dim=768, base_model=base_model)
    
    trainer = MLPEmbedderTrainer(
        model=embedder,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    trainer.train()
    trainer.save_trained_model(path=args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLPEmbedder model")
    parser.add_argument("--train_data", type=str,  help="Path to the training data", default="data/python_dataset/train")
    parser.add_argument("--eval_data", type=str,  help="Path to the evaluation data", default="data/python_dataset/valid")
    parser.add_argument("--tokenizer_name", type=str, default="jinaai/jina-embeddings-v2-base-code", help="Tokenizer name")
    parser.add_argument("--base_model_name", type=str, default="jinaai/jina-embeddings-v2-base-code", help="Base model name")
    parser.add_argument("--max_len", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="gpu", help="Device to train on (cpu or cuda)")
    parser.add_argument("--save_path", type=str, default="models/MLPEmbedder_finetune.pth", help="Path to save the trained model")
    
    args = parser.parse_args()
    main(args)
