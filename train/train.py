import argparse
from transformers import AutoTokenizer
import torch
# from torch.utils.data import Subset

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
                
from retrieval.embedder import Embedder, MLP, FFN
from dataset_python import PythonDataset
from trainer import MappingBlockTrainer, compute_and_save_embeddings
from dataset_python import PrecomputedEmbeddingsDataset



def main(args):
    print('args:', args)
    if args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    
    
    # Load precomputed embeddings
    train_dataset = PrecomputedEmbeddingsDataset(f"{args.train_emb_path}_query.pt", f"{args.train_emb_path}_code.pt")
    eval_dataset = PrecomputedEmbeddingsDataset(f"{args.eval_emb_path}_query.pt", f"{args.eval_emb_path}_code.pt")
    
    if args.mapping_block == "MLP":
        embedder = MLP(input_dim=768, hidden_dim=512, output_dim=768, residual=args.residual)
    elif args.mapping_block == "FFN":
        embedder = FFN(input_dim=768, hidden_dim=512, output_dim=768, residual=args.residual)
    else:
        raise ValueError("Mapping block must be either 'MLP' or 'FFN'")
    
    trainer = MappingBlockTrainer(
        model=embedder,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    trainer.train()
    # add time to save path
    import time
    save_path = "models/"+args.mapping_block+"_"+str(time.time())+".pth"
    trainer.save_trained_model(path=save_path)
    print(f"Model saved to {save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLPEmbedder model")
    parser.add_argument("--task", type=str, default="mapping", help="Task to perform (mapping or retrieval)")
    parser.add_argument("--embed_subset", type=str, default="both", help="Subset of embeddings to use (query or code or both)")
    parser.add_argument("--mapping_block", type=str, default="MLP", help="Mapping block to use (MLP or FFN)")
    parser.add_argument("--residual", type=bool, default=True, help="Whether to use residual connections in the mapping block")
    parser.add_argument("--emb_name", type=str, default="default", help="Name of the embeddings saved")
    
    parser.add_argument("--train_data", type=str,  help="Path to the training data", default="data/python_dataset/train")
    parser.add_argument("--eval_data", type=str,  help="Path to the evaluation data", default="data/python_dataset/valid")
    parser.add_argument("--tokenizer_name", type=str, default="jinaai/jina-embeddings-v2-base-code", help="Tokenizer name")
    parser.add_argument("--base_model_name", type=str, default="jinaai/jina-embeddings-v2-base-code", help="Base model name")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cpu or cuda)")
    # parser.add_argument("--save_path", type=str, default="None", help="Path to save the trained model")
    parser.add_argument("--train_emb_path", type=str, default="models/embeddings/train_embeddings", help="Path to save the training embeddings")
    parser.add_argument("--eval_emb_path", type=str, default="models/embeddings/eval_embeddings", help="Path to save the evaluation embeddings")
    # parser.add_argument("--subset_size", type=int, default=10, help="Number of examples to load for quick experimentation")
    
    args = parser.parse_args()
    if args.task == "mapping":
        main(args)
    elif args.task == "embedding":
        
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        train_dataset = PythonDataset(args.train_data, tokenizer, args.max_len)
        eval_dataset = PythonDataset(args.eval_data, tokenizer, args.max_len)
        
        base_model = Embedder(model_name=args.base_model_name)
        if args.embed_subset == "query":
            subset = ['query']
        elif args.embed_subset == "code":
            subset = ['code']
        elif args.embed_subset == "both":
            subset = ['query', 'code']
        else:
            raise ValueError("Subset must be either 'query' or 'code' or 'both'")
        
        train_save_path = args.train_emb_path + "_" + args.emb_name
        eval_save_path = args.eval_emb_path + "_" + args.emb_name
        
        compute_and_save_embeddings(train_dataset, base_model, batch_size=args.batch_size, save_path=train_save_path, subset=subset)
        compute_and_save_embeddings(eval_dataset, base_model, batch_size=args.batch_size, save_path=eval_save_path, subset=subset)
        # compute_and_save_embeddings(train_subset, base_model, batch_size=args.batch_size, save_path=args.train_emb_path)
        # compute_and_save_embeddings(eval_subset, base_model, batch_size=args.batch_size, save_path=args.eval_emb_path)
