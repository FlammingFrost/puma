import argparse
import torch
import sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.database import Database
from dataset_python import PrecomputedEmbeddingsDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test embedding quality")
    parser.add_argument("--query_embeddings_path", type=str, required=True, help="Path to the query embeddings")
    parser.add_argument("--code_embeddings_path", type=str, required=True, help="Path to the code embeddings")
    parser.add_argument("--vector_store_path", type=str, required=True, help="Path to the vector store")
    parser.add_argument("--top_k", type=int, default=5, help="Top K value for recall calculation")
    parser.add_argument("--test_name", type=str, default='dummy', help="Name of the test")

    args = parser.parse_args()
    from RAG_recall_test import test_embedding_quality
    test_embedding_quality(args.query_embeddings_path, args.code_embeddings_path, args.vector_store_path, args.top_k, args.test_name)