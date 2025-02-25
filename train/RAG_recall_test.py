import torch
from retrieval.database import Database
from train.trainer import PrecomputedEmbeddingsDataset

def test_embedding_quality(query_embeddings_path, code_embeddings_path, vector_store_path, top_k=5, test_name='dummy'):
    # Step 1: Initialize a chromadb database for vector retrieval
    db = Database(vector_store_path)
    
    # Step 2: Initialize a dataset using PrecomputedEmbeddingsDataset
    dataset = PrecomputedEmbeddingsDataset(query_embeddings_path, code_embeddings_path)
    
    # Step 3: Load code embeddings into vectorbase with unique ids
    for idx, (query_emb, code_emb) in enumerate(dataset):
        metadata = {}
        code_id = f'code_{idx}'
        db.insert_embedding(code_emb.view(-1).tolist(), code_id, metadata)
    
    # Step 4: Evaluate retrieval
    correct_top1 = 0
    correct_topk = 0
    total = len(dataset)
    
    for idx, (query_emb, code_emb) in enumerate(dataset):
        query_embedding = query_emb.view(-1).tolist()
        retrievals = db.retrieve(query_embedding)
        retrieved_file_paths = [row['file_path'] for row in retrievals["metadatas"]]
        
        if f'code_{idx}' in retrieved_file_paths[:1]:
            correct_top1 += 1
        if f'code_{idx}' in retrieved_file_paths[:top_k]: # actually it only contains topk retrieved file paths
            correct_topk += 1
    
    # Step 5: Record the Top1 and Topk recall
    top1_recall = correct_top1 / total
    topk_recall = correct_topk / total
    
    print(f"Top1 Recall: {top1_recall:.4f}")
    print(f"Top{top_k} Recall: {topk_recall:.4f}")

if __name__ == "__main__":
    query_embeddings_path = "embeddings_query.pt" # TODO: Replace with actual path
    code_embeddings_path = "embeddings_code.pt"
    vector_store_path = "tests/rag_db/chroma_db"
    test_embedding_quality(query_embeddings_path, code_embeddings_path, vector_store_path, top_k=5, test_name='MLPEmbedder')
