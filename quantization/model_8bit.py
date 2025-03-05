import torch
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.dataset_python import PythonDataset
from retrieval.database import Database

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
EVAL_DATASET_PATH = "data/python_dataset/valid"
TEST_DATASET_PATH = "data/python_dataset/test"
TEMP_VECTORSTORE_PATH = "train/test_rag/temp_store"

def eval():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_8bit = AutoModel.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")
    model_8bit.eval()
    
    eval_dataset = PythonDataset(EVAL_DATASET_PATH, tokenizer, max_len=512)
    for query_enc, code_enc in tqdm(eval_dataset):
        query_enc = {key: value.to("cuda") for key, value in query_enc.items()}
        code_enc = {key: value.to("cuda") for key, value in code_enc.items()}
        
        queries, codes = [], []
        with torch.no_grad():
            query_emb = model_8bit(query_enc)
            code_emb = model_8bit(code_enc)
            queries.append(query_emb)
            codes.append(code_emb)
    
    # save the embeddings
    torch.save(queries, "eval_embeddings_query_8bit.pt")
    torch.save(codes, "eval_embeddings_code_8bit.pt")

    # create a new vector-database for evaluation set
    db = Database(TEMP_VECTORSTORE_PATH)
    for idx, (query_emb, code_emb) in tqdm(enumerate(zip(queries, codes)), total=len(queries), desc="Loading embeddings"):
        metadata = {}
        code_id = f'code_{idx}'
        db.insert_embedding(code_emb.view(-1).tolist(), code_id, metadata)
    correct1, correctk, total = 0, 0, 0
    for idx, (query_emb, code_emb) in tqdm(enumerate(zip(queries, codes)), total=len(queries), desc="Evaluating embeddings"):
        query_embedding = query_emb.view(-1).tolist()
        retrievals = db.retrieve_by_embedding(query_embedding, top_k=5)
        retrieved_ids = retrievals['ids'][0]
        if f'code_{idx}' in retrieved_ids[:1]:
            correct1 += 1
        if f'code_{idx}' in retrieved_ids[:5]:
            correctk += 1
        total += 1
    top1_recall = correct1 / total
    top5_recall = correctk / total
    print(f"Top1 Recall: {top1_recall:.4f}")
    print(f"Top5 Recall: {top5_recall:.4f}")
    
    try:
        with open("quantization/eval_results_8bit.txt", "a") as f:
            f.write(f"Top1 Recall: {top1_recall:.4f}\n")
            f.write(f"Top5 Recall: {top5_recall:.4f}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")
    # delete the temp vector store
    for file_name in os.listdir(TEMP_VECTORSTORE_PATH):
        os.remove(os.path.join(TEMP_VECTORSTORE_PATH, file_name))
    os.rmdir(TEMP_VECTORSTORE_PATH)
    
def test():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_8bit = AutoModel.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")
    model_8bit.eval()
    
    test_dataset = PythonDataset(TEST_DATASET_PATH, tokenizer, max_len=512)
    
    for query_enc, code_enc in tqdm(test_dataset):
        query_enc = {key: value.to("cuda") for key, value in query_enc.items()}
        code_enc = {key: value.to("cuda") for key, value in code_enc.items()}
        
        queries, codes = [], []
        with torch.no_grad():
            query_emb = model_8bit(query_enc)
            code_emb = model_8bit(code_enc)
            queries.append(query_emb)
            codes.append(code_emb)
    
    # save the embeddings
    torch.save(queries, "test_embeddings_query_8bit.pt")
    torch.save(codes, "test_embeddings_code_8bit.pt")

    # create a new vector-database for evaluation set
    db = Database(TEMP_VECTORSTORE_PATH)
    for idx, (query_emb, code_emb) in tqdm(enumerate(zip(queries, codes)), total=len(queries), desc="Loading embeddings"):
        metadata = {}
        code_id = f'code_{idx}'
        db.insert_embedding(code_emb.view(-1).tolist(), code_id, metadata)
    correct1, correctk, total = 0, 0, 0
    for idx, (query_emb, code_emb) in tqdm(enumerate(zip(queries, codes)), total=len(queries), desc="Evaluating embeddings"):
        query_embedding = query_emb.view(-1).tolist()
        retrievals = db.retrieve_by_embedding(query_embedding, top_k=5)
        retrieved_ids = retrievals['ids'][0]
        if f'code_{idx}' in retrieved_ids[:1]:
            correct1 += 1
        if f'code_{idx}' in retrieved_ids[:5]:
            correctk += 1
        total += 1
    top1_recall = correct1 / total
    top5_recall = correctk / total
    print(f"Top1 Recall: {top1_recall:.4f}")
    print(f"Top5 Recall: {top5_recall:.4f}")
    
    try:
        with open("quantization/test_results_8bit.txt", "a") as f:
            f.write(f"Top1 Recall: {top1_recall:.4f}\n")
            f.write(f"Top5 Recall: {top5_recall:.4f}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")
    # delete the temp vector store
    for file_name in os.listdir(TEMP_VECTORSTORE_PATH):
        os.remove(os.path.join(TEMP_VECTORSTORE_PATH, file_name))
    os.rmdir(TEMP_VECTORSTORE_PATH)
