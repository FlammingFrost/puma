import torch
import shutil
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.dataset_python import PythonDataset
from retrieval.database import Database

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
EVAL_DATASET_PATH = "data/python_dataset/valid"
TEST_DATASET_PATH = "data/python_dataset/test"
TEMP_VECTORSTORE_PATH_EVAL = "train/test_rag/temp_store_eval_fp16"
TEMP_VECTORSTORE_PATH_TEST = "train/test_rag/temp_store_test_fp16"

def eval():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model_fp16 = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
    model_fp16.eval()
    
    if os.path.exists("eval_embeddings_query_fp16.pt") and os.path.exists("eval_embeddings_code_fp16.pt"):
        queries = torch.load("eval_embeddings_query_fp16.pt", weights_only=False)
        codes = torch.load("eval_embeddings_code_fp16.pt", weights_only=False)
    else:
        eval_dataset = PythonDataset(EVAL_DATASET_PATH, tokenizer, max_len=512)  
        queries, codes = [], []  
        for query_enc, code_enc in tqdm(eval_dataset):
            query_enc = {key: value.to("cuda") for key, value in query_enc.items()}
            code_enc = {key: value.to("cuda") for key, value in code_enc.items()}
            
            
            with torch.no_grad():
                query_emb = model_fp16(**query_enc).pooler_output
                code_emb = model_fp16(**code_enc).pooler_output
                queries.append(query_emb)
                codes.append(code_emb)
                assert type(query_emb) == torch.Tensor, "Query embeddings should be a PyTorch tensor"
        
        # save the embeddings
        torch.save(queries, "eval_embeddings_query_fp16.pt")
        torch.save(codes, "eval_embeddings_code_fp16.pt")

    # create a new vector-database for evaluation set
    # delete the temp vector store
    TEMP_VECTORSTORE_PATH = TEMP_VECTORSTORE_PATH_EVAL
    if os.path.exists(TEMP_VECTORSTORE_PATH):
        for file_name in os.listdir(TEMP_VECTORSTORE_PATH):
            file_path = os.path.join(TEMP_VECTORSTORE_PATH, file_name)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        os.rmdir(TEMP_VECTORSTORE_PATH)
    
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
        with open("quantization/eval_results.txt", "a") as f:
            f.write(f"Top1 Recall: {top1_recall:.4f}\n")
            f.write(f"Top5 Recall: {top5_recall:.4f}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")

def test():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model_fp16 = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
    model_fp16.eval()
    
    if os.path.exists("test_embeddings_query_fp16.pt") and os.path.exists("test_embeddings_code_fp16.pt"):
        queries = torch.load("test_embeddings_query_fp16.pt", weights_only=False)
        codes = torch.load("test_embeddings_code_fp16.pt", weights_only=False)
    else:
        test_dataset = PythonDataset(TEST_DATASET_PATH, tokenizer, max_len=512)
        queries, codes = [], []
        for query_enc, code_enc in tqdm(test_dataset):
            query_enc = {key: value.to("cuda") for key, value in query_enc.items()}
            code_enc = {key: value.to("cuda") for key, value in code_enc.items()}
            
            with torch.no_grad():
                query_emb = model_fp16(**query_enc).pooler_output
                code_emb = model_fp16(**code_enc).pooler_output
                queries.append(query_emb)
                codes.append(code_emb)
                assert type(query_emb) == torch.Tensor, "Query embeddings should be a PyTorch tensor"
        
        # save the embeddings
        torch.save(queries, "test_embeddings_query_fp16.pt")
        torch.save(codes, "test_embeddings_code_fp16.pt")

    # create a new vector-database for evaluation set
    TEMP_VECTORSTORE_PATH = TEMP_VECTORSTORE_PATH_TEST
    if os.path.exists(TEMP_VECTORSTORE_PATH):
        for file_name in os.listdir(TEMP_VECTORSTORE_PATH):
            file_path = os.path.join(TEMP_VECTORSTORE_PATH, file_name)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        os.rmdir(TEMP_VECTORSTORE_PATH)
    
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
        with open("quantization/test_results_fp16.txt", "a") as f:
            f.write(f"Top1 Recall: {top1_recall:.4f}\n")
            f.write(f"Top5 Recall: {top5_recall:.4f}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")
