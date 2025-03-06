python train/evaluations/mlp_eval.py \
    --mapping_block $1 \
    --model_path $2 \
    --transformed_query_embeddings_path $3

python train/test_rag.py \
    --query_embeddings_path $3 \
    --code_embeddings_path models/embeddings/test_embeddings_code.pt \
    --vector_store_path $4 \
