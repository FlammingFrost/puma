#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mapping_block> <residual> <ffn_nblocks>"
    exit 1
fi

MAPPING_BLOCK=$1
RESIDUAL=$2

python train/train.py \
    --mapping_block $MAPPING_BLOCK \
    --train_data data/python_dataset/train \
    --eval_data data/python_dataset/valid \
    --tokenizer_name jinaai/jina-embeddings-v2-base-code \
    --base_model_name jinaai/jina-embeddings-v2-base-code \
    --max_len 512 \
    --epochs 100 \
    --batch_size 1024 \
    --learning_rate 2e-5 \
    --device cuda \
    --train_query_emb_path models/embeddings/small_train_embeddings_query.pt \
    --train_code_emb_path models/embeddings/train_embeddings_query.pt \
    --eval_query_emb_path models/embeddings/small_eval_embeddings_query.pt \
    --eval_code_emb_path models/embeddings/eval_embeddings_code.pt \
    --residual $RESIDUAL \
    --save_path models/${MAPPING_BLOCK}${3}_residual_${RESIDUAL}_on_queries.pth \
    --ffn_nblocks $3