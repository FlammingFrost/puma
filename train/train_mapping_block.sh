#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <mapping_block> <train_query_emb_path> <train_code_emb_path> <eval_query_emb_path> <eval_code_emb_path> <residual>"
    exit 1
fi

MAPPING_BLOCK=$1
TRAIN_QUERY_EMB_PATH=$2
TRAIN_CODE_EMB_PATH=$3
EVAL_QUERY_EMB_PATH=$4
EVAL_CODE_EMB_PATH=$5
RESIDUAL=$6

python train/train.py \
    --mapping_block $MAPPING_BLOCK \
    --train_data data/python_dataset/train \
    --eval_data data/python_dataset/valid \
    --tokenizer_name jinaai/jina-embeddings-v2-base-code \
    --base_model_name jinaai/jina-embeddings-v2-base-code \
    --max_len 512 \
    --epochs 200 \
    --batch_size 4096 \
    --learning_rate 2e-5 \
    --device cuda \
    --train_query_emb_path $TRAIN_QUERY_EMB_PATH \
    --train_code_emb_path $TRAIN_CODE_EMB_PATH \
    --eval_query_emb_path $EVAL_QUERY_EMB_PATH \
    --eval_code_emb_path $EVAL_CODE_EMB_PATH \
    --residual $RESIDUAL