#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mapping_block> <train_emb_path> <eval_emb_path>"
    exit 1
fi

MAPPING_BLOCK=$1
TRAIN_EMB_PATH=$2
EVAL_EMB_PATH=$3

python train.py \
    --mapping_block $MAPPING_BLOCK \
    --train_data data/python_dataset/train \
    --eval_data data/python_dataset/valid \
    --tokenizer_name jinaai/jina-embeddings-v2-base-code \
    --base_model_name jinaai/jina-embeddings-v2-base-code \
    --max_len 512 \
    --epochs 200 \
    --batch_size 512 \
    --learning_rate 2e-5 \
    --device cuda \
    --save_path models/MLP.pth \
    --train_emb_path $TRAIN_EMB_PATH \
    --eval_emb_path $EVAL_EMB_PATH