#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <base_model_name>"
    exit 1
fi

BASE_MODEL_NAME=$1

python train.py \
    --task embedding \
    --base_model_name $BASE_MODEL_NAME \
    --tokenizer_name $BASE_MODEL_NAME \
    --batch_size 512 \
    --max_len 512
