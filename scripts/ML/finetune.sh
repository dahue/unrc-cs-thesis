#!/bin/bash

# Load .env file into the current shell session
set -o allexport
source .env
set +o allexport

# Now use the variables
echo "Project root is: $ROOT_PATH"

export MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"

mlx_lm.lora \
    --model $MODEL \
    --train \
    --data $ROOT_PATH/data/training/nl2SQL \
    --adapter-path $ROOT_PATH/experiments/models/Llama-3.2-3B-Instruct-4bit/nl2SQL/adapters \
    --iters 500 \
    --batch-size 2 \
    --num-layers 16
