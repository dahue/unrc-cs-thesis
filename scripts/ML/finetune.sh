#!/bin/bash

export ROOT_PATH="/Users/atissera/Developer/repos/unrc-cs-thesis"
export MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"

mlx_lm.lora \
    --model $MODEL \
    --train \
    --data $ROOT_PATH/data/training/nl2SQL \
    --adapter-path $ROOT_PATH/models/Llama-3.2-3B-Instruct-4bit/nl2SQL/adapters \
    --iters 500 \
    --batch-size 2 \
    --num-layers 16
