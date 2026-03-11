#!/bin/bash

# preSQL generation
python -m scripts.ML.predict \
    --presql \
    --models mlx-community/Qwen3-14B-4bit \
             mlx-community/phi-4-4bit \
             mlx-community/Ministral-3-14B-Instruct-2512-4bit \
             mlx-community/Ministral-8B-Instruct-2410-4bit \
             mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
             mlx-community/Olmo-3-7B-Instruct-4bit \
             mlx-community/Llama-3.2-3B-Instruct-4bit \
    --strategy nl2SQL \
    --template template_00 \
    --consistency-mode cross \
    --batch-size 5 \
    --max-tokens 512 \
    --input-file test

# finSQL generation
python -m scripts.ML.predict \
    --finsql \
    --models mlx-community/Qwen3-14B-4bit \
             mlx-community/phi-4-4bit \
             mlx-community/Ministral-3-14B-Instruct-2512-4bit \
             mlx-community/Ministral-8B-Instruct-2410-4bit \
             mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
             mlx-community/Olmo-3-7B-Instruct-4bit \
             mlx-community/Llama-3.2-3B-Instruct-4bit \
    --strategy nl2SQL \
    --template template_00 \
    --consistency-mode cross \
    --batch-size 5 \
    --max-tokens 512 \
    --presql-file data/predictions/nl2SQL/template_00/cross7models/test_predictions_presql_detailed.jsonl

python -m scripts.ML.benchmark \
    --model cross7models \
    --strategy nl2SQL \
    --template template_00 \
    --prediction-file test_predictions_cross_7models



## 
python -m scripts.ML.predict \
    --presql \
    --models mlx-community/Qwen3-14B-4bit \
    --strategy nl2SQL \
    --template 00_baseline \
    --input-file test \
    --batch-size 5


python -m scripts.ML.benchmark \
    --model mlx-community/Qwen3-14B-4bit \
    --strategy nl2SQL \
    --template 00_baseline \
    --prediction-file test_predictions_presql