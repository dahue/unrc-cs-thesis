#!/bin/bash

# Load .env file into the current shell session
set -o allexport
source .env
set +o allexport

# Now use the variables
echo "Project root is: $ROOT_PATH"

export MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"

mlx_lm.generate \
    --model $MODEL \
    --adapter-path $ROOT_PATH/models/Llama-3.2-3B-Instruct-4bit/nl2SQL/adapters \
    --prompt "[QUESTION]Show all artist names and the year joined who are not from United States.[/QUESTION]\n[SCHEMA]artist(Artist_ID, Name, Country, Year_Join, Age)\nexhibition(Exhibition_ID, Year, Theme, Artist_ID, Ticket_Price)\nexhibition_record(Exhibition_ID, Date, Attendance)[/SCHEMA]\n[NatSQL]" \
    --max-tokens 1024