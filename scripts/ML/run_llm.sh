#!/bin/bash

# Parse arguments for optional --adapter-path and --prompt
ADAPTER_PATH=""
PROMPT=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --adapter-path)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Load .env file into the current shell session
set -o allexport
source .env
set +o allexport

# Now use the variables
echo "Project root is: $ROOT_PATH"

export MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"

# Build adapter argument if provided
if [[ -n "$ADAPTER_PATH" ]]; then
  ADAPTER_ARG="--adapter-path $ADAPTER_PATH"
else
  ADAPTER_ARG=""
fi

# Build prompt argument if provided
if [[ -n "$PROMPT" ]]; then
  PROMPT_ARG="--prompt \"$PROMPT\""
else
  PROMPT_ARG='--prompt "[QUESTION]Show all artist names and the year joined who are not from United States.[/QUESTION]\n[SCHEMA]artist(Artist_ID, Name, Country, Year_Join, Age)\nexhibition(Exhibition_ID, Year, Theme, Artist_ID, Ticket_Price)\nexhibition_record(Exhibition_ID, Date, Attendance)[/SCHEMA]\n[NatSQL]"'
fi

mlx_lm.generate \
    --model $MODEL \
    $ADAPTER_ARG \
    $PROMPT_ARG \
    --max-tokens 1024
