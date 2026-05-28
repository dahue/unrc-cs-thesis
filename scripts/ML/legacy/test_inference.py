"""
test_inference.py — Run inference and print the generated SQL.

Two modes:
  1. Raw prompts: pass prompts as positional arguments.
  2. Config mode: use --config to build prompts from the database.

Usage:
  uv run python -m scripts.test_inference --model Llama-3.2-3B-Instruct-4bit "List all clubs"
  uv run python -m scripts.test_inference --model Llama-3.2-3B-Instruct-4bit "List all clubs" "Count students per course"
  uv run python -m scripts.test_inference --model Llama-3.2-3B-Instruct-4bit --config OpenText2SQL.json --source dev --limit 2
  uv run python -m scripts.test_inference --model Llama-3.2-3B-Instruct-4bit:fine-tuned --adapter-path /path/to/adapter --config OpenText2SQL.json

Model keys are short names from scripts/ML/models.json (e.g. "Llama-3.2-3B-Instruct-4bit").
Full HuggingFace paths are also accepted as a fallback.
"""

import os
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
DB = f"{ROOT_PATH}/database/OpenText2SQL.db" if ROOT_PATH else ""
PROMPTS_DIR = f"{ROOT_PATH}/data/prompt" if ROOT_PATH else ""


def main():
    parser = argparse.ArgumentParser(description="Test MLX model inference for SQL generation.")
    parser.add_argument("prompts", nargs="*", help="Raw natural-language prompts (mutually exclusive with --config).")
    parser.add_argument("--model", required=True, help="Model spec: 'model_name' or 'model_name:fine-tuned'.")
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA adapter directory (required when using :fine-tuned).")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts per batch (default: 1).")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate (default: 512).")

    # Config mode args
    parser.add_argument("--config", default=None, help="Prompt config filename, e.g. OpenText2SQL.json. Builds prompts from the database.")
    parser.add_argument("--source", choices=["train", "dev", "test"], default="dev",
                        help="Dataset split to use with --config (default: dev).")
    parser.add_argument("--difficulty", nargs="*", choices=["easy", "medium", "hard", "extra"],
                        help="Difficulty filter for --config mode.")
    parser.add_argument("--limit", type=int, default=2,
                        help="Max records to process in --config mode (default: 2).")
    parser.add_argument("--top-k-few-shot", type=int, default=3,
                        help="Few-shot examples per prompt in --config mode (default: 3).")
    args = parser.parse_args()

    if args.prompts and args.config:
        parser.error("Cannot combine positional prompts with --config. Use one or the other.")
    if not args.prompts and not args.config:
        parser.error("Provide either positional prompts or --config.")

    from scripts.util.llm import infer

    if args.config:
        if not ROOT_PATH:
            raise ValueError("ROOT_PATH not set. Add it to your .env file.")
        config_path = os.path.join(PROMPTS_DIR, args.config)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Prompt config not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        from scripts.util.llm import prompt_generation
        difficulty = args.difficulty[0] if args.difficulty and len(args.difficulty) == 1 else args.difficulty
        records = prompt_generation(
            config=config,
            db_path=DB,
            source=args.source,
            difficulty=difficulty,
            limit=args.limit,
            top_k_few_shot=args.top_k_few_shot,
        )
        if not records:
            print("No records found for the given filters.")
            return
        prompts = [r["prompt"] for r in records]
        questions = [r["question"] for r in records]
        gold_sqls = [r["query"] for r in records]
    else:
        prompts = args.prompts
        questions = args.prompts
        gold_sqls = [""] * len(args.prompts)

    sql_results = infer(
        model=args.model,
        prompts=prompts,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        adapter_path=args.adapter_path,
    )

    print()
    for question, gold, sql in zip(questions, gold_sqls, sql_results):
        print(f"Question: {question}")
        if gold:
            print(f"Gold SQL: {gold}")
        print(f"SQL:      {sql}")
        print()


if __name__ == "__main__":
    main()
