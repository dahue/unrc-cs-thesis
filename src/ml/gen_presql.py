"""
gen_presql.py — Generate preSQL predictions for a dataset split and write results to JSONL.

Usage:
  uv run python -m src.ml.gen_presql --config OpenText2SQL.json --model Llama-3.2-3B-Instruct-4bit
  uv run python -m src.ml.gen_presql \
      --config OpenText2SQL.json \
      --model Llama-3.2-3B-Instruct-4bit \
      --source test \
      --difficulty medium \
      --limit 100 \
      --batch-size 2

Output:
  experiment/<YYYY-MM-DD_HH-MM-SS>/presql.jsonl
  Each line: {"prompt": "...", "presql": "...", "question": "...", "db_id": "...",
              "source": "...", "difficulty": "...", "gold_sql": "...",
              "simplified_ddl": "...", "foreign_keys": "...", "cell_values": "...", "few_shot": [...],
              "model": "...", "config": "..."}
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH not set. Add it to your .env file.")

DB = f"{ROOT_PATH}/database/OpenText2SQL.db"
PROMPTS_DIR = f"{ROOT_PATH}/config/prompt"
EXPERIMENTS_DIR = f"{ROOT_PATH}/experiment"


def main():
    parser = argparse.ArgumentParser(description="Generate preSQL predictions from a prompt config + gold_dataset.")

    # Prompt generation params (mirrors test_prompt_gen.py)
    parser.add_argument("--config", required=True,
                        help="Prompt config filename, e.g. OpenText2SQL.json")
    parser.add_argument("--source", choices=["train", "dev", "test"], default="dev",
                        help="Dataset split to use (default: dev).")
    parser.add_argument("--difficulty", nargs="*", choices=["easy", "medium", "hard", "extra"],
                        help="Filter by one or more difficulty levels.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of records to process.")
    parser.add_argument("--top-k-few-shot", type=int, default=3,
                        help="Few-shot examples to retrieve per prompt (default: 3).")

    # Inference params (mirrors test_inference.py)
    parser.add_argument("--model", required=True,
                        help="Model key (from models.json) or full HuggingFace path.")
    parser.add_argument("--adapter-path", default=None,
                        help="Path to LoRA adapter directory (required for :fine-tuned models).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Prompts per inference batch (default: 1).")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate per prompt (default: 512).")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to write presql.jsonl into. "
                             "Defaults to experiment/<YYYY-MM-DD_HH-MM-SS>/.")

    args = parser.parse_args()

    config_path = os.path.join(PROMPTS_DIR, args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    difficulty = args.difficulty[0] if args.difficulty and len(args.difficulty) == 1 else args.difficulty

    # ── Step 1: generate prompts ──────────────────────────────────────────────
    from src.util.llm import prompt_generation, infer

    print(f"Generating prompts from {args.config} ({args.source}, difficulty={difficulty or 'all'}, limit={args.limit})...")
    records = prompt_generation(
        config=config,
        db_path=DB,
        source=args.source,
        difficulty=difficulty,
        limit=args.limit,
        top_k_few_shot=args.top_k_few_shot,
    )

    if not records:
        print("No records found for the given filters. Exiting.")
        return

    print(f"Generated {len(records)} prompts.")

    # ── Step 2: run inference ─────────────────────────────────────────────────
    prompts = [rec["prompt"] for rec in records]

    print(f"Running inference with model '{args.model}'...")
    sql_results = infer(
        model=args.model,
        prompts=prompts,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        adapter_path=args.adapter_path,
    )

    # ── Step 3: write JSONL ───────────────────────────────────────────────────
    if args.out_dir:
        out_dir = args.out_dir
    else:
        experiment_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "presql.jsonl")

    from src.util.llm import resolve_model
    resolved_model = resolve_model(args.model.partition(":")[0])

    with open(out_path, "w", encoding="utf-8") as f:
        for rec, presql in zip(records, sql_results):
            line = {
                "prompt":         " ".join(rec["prompt"].split()),  # collapse to one line
                "presql":         presql,
                "question":       rec["question"],
                "db_id":          rec["db_id"],
                "source":         rec["source"],
                "difficulty":     rec["difficulty"],
                "gold_sql":       rec["query"],
                "simplified_ddl": rec["simplified_ddl"],
                "foreign_keys":   rec["foreign_keys"],
                "cell_values":    rec["cell_values"],
                "few_shot":       rec["few_shot"],
                "model":          resolved_model,
                "config":         args.config,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"✓ {len(records)} records written to {out_path}")


if __name__ == "__main__":
    main()
