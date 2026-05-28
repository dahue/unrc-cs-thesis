"""
test_prompt_gen.py — Preview rendered prompts from a prompt config + gold_dataset.

Usage:
  uv run python -m scripts.test_prompt_gen --config OpenText2SQL.json
  uv run python -m scripts.test_prompt_gen --config OpenText2SQL.json --source dev --difficulty easy --limit 3
"""

import os
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH not set. Add it to your .env file.")

DB = f"{ROOT_PATH}/database/OpenText2SQL.db"
PROMPTS_DIR = f"{ROOT_PATH}/data/prompt"


def main():
    parser = argparse.ArgumentParser(description="Preview rendered prompts from a prompt config.")
    parser.add_argument("--config", required=True, help="Prompt config filename, e.g. OpenText2SQL.json")
    parser.add_argument("--source", choices=["train", "dev", "test"], default="dev",
                        help="Dataset split to sample from (default: dev).")
    parser.add_argument("--difficulty", nargs="*", choices=["easy", "medium", "hard", "extra"],
                        help="Filter by one or more difficulty levels.")
    parser.add_argument("--limit", type=int, default=2,
                        help="Number of prompts to render and display (default: 2).")
    parser.add_argument("--top-k-few-shot", type=int, default=3,
                        help="Few-shot examples to retrieve per prompt (default: 3).")
    args = parser.parse_args()

    config_path = os.path.join(PROMPTS_DIR, args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    from scripts.util.llm import prompt_generation

    difficulty = args.difficulty[0] if args.difficulty and len(args.difficulty) == 1 else args.difficulty

    print(f"Config:     {args.config}")
    print(f"Source:     {args.source}")
    print(f"Difficulty: {difficulty or 'all'}")
    print(f"Limit:      {args.limit}")
    print()

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

    sep = "─" * 70
    for i, rec in enumerate(records, 1):
        print(f"{'═' * 70}")
        print(f"  Record {i}/{len(records)}  |  db_id: {rec['db_id']}  |  difficulty: {rec['difficulty']}  |  source: {rec['source']}")
        print(f"{'═' * 70}")
        print(rec["prompt"])
        print(f"{sep}")
        print(f"  Gold SQL: {rec['query']}")
        print()


if __name__ == "__main__":
    main()
