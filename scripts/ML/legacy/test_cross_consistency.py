"""
test_cross_consistency.py — Run cross-consistency inference and preview results.

Usage:
  uv run python -m scripts.test_cross_consistency \\
      --config OpenText2SQL.json \\
      --models Llama-3.2-3B-Instruct-4bit Qwen3-14B-4bit \\
      --limit 3
  uv run python -m scripts.test_cross_consistency \\
      --config OpenText2SQL.json \\
      --models Llama-3.2-3B-Instruct-4bit Qwen3-14B-4bit \\
      --source dev --difficulty easy --limit 5 --batch-size 2
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


def _short(value, max_len=120):
    s = str(value)
    return s if len(s) <= max_len else s[:max_len] + "…"


def _model_label(full_name: str) -> str:
    """Strip org prefix for compact display."""
    if "/" in full_name:
        return full_name.split("/")[-1]
    return full_name


def _print_record(idx: int, total: int, rec: dict, result: dict):
    sep  = "─" * 70
    sep2 = "═" * 70

    models = result.get("models", [])
    all_sql = result.get("all_sql", [])
    winning_sql = result.get("sql") or "(none)"
    score = result.get("consistency_score", 0.0)
    gold = rec.get("query") or rec.get("gold_sql") or ""

    print(sep2)
    print(f"  Record {idx}/{total}  |  db_id: {rec.get('db_id')}  |  difficulty: {rec.get('difficulty')}  |  source: {rec.get('source')}")
    print(sep2)
    print(f"  Question : {rec.get('question')}")
    print(f"  Gold SQL : {gold}")
    print()

    # Per-model candidates
    print(f"  Candidates  ({len(models)} models)")
    for model, sql in zip(models, all_sql):
        is_winner = sql == winning_sql
        marker = "★" if is_winner else " "
        print(f"    {marker} [{_model_label(model)}]")
        print(f"        {_short(sql or '(empty)')}")

    # Winner + score
    agreed = round(score * len(models))
    print()
    print(f"  Winner  ({agreed}/{len(models)} models agreed, score={score:.2f})")
    print(f"    {_short(winning_sql)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Preview cross-consistency inference results.")
    parser.add_argument("--config", required=True,
                        help="Prompt config filename, e.g. OpenText2SQL.json")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Two or more model specs (short keys from models.json or full HuggingFace paths).")
    parser.add_argument("--source", choices=["train", "dev", "test"], default="dev",
                        help="Dataset split to use (default: dev).")
    parser.add_argument("--difficulty", nargs="*", choices=["easy", "medium", "hard", "extra"],
                        help="Filter by one or more difficulty levels.")
    parser.add_argument("--limit", type=int, default=2,
                        help="Max records to process (default: 2).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Prompts per inference batch (default: 1).")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate per prompt (default: 512).")
    parser.add_argument("--top-k-few-shot", type=int, default=3,
                        help="Few-shot examples per prompt (default: 3).")
    args = parser.parse_args()

    if len(args.models) < 2:
        parser.error("--models requires at least 2 model specs for cross-consistency.")

    config_path = os.path.join(PROMPTS_DIR, args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    from scripts.util.llm import prompt_generation, cross_consistency

    difficulty = args.difficulty[0] if args.difficulty and len(args.difficulty) == 1 else args.difficulty

    print(f"Config  : {args.config}")
    print(f"Models  : {', '.join(args.models)}")
    print(f"Source  : {args.source}")
    print(f"Difficulty: {difficulty or 'all'}")
    print(f"Limit   : {args.limit}")
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

    print(f"Running cross-consistency with {len(args.models)} models on {len(records)} records...")
    print()

    results = cross_consistency(
        models=args.models,
        records=records,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    for i, (rec, result) in enumerate(zip(records, results), 1):
        _print_record(i, len(records), rec, result)

    # Summary
    avg_score = sum(r.get("consistency_score", 0) for r in results) / len(results)
    full_agreement = sum(1 for r in results if r.get("consistency_score", 0) == 1.0)
    print("─" * 70)
    print(f"  Summary: {len(results)} records")
    print(f"  Average consistency score : {avg_score:.3f}")
    print(f"  Full agreement (score=1.0): {full_agreement}/{len(results)}")
    print()


if __name__ == "__main__":
    main()
