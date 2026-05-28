"""
gen_finsql.py — Generate finSQL predictions via cross-consistency from a presql.jsonl file.

Pipeline:
  1. Load records from presql.jsonl (all schema data is already embedded).
  2. Apply schema linking to prune simplified_ddl / foreign_keys / cell_values to only
     the tables referenced in the preSQL, and compute section_visibility overrides.
  3. Render new finSQL prompts using the pruned schema, disabling config sections that
     became empty after linking (e.g. foreign_keys with no surviving entries).
  4. Run cross-consistency: each model in --models generates SQL sequentially; results
     are aggregated by semantic equivalence (SQL execution against the Spider DB).
  5. Write finsql.jsonl next to the input presql.jsonl.

Usage:
    uv run python -m scripts.ML.gen_finsql \
        --presql data/experiment/2026-05-26_02-15-18/presql.jsonl \
        --config OpenText2SQL.json \
        --models Llama-3.2-3B-Instruct-4bit Qwen3-14B-4bit

    uv run python -m scripts.ML.gen_finsql \
        --presql data/experiment/2026-05-26_02-15-18/presql.jsonl \
        --config OpenText2SQL.json \
        --models Llama-3.2-3B-Instruct-4bit Qwen3-14B-4bit \
        --limit 50 --batch-size 4

    uv run python -m scripts.ML.gen_finsql \
        --presql data/experiment/2026-05-26_02-15-18/presql.jsonl \
        --config OpenText2SQL.json \
        --models Qwen3-14B-4bit

Output:
  data/experiment/2026-05-26_02-15-18/finsql.jsonl
  Each line: {question, db_id, source, difficulty, gold_sql,
              presql, presql_model, presql_config, presql_prompt,
              simplified_ddl, foreign_keys, cell_values, few_shot, section_visibility,
              finsql_prompt, finsql, all_sql, consistency_score, models, config}
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH not set. Add it to your .env file.")

PROMPTS_DIR = f"{ROOT_PATH}/data/prompt"


def main():
    parser = argparse.ArgumentParser(
        description="Generate finSQL via cross-consistency from a presql.jsonl file."
    )
    parser.add_argument("--presql", required=True,
                        help="Path to presql.jsonl (e.g. data/experiment/2026-05-26.../presql.jsonl).")
    parser.add_argument("--config", required=True,
                        help="Prompt config filename, e.g. OpenText2SQL.json")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Two or more model specs for cross-consistency (short keys or full HuggingFace paths).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of records to process.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Prompts per inference batch (default: 1).")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate per prompt (default: 512).")
    args = parser.parse_args()

    # ── Step 1: load presql.jsonl ─────────────────────────────────────────────
    presql_path = Path(args.presql)
    if not presql_path.exists():
        raise FileNotFoundError(f"presql.jsonl not found: {presql_path}")

    with open(presql_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        records = records[:args.limit]

    print(f"Loaded {len(records)} records from {presql_path}")

    # ── Step 2: load prompt config ────────────────────────────────────────────
    config_path = os.path.join(PROMPTS_DIR, args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # ── Step 3: schema linking ────────────────────────────────────────────────
    from scripts.util.llm import schema_linking, render_prompt, cross_consistency, resolve_model

    print("Applying schema linking...")
    linked = schema_linking(records)

    # ── Step 4: render finSQL prompts ─────────────────────────────────────────
    print("Rendering finSQL prompts with pruned schema...")
    for rec in linked:
        rendered = render_prompt(config, rec, rec.get("section_visibility"))
        rec["prompt"] = " ".join(rendered.split())  # collapse to one line

    # ── Step 5: cross-consistency inference ───────────────────────────────────
    if len(args.models) > 1:
        print(f"Running cross-consistency with {len(args.models)} models on {len(linked)} records...")
    else:
        print(f"Running inference (cross-consistency disabled — single model) on {len(linked)} records...")
    results = cross_consistency(
        models=args.models,
        records=linked,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    # ── Step 6: write finsql.jsonl ────────────────────────────────────────────
    out_path = presql_path.parent / "finsql.jsonl"
    resolved_models = [resolve_model(m.partition(":")[0]) for m in args.models]

    with open(out_path, "w", encoding="utf-8") as f:
        for orig, rec_linked, result in zip(records, linked, results):
            line = {
                # identity
                "question":           orig.get("question"),
                "db_id":              orig.get("db_id"),
                "source":             orig.get("source"),
                "difficulty":         orig.get("difficulty"),
                "gold_sql":           orig.get("gold_sql"),
                # preSQL step (preserved from input)
                "presql":             orig.get("presql"),
                "presql_model":       orig.get("model"),
                "presql_config":      orig.get("config"),
                "presql_prompt":      orig.get("prompt"),
                # schema linking output
                "simplified_ddl":     rec_linked.get("simplified_ddl"),
                "foreign_keys":       rec_linked.get("foreign_keys"),
                "cell_values":        rec_linked.get("cell_values"),
                "few_shot":           orig.get("few_shot"),
                "section_visibility": rec_linked.get("section_visibility"),
                # finSQL step
                "finsql_prompt":      rec_linked.get("prompt"),
                "finsql":             result.get("sql"),
                "all_sql":            result.get("all_sql"),
                "consistency_score":  result.get("consistency_score"),
                "models":             resolved_models,
                "config":             args.config,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"✓ {len(results)} records written to {out_path}")


if __name__ == "__main__":
    main()
