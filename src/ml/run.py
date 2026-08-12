"""
run.py — Run the full OpenText2SQL pipeline end-to-end.

Pipeline:
  1. preSQL      — generate preliminary SQL used exclusively for schema linking.
  2. finSQL      — apply schema linking, build pruned prompts, run cross-consistency
                   inference with multiple models, and vote by semantic equivalence.
  3. metrics     — evaluate both against gold SQL; write metrics.md + raw_metrics.txt.

All three steps write their output under the same experiment directory:
  experiments/<YYYY-MM-DD_HH-MM-SS>/
    presql.jsonl
    finsql.jsonl
    metrics.md
    raw_metrics.txt

Usage:
    uv run python -m src.ml.run \
        --config OpenText2SQL.json \
        --models Qwen3.5-9B-MLX-4bit Qwen3-14B-4bit gemma-3-12b-it-4bit-DWQ phi-4-4bit granite-4.1-8b-4bit \
        --source test --difficulty hard --limit 100 \
        --batch-size 2

    uv run python -m src.ml.run \
        --config OpenText2SQL.json \
        --presql-model Qwen3.5-9B-MLX-4bit \
        --finsql-models Qwen3.5-9B-MLX-4bit Qwen3-14B-4bit \
        --source test --difficulty hard --limit 100 \
        --batch-size 2

  Skip preSQL (baseline: full-schema prompt fed directly into finSQL, no schema linking):

    uv run python -m src.ml.run \
        --config OpenText2SQL.json \
        --skip-presql \
        --finsql-models Qwen3.5-9B-MLX-4bit Qwen3-14B-4bit \
        --source test --difficulty hard --limit 100 \
        --batch-size 2

    uv run python -m src.ml.run \
        --config OpenText2SQL.json \
        --skip-presql \
        --models Qwen3.5-9B-MLX-4bit Qwen3-14B-4bit \
        --source test --difficulty hard --limit 100 \
        --batch-size 2
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH not set. Add it to your .env file.")

DB          = f"{ROOT_PATH}/database/OpenText2SQL.db"
PROMPTS_DIR = f"{ROOT_PATH}/config/prompt"
EXP_DIR     = f"{ROOT_PATH}/experiments"


def main():
    parser = argparse.ArgumentParser(
        description="Run the full OpenText2SQL pipeline: preSQL → finSQL → metrics."
    )

    # Prompt / dataset params
    parser.add_argument("--config", required=True,
                        help="Prompt config filename, e.g. OpenText2SQL.json")
    parser.add_argument("--source", choices=["train", "dev", "test"], default="dev",
                        help="Dataset split to use (default: dev).")
    parser.add_argument("--difficulty", nargs="*", choices=["easy", "medium", "hard", "extra"],
                        help="Filter by one or more difficulty levels.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of records to process.")
    parser.add_argument("--top-k-few-shot", type=int, default=3,
                        help="Few-shot examples per prompt (default: 3).")

    # Inference params
    parser.add_argument("--models", nargs="+",
                        help="Model list: first model used for preSQL, all models used for finSQL. "
                             "Mutually exclusive with --presql-model / --finsql-models.")
    parser.add_argument("--presql-model",
                        help="Model for the preSQL step (short key or full HuggingFace path).")
    parser.add_argument("--finsql-models", nargs="+",
                        help="One or more models for finSQL cross-consistency.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Prompts per inference batch (default: 1).")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate per prompt (default: 512).")
    parser.add_argument("--skip-presql", action="store_true",
                        help="Skip preSQL inference and feed the initial full-schema prompts "
                             "directly into finSQL cross-consistency. --presql-model is ignored.")

    # Evaluation params
    parser.add_argument("--etype", default="all",
                        choices=["all", "easy", "medium", "hard", "extra"],
                        help="Spider evaluation type (default: all).")

    args = parser.parse_args()

    # ── Resolve model args ────────────────────────────────────────────────────
    if args.models:
        if args.presql_model or args.finsql_models:
            parser.error("--models cannot be combined with --presql-model or --finsql-models.")
        presql_model  = args.models[0]
        finsql_models = args.models
    elif args.skip_presql:
        if not args.finsql_models:
            parser.error("With --skip-presql, provide --finsql-models (or use --models).")
        presql_model  = None
        finsql_models = args.finsql_models
    else:
        if not args.presql_model or not args.finsql_models:
            parser.error("Provide either --models or both --presql-model and --finsql-models.")
        presql_model  = args.presql_model
        finsql_models = args.finsql_models

    # ── Create experiment directory ───────────────────────────────────────────
    experiment_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(EXP_DIR, experiment_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Experiment: {out_dir}")
    print()

    # ── Load prompt config ────────────────────────────────────────────────────
    config_path = os.path.join(PROMPTS_DIR, args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    sep = "─" * 60
    print(f"{sep}")
    print(f"  Prompt config: {args.config}")
    print(f"{sep}")
    for section, cfg in config.items():
        status = "on " if cfg.get("visible", True) else "OFF"
        print(f"  [{status}]  {section}")
    print()

    difficulty = args.difficulty[0] if args.difficulty and len(args.difficulty) == 1 else args.difficulty

    from src.util.llm import (
        prompt_generation, infer, resolve_model,
        schema_linking, render_prompt, cross_consistency,
    )
    from src.ml.gen_presql import write_presql_jsonl
    from src.ml.gen_finsql import write_finsql_jsonl

    # ══ Step 1: preSQL ═══════════════════════════════════════════════════════
    print(f"{sep}")
    presql_label = "SKIPPED" if args.skip_presql else presql_model
    print(f"  Step 1/3 — preSQL  [{presql_label}]")
    print(f"{sep}")

    print(f"Generating prompts ({args.source}, difficulty={difficulty or 'all'}, limit={args.limit})...")
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

    if args.skip_presql:
        print("preSQL inference skipped — initial prompts will be used directly for finSQL.")
        presql_path = None
        # Normalise to the same field layout that finsql steps expect from presql_records
        presql_records = [{
            "question":       rec["question"],
            "db_id":          rec["db_id"],
            "source":         rec["source"],
            "difficulty":     rec["difficulty"],
            "gold_sql":       rec["query"],
            "presql":         None,
            "model":          None,
            "config":         args.config,
            "prompt":         rec["prompt"],
            "few_shot":       rec["few_shot"],
            "simplified_ddl": rec["simplified_ddl"],
            "foreign_keys":   rec["foreign_keys"],
            "cell_values":    rec["cell_values"],
        } for rec in records]
    else:
        prompts = [r["prompt"] for r in records]
        print(f"Running preSQL inference...")
        presql_results = infer(
            model=presql_model,
            prompts=prompts,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )

        resolved_presql_model = resolve_model(presql_model.partition(":")[0])
        presql_path = write_presql_jsonl(out_dir, records, presql_results, resolved_presql_model, args.config)
        print(f"✓ presql.jsonl → {presql_path}")
    print()

    # ══ Step 2: finSQL ═══════════════════════════════════════════════════════
    print(f"{sep}")
    print(f"  Step 2/3 — finSQL  [{', '.join(finsql_models)}]")
    print(f"{sep}")

    if args.skip_presql:
        print("Schema linking: SKIPPED")
        linked = [dict(rec) for rec in presql_records]
    else:
        with open(presql_path, encoding="utf-8") as f:
            presql_records = [json.loads(line) for line in f if line.strip()]

        print("Applying schema linking...")
        linked = schema_linking(presql_records)

        print("Rendering finSQL prompts with pruned schema...")
        for rec in linked:
            rendered = render_prompt(config, rec, rec.get("section_visibility"))
            rec["prompt"] = " ".join(rendered.split())

    if len(finsql_models) > 1:
        print(f"Running cross-consistency with {len(finsql_models)} models...")
    else:
        print(f"Running inference (cross-consistency disabled — single model)...")
    finsql_results = cross_consistency(
        models=finsql_models,
        records=linked,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    resolved_finsql_models = [resolve_model(m.partition(":")[0]) for m in finsql_models]
    finsql_path = write_finsql_jsonl(out_dir, presql_records, linked, finsql_results, resolved_finsql_models, args.config)
    print(f"✓ finsql.jsonl → {finsql_path}")
    print()

    # ══ Step 3: metrics ══════════════════════════════════════════════════════
    print(f"{sep}")
    print(f"  Step 3/3 — metrics  [etype={args.etype}]")
    print(f"{sep}")

    from src.ml.gen_metrics import (
        _build_kmaps, _evaluate_file, _parse_spider_metrics,
        _comparison_table, _single_table, _markdown_header,
    )

    print("Building foreign-key maps...", file=sys.stderr)
    kmaps = _build_kmaps()

    finsql_raw, finsql_meta = _evaluate_file(Path(finsql_path), "finsql", kmaps, args.etype)
    finsql_m = _parse_spider_metrics(finsql_raw)

    out_dir_path = Path(out_dir)
    if presql_path is not None:
        presql_raw, presql_meta = _evaluate_file(Path(presql_path), "presql", kmaps, args.etype)
        presql_m = _parse_spider_metrics(presql_raw)
        header = _markdown_header(presql_meta, finsql_meta)
        report = f"{header}\n\n{_comparison_table(presql_m, finsql_m)}\n"
    else:
        presql_raw = ""
        header = _markdown_header(finsql_meta)
        report = f"{header}\n\n{_single_table(finsql_m, 'finsql')}\n"

    metrics_md = out_dir_path / "metrics.md"
    metrics_md.write_text(report, encoding="utf-8")
    print(f"✓ metrics.md → {metrics_md}")

    raw_txt = out_dir_path / "raw_metrics.txt"
    raw_txt.write_text((presql_raw + "\n" if presql_raw else "") + finsql_raw, encoding="utf-8")
    print(f"✓ raw_metrics.txt → {raw_txt}")


if __name__ == "__main__":
    main()
