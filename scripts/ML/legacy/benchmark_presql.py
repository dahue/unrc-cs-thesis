"""
benchmark_presql.py — Run gen_presql for every model in models.json, evaluate each
with gen_metrics, and print a ranked table.

Usage:
    uv run python -m scripts.ML.benchmark_presql \
        --config OpenText2SQL.json \
        --source test --difficulty easy --limit 100 --batch-size 2
"""

import os
import re
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH not set. Add it to your .env file.")

MODELS_FILE = Path(__file__).parent / "models.json"
EXP_DIR     = Path(ROOT_PATH) / "data" / "experiment"


def _parse_spider_metrics(text: str):
    exec_m  = re.search(r"^execution\s+([\d.]+)(?:\s+[\d.]+){3}\s+([\d.]+)", text, re.MULTILINE)
    exact_m = re.search(r"^exact match\s+([\d.]+)(?:\s+[\d.]+){3}\s+([\d.]+)", text, re.MULTILINE)
    exec_acc  = float(exec_m.group(2))  if exec_m  else None
    exact_acc = float(exact_m.group(2)) if exact_m else None
    return exec_acc, exact_acc


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all models in models.json on preSQL generation."
    )
    parser.add_argument("--config", required=True,
                        help="Prompt config filename, e.g. OpenText2SQL.json")
    parser.add_argument("--source", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--difficulty", nargs="*", choices=["easy", "medium", "hard", "extra"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    with open(MODELS_FILE, encoding="utf-8") as f:
        models = list(json.load(f).keys())

    sep  = "─" * 60
    sep2 = "═" * 60

    difficulty_args = []
    if args.difficulty:
        difficulty_args = ["--difficulty"] + args.difficulty

    results = []

    for idx, model_key in enumerate(models, 1):
        print(f"\n{sep2}", flush=True)
        print(f"  [{idx}/{len(models)}] {model_key}", flush=True)
        print(f"{sep2}", flush=True)

        exp_id  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = str(EXP_DIR / f"{exp_id}__{model_key}")

        presql_cmd = [
            "uv", "run", "python", "-m", "scripts.ML.gen_presql",
            "--config",     args.config,
            "--model",      model_key,
            "--source",     args.source,
            "--batch-size", str(args.batch_size),
            "--max-tokens", str(args.max_tokens),
            "--out-dir",    out_dir,
            *difficulty_args,
        ]
        if args.limit:
            presql_cmd += ["--limit", str(args.limit)]

        ret = subprocess.run(presql_cmd)
        if ret.returncode != 0:
            print(f"  ✗ gen_presql failed for {model_key}, skipping.", flush=True)
            results.append((model_key, None, None))
            continue

        presql_path = Path(out_dir) / "presql.jsonl"
        if not presql_path.exists():
            print(f"  ✗ presql.jsonl not found for {model_key}, skipping.", flush=True)
            results.append((model_key, None, None))
            continue

        print(f"\n{sep}", flush=True)
        metrics_proc = subprocess.run(
            ["uv", "run", "python", "-m", "scripts.ML.gen_metrics",
             str(presql_path), "--sql", "presql"],
            capture_output=True, text=True,
        )
        print(metrics_proc.stdout, flush=True)
        if metrics_proc.stderr:
            print(metrics_proc.stderr, file=sys.stderr, flush=True)

        exec_acc, exact_acc = _parse_spider_metrics(metrics_proc.stdout)
        results.append((model_key, exec_acc, exact_acc))

    # ── Final ranking ─────────────────────────────────────────────────────────
    print(f"\n\n{sep2}", flush=True)
    print(f"  RANKING — preSQL  ({args.source}, difficulty={args.difficulty or 'all'}, n={args.limit or 'all'})", flush=True)
    print(f"{sep2}", flush=True)
    print(f"  {'#':<4} {'Model':<42} {'Exec':>6}  {'Exact':>6}", flush=True)
    print(f"  {sep}", flush=True)

    ranked = sorted(
        [(m, e, x) for m, e, x in results if e is not None],
        key=lambda t: t[1],
        reverse=True,
    )
    failed = [(m, e, x) for m, e, x in results if e is None]

    for i, (model, exec_acc, exact_acc) in enumerate(ranked, 1):
        exact_str = f"{exact_acc:.3f}" if exact_acc is not None else "  N/A"
        print(f"  {i:<4} {model:<42} {exec_acc:.3f}  {exact_str}", flush=True)

    for model, _, _ in failed:
        print(f"  {'—':<4} {model:<42} {'FAILED':>6}", flush=True)

    print(flush=True)


if __name__ == "__main__":
    main()
