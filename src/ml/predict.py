"""
predict.py — Run inference with a fine-tuned model on the Spider gold_dataset.

Builds prompts in the same format used during QLoRA fine-tuning (Finetune.json):
  instruction + full DDL schema + question → SQL prediction.

Output is written as a JSONL file compatible with gen_metrics.py.

Usage:
  uv run python -m src.ml.predict --model mlx-community/Llama-3.2-3B-Instruct-4bit
  uv run python -m src.ml.predict --model mlx-community/Llama-3.2-3B-Instruct-4bit \\
      --source test --difficulty medium --limit 100
  uv run python -m src.ml.predict --model mlx-community/Llama-3.2-3B-Instruct-4bit \\
      --adapter-path config/adapter/Llama-3.2-3B-Instruct-4bit
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Environment

load_dotenv()

ROOT_PATH   = os.environ.get("ROOT_PATH", ".")
DB_PATH     = os.path.join(ROOT_PATH, "database", "OpenText2SQL.db")
ADAPTER_DIR = os.path.join(ROOT_PATH, "config", "adapter")
CONFIG_DIR  = os.path.join(ROOT_PATH, "config", "prompt")
EXP_DIR     = os.path.join(ROOT_PATH, "experiments")


def _load_config(config_name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, config_name), encoding="utf-8") as f:
        return json.load(f)


def _build_prompt(config: dict, full_ddl_json: str, question: str) -> str:
    parts = [s["text"] for s in config.values() if s.get("visible", True)]
    template = "\n\n".join(parts)

    ddl = "\n".join(json.loads(full_ddl_json))

    # Detect and replicate any comment-style prefix before {{full_ddl}}
    m = re.search(r'^([\s#\-\*]*)\{\{\s*full_ddl\s*\}\}', template, re.MULTILINE)
    if m and m.group(1):
        ddl = ddl.replace("\n", "\n" + m.group(1))

    return Environment().from_string(template).render(full_ddl=ddl, question=question)


def _load_records(
    difficulty: list[str] | None,
    source: str,
    limit: int | None,
) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM gold_dataset WHERE source = ? AND is_valid = 1"
        params: list = [source]
        if difficulty:
            placeholders = ",".join("?" * len(difficulty))
            query += f" AND difficulty IN ({placeholders})"
            params.extend(difficulty)
        if limit:
            query += f" LIMIT {limit}"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def main(
    model: str,
    adapter_path: str | None,
    config_name: str,
    source: str,
    difficulty: list[str] | None,
    limit: int | None,
    batch_size: int,
    max_tokens: int,
    out_dir: str | None,
) -> Path:
    from src.util.llm import infer

    # Resolve adapter path
    # None  → auto-detect from config/adapter/<model-name>/
    # False → explicitly disabled (--no-adapter)
    # str   → explicit path
    if adapter_path is False:
        adapter_path = None
    elif adapter_path is None:
        model_name = model.split("/")[-1]
        candidate = os.path.join(ADAPTER_DIR, model_name)
        if os.path.isdir(candidate):
            adapter_path = candidate
        else:
            print(
                f"Warning: no adapter found at {candidate}, running without adapter.",
                file=sys.stderr,
            )

    config = _load_config(config_name)

    records = _load_records(difficulty, source, limit)
    if not records:
        raise ValueError("No records matched the given filters.")
    print(f"Loaded {len(records)} records ({source}, difficulty={difficulty or 'all'}).", file=sys.stderr)

    prompts = [_build_prompt(config, r["full_ddl"], r["question"]) for r in records]

    model_spec = f"{model}:fine-tuned" if adapter_path else model
    predictions = infer(
        model=model_spec,
        prompts=prompts,
        batch_size=batch_size,
        max_tokens=max_tokens,
        adapter_path=adapter_path,
    )

    # Write output JSONL
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(EXP_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / "predict.jsonl"

    model_short = model.split("/")[-1]
    with open(out_path, "w", encoding="utf-8") as f:
        for rec, pred in zip(records, predictions):
            f.write(json.dumps({
                "db_id":      rec["db_id"],
                "source":     rec["source"],
                "difficulty": rec["difficulty"],
                "question":   rec["question"],
                "gold_sql":   rec["query"],
                "finsql":     pred,
                "config":     config_name,
                "models":     [model_short],
            }, ensure_ascii=False) + "\n")

    print(f"Saved predictions: {out_path}", file=sys.stderr)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with a fine-tuned model on the Spider gold_dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model path (e.g. mlx-community/Llama-3.2-3B-Instruct-4bit)",
    )
    parser.add_argument(
        "--adapter-path", default=None,
        help="Path to LoRA adapter directory. Defaults to config/adapter/<model-name>/.",
    )
    parser.add_argument(
        "--no-adapter", action="store_true",
        help="Disable adapter loading (baseline mode, no fine-tuning).",
    )
    parser.add_argument(
        "--config", default="Finetune.json",
        help="Prompt config filename (for metadata only).",
    )
    parser.add_argument(
        "--source", default="test", choices=["train", "dev", "test"],
        help="Dataset split.",
    )
    parser.add_argument(
        "--difficulty", nargs="+",
        choices=["easy", "medium", "hard", "extra"],
        help="Filter by difficulty level(s).",
    )
    parser.add_argument("--limit",      type=int, default=None, help="Max records.")
    parser.add_argument("--batch-size", type=int, default=1,    help="Inference batch size.")
    parser.add_argument("--max-tokens", type=int, default=256,  help="Max tokens per prediction.")
    parser.add_argument("--out-dir",    default=None,           help="Output directory.")
    args = parser.parse_args()

    out_path = main(
        model=args.model,
        adapter_path=False if args.no_adapter else args.adapter_path,
        config_name=args.config,
        source=args.source,
        difficulty=args.difficulty,
        limit=args.limit,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        out_dir=args.out_dir,
    )

    # Run gen_metrics automatically
    import subprocess
    print("\nRunning gen_metrics...", file=sys.stderr)
    subprocess.run(
        [sys.executable, "-m", "src.ml.gen_metrics", str(out_path)],
        check=True,
    )
