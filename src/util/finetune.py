"""
finetune.py — QLoRA fine-tuning of MLX models on the Spider gold_dataset.

Usage:
  uv run python -m src.util.finetune --model mlx-community/Llama-3.2-3B-Instruct-4bit
  uv run python -m src.util.finetune --model mlx-community/Llama-3.2-3B-Instruct-4bit --iters 500 --batch-size 4
"""

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import subprocess

from dotenv import load_dotenv

load_dotenv()

ROOT_PATH   = os.environ.get("ROOT_PATH", ".")
DB_PATH     = os.path.join(ROOT_PATH, "database", "OpenText2SQL.db")
ADAPTER_DIR = os.path.join(ROOT_PATH, "config", "adapter")

_OPTIMIZATION_RULE = "Given the database schema, write a SQL query that answers the question."


def _format_user_content(full_ddl_json: str, question: str) -> str:
    ddl = "\n".join(json.loads(full_ddl_json))
    return f"{_OPTIMIZATION_RULE}\n\n### Schema:\n{ddl}\n\n### Question:\n{question}"


_SPLIT_MAP = {"train": "train", "dev": "valid", "test": "test"}


def _write_split(conn: sqlite3.Connection, source: str, path: str) -> int:
    rows = conn.execute(
        "SELECT id, db_id, question, query, full_ddl, difficulty FROM gold_dataset"
        " WHERE source = ? AND is_valid = 1",
        (source,),
    ).fetchall()
    split = _SPLIT_MAP[source]
    with open(path, "w", encoding="utf-8") as f:
        for id_, db_id, question, query, full_ddl, difficulty in rows:
            user_content = _format_user_content(full_ddl, question)
            record = {
                "messages": [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": query},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            conn.execute(
                """INSERT OR REPLACE INTO finetune_dataset
                   (id, db_id, source, split, difficulty, optimization_rule, input, output)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (id_, db_id, source, split, difficulty, _OPTIMIZATION_RULE, user_content, query),
            )
    conn.commit()
    return len(rows)


def main(model: str, iters: int, batch_size: int, grad_accumulation: int, num_layers: int, learning_rate: float, resume: str | None) -> None:
    model_name   = model.split("/")[-1]
    adapter_path = os.path.join(ADAPTER_DIR, model_name)
    os.makedirs(adapter_path, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        with tempfile.TemporaryDirectory() as data_dir:
            train_n = _write_split(conn, "train", os.path.join(data_dir, "train.jsonl"))
            valid_n = _write_split(conn, "dev",   os.path.join(data_dir, "valid.jsonl"))
            test_n  = _write_split(conn, "test",  os.path.join(data_dir, "test.jsonl"))
            print(f"Dataset: {train_n} train / {valid_n} valid / {test_n} test")
            print(f"Adapter → {adapter_path}\n")

            subprocess.run(
                [
                    sys.executable, "-m", "mlx_lm", "lora",
                    "--model",        model,
                    "--train",
                    "--data",         data_dir,
                    "--adapter-path", adapter_path,
                    "--iters",        str(iters),
                    "--batch-size",              str(batch_size),
                    "--grad-accumulation-steps", str(grad_accumulation),
                    "--num-layers",     str(num_layers),
                    "--learning-rate",  str(learning_rate),
                    "--val-batches",    "10",
                    "--save-every",     "100",
                    *(["--resume-adapter-file", resume] if resume else []),
                    "--mask-prompt",
                    "--grad-checkpoint",
                ],
                check=True,
            )
    finally:
        conn.close()

    print(f"\nDone. Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning on the Spider gold_dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model path (e.g. mlx-community/Llama-3.2-3B-Instruct-4bit)",
    )
    parser.add_argument("--iters",              type=int,   default=500,  help="Training iterations")
    parser.add_argument("--batch-size",         type=int,   default=4,    help="Batch size per step")
    parser.add_argument("--grad-accumulation",  type=int,   default=1,    help="Gradient accumulation steps (effective batch = batch-size × grad-accumulation)")
    parser.add_argument("--num-layers",         type=int,   default=16,   help="LoRA layers")
    parser.add_argument("--learning-rate",      type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint adapter to resume from (e.g. config/adapter/.../0000800_adapters.safetensors)")
    args = parser.parse_args()

    main(args.model, args.iters, args.batch_size, args.grad_accumulation, args.num_layers, args.learning_rate, args.resume)
