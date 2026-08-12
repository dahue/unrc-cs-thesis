"""
gen_metrics.py — Evaluate SQL predictions and export a metrics.md report.

Parses Spider's multi-column output and picks the 'all' column for each metric,
then writes a clean markdown table to metrics.md next to the input file(s).
Use --raw-metrics to also export Spider's full evaluation output as raw_metrics.txt.

Usage:
  uv run python -m src.ml.gen_metrics experiment/.../finsql.jsonl
  uv run python -m src.ml.gen_metrics experiment/.../finsql.jsonl --sql presql
  uv run python -m src.ml.gen_metrics experiment/.../presql.jsonl experiment/.../finsql.jsonl
  uv run python -m src.ml.gen_metrics experiment/.../presql.jsonl experiment/.../finsql.jsonl --raw-metrics
"""

import io
import os
import sys
import json
import sqlite3
import tempfile
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

OPENTEXT2SQL_DB = f"{ROOT_PATH}/database/OpenText2SQL.db"
DB_DIR          = f"{ROOT_PATH}/database/spider"

# Metrics to include in the report, in order.
# Keys must match the first token(s) of the Spider output row.
METRICS = [
    ("execution",        "Execution Accuracy"),
    ("exact match",      "Exact Match"),
    ("select",           "Select F1"),
    ("where",            "Where F1"),
    ("group(no Having)", "Group F1"),
    ("order",            "Order F1"),
    ("and/or",           "And/Or F1"),
    ("IUEN",             "IUEN F1"),
    ("keywords",         "Keywords F1"),
]


# ---------------------------------------------------------------------------
# Spider helpers
# ---------------------------------------------------------------------------

def _build_kmaps():
    import spider.evaluation as sp

    conn = sqlite3.connect(OPENTEXT2SQL_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM spider_tables").fetchall()
    conn.close()

    json_fields = [
        "table_names", "table_names_original",
        "column_names", "column_names_original",
        "column_types", "primary_keys", "foreign_keys",
    ]
    table_data = []
    for row in rows:
        d = dict(row)
        for field in json_fields:
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    pass
        table_data.append(d)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(table_data, tmp, indent=2)
        tmp_path = tmp.name
    try:
        kmaps = sp.build_foreign_key_map_from_json(tmp_path)
    finally:
        os.unlink(tmp_path)
    return kmaps


def _evaluate_file(jsonl_path: Path, sql_key: str, kmaps, etype: str) -> tuple[str, dict]:
    """Run Spider evaluation and return (captured_output, metadata)."""
    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Metadata
    first     = records[0] if records else {}
    sources   = sorted(set(r.get("source", "") for r in records if r.get("source")))
    diffs     = sorted(set(r.get("difficulty", "") for r in records if r.get("difficulty")))
    presql_m  = first.get("presql_model") or first.get("model") or ""
    finsql_ms = first.get("models") or []
    config    = first.get("config") or first.get("presql_config") or ""
    meta = {
        "records":       len(records),
        "source":        ", ".join(sources) or "?",
        "difficulty":    ", ".join(diffs)   or "?",
        "config":        config,
        "presql_model":  presql_m.split("/")[-1] if presql_m else "?",
        "finsql_models": [m.split("/")[-1] for m in finsql_ms],
        "sql_key":       sql_key,
    }

    gold_lines, predict_lines = [], []
    for rec in records:
        gold_sql = rec.get("gold_sql") or ""
        db_id    = rec.get("db_id") or ""
        pred_sql = rec.get(sql_key) or ""
        if gold_sql and db_id:
            gold_lines.append(f"{gold_sql}\t{db_id}")
            predict_lines.append(pred_sql)

    import spider.evaluation as sp

    buf = io.StringIO()
    original = sys.stdout
    sys.stdout = buf
    try:
        gold_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False)
        pred_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False)
        try:
            gold_tmp.write("\n".join(gold_lines) + "\n")
            pred_tmp.write("\n".join(predict_lines) + "\n")
            gold_tmp.close()
            pred_tmp.close()
            sp.evaluate(
                gold=gold_tmp.name, predict=pred_tmp.name,
                db_dir=DB_DIR, etype=etype, kmaps=kmaps,
            )
        finally:
            os.unlink(gold_tmp.name)
            os.unlink(pred_tmp.name)
    finally:
        sys.stdout = original

    return buf.getvalue(), meta


# ---------------------------------------------------------------------------
# Metric parsing — picks the 'all' (rightmost) column from Spider output
# ---------------------------------------------------------------------------

def _parse_spider_metrics(text: str) -> dict:
    """
    Parse Spider evaluation output and return {metric_name: all_column_value}.

    Spider outputs one row per metric with columns:
      easy  medium  hard  extra  all
    The 'all' column is always the last numeric value on the line.
    """
    result = {}
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 2:
            continue
        # Find where the numeric sequence begins
        float_start = None
        for i, t in enumerate(tokens):
            try:
                float(t)
                float_start = i
                break
            except ValueError:
                continue
        if float_start is None or float_start == 0:
            continue
        name = " ".join(tokens[:float_start])
        try:
            values = [float(t) for t in tokens[float_start:]]
            if values:
                result[name] = values[-1]  # rightmost = "all" column
        except ValueError:
            continue
    return result


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------

def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _comparison_table(presql_m: dict, finsql_m: dict) -> str:
    rows = [("Metric", "preSQL", "finSQL"), ("---", "---", "---")]
    for key, label in METRICS:
        rows.append((label, _fmt(presql_m.get(key)), _fmt(finsql_m.get(key))))
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def _single_table(metrics: dict, sql_key: str) -> str:
    rows = [("Metric", sql_key), ("---", "---")]
    for key, label in METRICS:
        rows.append((label, _fmt(metrics.get(key))))
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def _markdown_header(meta: dict, meta2: dict | None = None) -> str:
    config = meta.get("config", "?")
    source = meta.get("source", "?")
    diff   = meta.get("difficulty", "?")
    n      = meta.get("records", "?")

    if meta2:
        title        = f"preSQL vs finSQL — {config}"
        presql_name  = meta.get("presql_model", "?")
        finsql_names = ", ".join(meta2.get("finsql_models") or [meta2.get("presql_model", "?")])
        model_line   = f"preSQL: {presql_name}  ·  finSQL: {finsql_names}"
    else:
        sql_key    = meta.get("sql_key", "finsql")
        title      = f"{sql_key} evaluation — {config}"
        models     = meta.get("finsql_models") or []
        name       = ", ".join(models) if models else meta.get("presql_model", "?")
        model_line = f"Model: {name}"

    info_line = f"Source: {source}  ·  Difficulty: {diff}  ·  n={n}"
    return f"## {title}\n\n{model_line}  \n{info_line}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _detect_sql_key(path: Path) -> str:
    return "presql" if "presql" in path.stem.lower() else "finsql"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SQL predictions and export a metrics.md report."
    )
    parser.add_argument(
        "jsonl", nargs="+",
        help="One or two JSONL files (presql.jsonl and/or finsql.jsonl).",
    )
    parser.add_argument(
        "--sql", default=None, choices=["finsql", "presql"],
        help="SQL field to evaluate (single-file mode only; auto-detected otherwise).",
    )
    parser.add_argument(
        "--etype", default="all",
        choices=["all", "easy", "medium", "hard", "extra"],
        help="Spider evaluation type (default: all).",
    )
    parser.add_argument(
        "--raw-metrics", action="store_true",
        help="Also export Spider's full evaluation output as raw_metrics.txt.",
    )
    args = parser.parse_args()

    if len(args.jsonl) > 2:
        parser.error("At most two JSONL files can be provided.")

    paths = [Path(p) for p in args.jsonl]
    for p in paths:
        if not p.exists():
            parser.error(f"File not found: {p}")

    out_dir = paths[0].parent

    print("Building foreign-key maps...", file=sys.stderr)
    kmaps = _build_kmaps()

    # ── Single-file mode ──────────────────────────────────────────────────────
    if len(paths) == 1:
        sql_key = args.sql or _detect_sql_key(paths[0])
        raw, meta = _evaluate_file(paths[0], sql_key, kmaps, args.etype)
        m = _parse_spider_metrics(raw)
        report = f"{_markdown_header(meta)}\n\n{_single_table(m, sql_key)}\n"
        raw_combined = raw

    # ── Two-file comparison mode ──────────────────────────────────────────────
    else:
        keys = [args.sql or _detect_sql_key(p) for p in paths]

        presql_raw = finsql_raw = ""
        presql_meta = finsql_meta = {}
        for path, key in zip(paths, keys):
            raw, meta = _evaluate_file(path, key, kmaps, args.etype)
            if key == "presql":
                presql_raw, presql_meta = raw, meta
            else:
                finsql_raw, finsql_meta = raw, meta

        presql_m = _parse_spider_metrics(presql_raw)
        finsql_m = _parse_spider_metrics(finsql_raw)
        header   = _markdown_header(presql_meta, finsql_meta)
        report   = f"{header}\n\n{_comparison_table(presql_m, finsql_m)}\n"
        raw_combined = presql_raw + "\n" + finsql_raw

    metrics_path = out_dir / "metrics.md"
    metrics_path.write_text(report, encoding="utf-8")
    print(f"Saved: {metrics_path}", file=sys.stderr)

    if args.raw_metrics:
        raw_path = out_dir / "raw_metrics.txt"
        raw_path.write_text(raw_combined, encoding="utf-8")
        print(f"Saved: {raw_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
