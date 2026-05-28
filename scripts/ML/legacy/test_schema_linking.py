"""
test_schema_linking.py — Preview schema linking results for a presql.jsonl file.

Usage:
  uv run python -m scripts.test_schema_linking data/experiment/2026-05-26_01-07-40/presql.jsonl
  uv run python -m scripts.test_schema_linking data/experiment/2026-05-26_01-07-40/presql.jsonl --limit 5
"""

import sys
import json
import argparse


def _short(value, max_len=120):
    s = str(value)
    return s if len(s) <= max_len else s[:max_len] + "…"


def _print_record(idx: int, orig: dict, linked: dict):
    sep  = "─" * 70
    sep2 = "═" * 70

    print(sep2)
    print(f"  Record {idx}  |  db_id: {orig.get('db_id')}  |  difficulty: {orig.get('difficulty')}  |  source: {orig.get('source')}")
    print(sep2)
    print(f"  Question : {orig.get('question')}")
    print(f"  preSQL   : {orig.get('presql') or '(empty)'}")
    print(f"  Gold SQL : {orig.get('gold_sql')}")
    print()

    # simplified_ddl
    before = json.loads(orig.get("simplified_ddl") or "[]")
    after  = json.loads(linked.get("simplified_ddl") or "[]")
    removed = [t for t in before if t not in after]
    print(f"  simplified_ddl  ({len(before)} → {len(after)} tables)")
    for t in after:
        print(f"    ✓  {t}")
    for t in removed:
        print(f"    ✗  {t}")

    # foreign_keys
    fk_before = json.loads(orig.get("foreign_keys") or "[]")
    fk_after  = json.loads(linked.get("foreign_keys") or "[]")
    fk_removed = [f for f in fk_before if f not in fk_after]
    print(f"\n  foreign_keys  ({len(fk_before)} → {len(fk_after)} entries)")
    for f in fk_after:
        print(f"    ✓  {f}")
    for f in fk_removed:
        print(f"    ✗  {f}")

    # cell_values
    cv_before = [l for l in (orig.get("cell_values") or "").splitlines() if l.strip()]
    cv_after  = [l for l in (linked.get("cell_values") or "").splitlines() if l.strip()]
    cv_removed = [l for l in cv_before if l not in cv_after]
    print(f"\n  cell_values  ({len(cv_before)} → {len(cv_after)} tables)")
    for l in cv_after:
        print(f"    ✓  {_short(l)}")
    for l in cv_removed:
        print(f"    ✗  {_short(l)}")

    # section visibility
    visibility = linked.get("section_visibility", {})
    if visibility:
        print(f"\n  section_visibility")
        for section, visible in visibility.items():
            icon = "on " if visible else "OFF"
            print(f"    {icon}  {section}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Preview schema linking results from a presql.jsonl file.")
    parser.add_argument("jsonl", help="Path to presql.jsonl file.")
    parser.add_argument("--limit", type=int, default=None, help="Max records to display.")
    args = parser.parse_args()

    with open(args.jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        records = records[:args.limit]

    from scripts.util.llm import schema_linking
    linked = schema_linking(records)

    print(f"File   : {args.jsonl}")
    print(f"Records: {len(records)}")
    print()

    for i, (orig, out) in enumerate(zip(records, linked), 1):
        _print_record(i, orig, out)


if __name__ == "__main__":
    main()
