"""
test_few_shot.py — Retrieve the 3 most similar few-shot examples for a natural language query.

Usage:
  uv run python -m scripts.test_few_shot "How many students are enrolled in each course?"
  uv run python -m scripts.test_few_shot "How many students?" --db-id student_transcripts_tracking
  uv run python -m scripts.test_few_shot "How many students?" --schema '["students(student_id, name)"]'
"""

import sys
import os
import sqlite3
import argparse
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH not set. Add it to your .env file.")

DB = f"{ROOT_PATH}/database/OpenText2SQL.db"


def _get_schema_by_db_id(db_id):
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT simplified_ddl FROM gold_dataset WHERE db_id = ? LIMIT 1", (db_id,)
        ).fetchone()
        return row[0] if row else "[]"
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Retrieve few-shot examples by semantic similarity.")
    parser.add_argument("query", help="Natural language query to find similar examples for.")
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument("--db-id", help="Look up simplified_ddl from the gold DB by database ID.")
    schema_group.add_argument("--schema", help="Simplified DDL JSON string, e.g. '[\"club(Club_ID, Name)\"]'.")
    args = parser.parse_args()

    if args.schema:
        schema = args.schema
    elif args.db_id:
        schema = _get_schema_by_db_id(args.db_id)
    else:
        schema = "[]"

    from scripts.util.nlp import get_few_shot

    examples = get_few_shot(args.query, schema, DB, DB, top_k=3)

    if not examples:
        print("No similar examples found.")
        sys.exit(0)

    print(f"Query: {args.query}\n")
    for idx, ex in enumerate(examples, 1):
        print(f"── Example {idx} {'─' * 50}  (distance: {ex['distance']:.4f})")
        print(f"  Q:   {ex['question']}")
        print(f"  SQL: {ex['sql']}")
    print()


if __name__ == "__main__":
    main()
