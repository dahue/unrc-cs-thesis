"""
ingest.py — Single-pass ingestion pipeline.

Creates database/OpenText2SQL.db containing:
  - bronze_dataset   raw Spider Q/SQL pairs
  - spider_tables    raw schema metadata
  - silver_dataset   cleaned, schema-enriched, difficulty-labelled rows
  - gold_dataset     final curated dataset consumed by the ML pipeline

CLI:
  uv run python -m src.pipeline.ingest
"""

import os
import json
import re
import sqlite3

from dotenv import load_dotenv
from spider import evaluation, process_sql

# Teach sqlite3 to round-trip Python booleans through BOOLEAN columns.
sqlite3.register_adapter(bool, int)
sqlite3.register_converter("BOOLEAN", lambda v: bool(int(v)))

load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH not set. Add it to your .env file.")

TMP_DIR = os.environ.get("TMP_DIR")
if not TMP_DIR:
    raise ValueError("TMP_DIR not set. Add it to your .env file.")

SPIDER_DIR = f"{TMP_DIR}/spider_data"
SPIDER_DB_PATH = f"{ROOT_PATH}/database/spider"
OUT_DB = f"{ROOT_PATH}/database/OpenText2SQL.db"
SCHEMA_FILE = f"{ROOT_PATH}/database/OpenText2SQL.sql"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_question(text):
    return text.strip().lower()


def _normalize_sql(sql):
    return re.sub(r"\s+", " ", sql.strip().lower())


def _get_difficulty(sql_str, db_id):
    try:
        evaluator = evaluation.Evaluator()
        schema_path = os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")
        schema = process_sql.Schema(process_sql.get_schema(schema_path))
        parsed = process_sql.get_sql(schema, sql_str)
        return evaluator.eval_hardness(parsed)
    except Exception as e:
        print(f"  ⚠️  difficulty eval failed for {db_id}: {e}")
        return None


def _get_schema_context(conn, db_id):
    row = conn.execute(
        "SELECT table_names_original, column_names_original, column_types, foreign_keys "
        "FROM spider_tables WHERE db_id = ?",
        (db_id,),
    ).fetchone()
    if not row:
        return "", "", ""

    table_names = json.loads(row[0])
    column_names = json.loads(row[1])
    column_types = json.loads(row[2])
    foreign_keys_raw = json.loads(row[3])

    table_columns = {t: [] for t in table_names}
    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx >= 0:
            table_columns[table_names[table_idx]].append((col_name, column_types[idx]))

    simplified_ddl = [
        f"{table}({', '.join(col for col, _ in cols)})"
        for table, cols in table_columns.items()
    ]
    full_ddl = [
        f"CREATE TABLE {table}({', '.join(f'{col} {typ}' for col, typ in cols)});"
        for table, cols in table_columns.items()
    ]

    fk_list = []
    for i, j in foreign_keys_raw:
        src_table_idx, src_col = column_names[i]
        tgt_table_idx, tgt_col = column_names[j]
        fk_list.append(
            f"{table_names[src_table_idx]}({src_col}) "
            f"REFERENCES {table_names[tgt_table_idx]}({tgt_col})"
        )

    return json.dumps(simplified_ddl), json.dumps(full_ddl), json.dumps(fk_list)


# ---------------------------------------------------------------------------
# Ingestion phases
# ---------------------------------------------------------------------------

def _ingest_spider(conn):
    def insert_json(file_name, source):
        data = _load_json(os.path.join(SPIDER_DIR, file_name))
        conn.executemany(
            "INSERT INTO bronze_dataset "
            "(id, db_id, source, question, question_toks, query, query_toks, query_toks_no_value, sql_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    idx,
                    r["db_id"],
                    source,
                    r["question"],
                    json.dumps(r.get("question_toks", [])),
                    r["query"],
                    json.dumps(r.get("query_toks", [])),
                    json.dumps(r.get("query_toks_no_value", [])),
                    json.dumps(r.get("sql", {})),
                )
                for idx, r in enumerate(data)
            ],
        )
        print(f"  bronze_dataset ← {file_name} ({len(data)} rows)")

    def insert_tables(file_name, source):
        data = _load_json(os.path.join(SPIDER_DIR, file_name))
        conn.executemany(
            "INSERT INTO spider_tables "
            "(db_id, source, table_names, table_names_original, column_names, "
            "column_names_original, column_types, primary_keys, foreign_keys) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    s["db_id"],
                    source,
                    json.dumps(s.get("table_names", [])),
                    json.dumps(s.get("table_names_original", [])),
                    json.dumps(s.get("column_names", [])),
                    json.dumps(s.get("column_names_original", [])),
                    json.dumps(s.get("column_types", [])),
                    json.dumps(s.get("primary_keys", [])),
                    json.dumps(s.get("foreign_keys", [])),
                )
                for s in data
            ],
        )
        print(f"  spider_tables  ← {file_name} ({len(data)} rows)")

    insert_json("train_spider.json", "train")
    insert_json("dev.json", "dev")
    insert_json("test.json", "test")
    insert_tables("tables.json", "train_dev")
    insert_tables("test_tables.json", "test")


def _build_silver(conn):
    rows = conn.execute(
        "SELECT id, db_id, source, question, query, query_toks_no_value, sql_json "
        "FROM bronze_dataset"
    ).fetchall()

    errors = 0
    batch = []
    for row_id, db_id, source, question, query, query_toks_no_value, sql_json in rows:
        try:
            simplified_ddl, full_ddl, foreign_keys = _get_schema_context(conn, db_id)
            difficulty = _get_difficulty(query, db_id)
            batch.append((
                row_id,
                db_id,
                source,
                _clean_question(question),
                _normalize_sql(query),
                query_toks_no_value,
                sql_json,
                True,
                simplified_ddl,
                full_ddl,
                foreign_keys,
                difficulty,
            ))
        except Exception as e:
            print(f"  ❌ {db_id} id={row_id}: {e}")
            errors += 1

    conn.executemany(
        "INSERT INTO silver_dataset "
        "(id, db_id, source, question, query, query_toks_no_value, sql_json, "
        "is_valid, simplified_ddl, full_ddl, foreign_keys, difficulty) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        batch,
    )
    print(f"  silver_dataset ← {len(batch)} rows ({errors} errors)")


def _build_gold(conn):
    rows = conn.execute(
        "SELECT id, db_id, source, question, query, is_valid, "
        "simplified_ddl, full_ddl, foreign_keys, difficulty "
        "FROM silver_dataset"
    ).fetchall()

    conn.executemany(
        "INSERT INTO gold_dataset "
        "(id, db_id, source, question, query, is_valid, "
        "simplified_ddl, full_ddl, foreign_keys, difficulty) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    print(f"  gold_dataset   ← {len(rows)} rows")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUT_DB), exist_ok=True)

    if not os.path.exists(SCHEMA_FILE):
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")

    conn = sqlite3.connect(OUT_DB, detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

        print("Phase 1 — ingesting Spider data...")
        _ingest_spider(conn)

        print("Phase 2 — building silver_dataset...")
        _build_silver(conn)

        print("Phase 3 — building gold_dataset...")
        _build_gold(conn)

        conn.commit()
        print(f"\n✓ Done → {OUT_DB}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
