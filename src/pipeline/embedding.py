"""
embedding.py — Builds the few-shot vector index using sqlite-vec.

Public API:
  build_index(gold_db_path, index_db_path)

Retrieval utilities (get_question_skeleton, get_few_shot) live in src.util.nlp.

CLI:
  uv run python -m src.pipeline.embedding
"""

import os
import sqlite3

import nltk
import sqlite_vec

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from src.util.nlp import _get_nlp, _open_vec_conn, get_question_skeleton

_TABLE = "embedding_dataset"
_SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "database", "embedding.sql")


def _load_schema():
    with open(os.path.normpath(_SCHEMA_FILE), "r", encoding="utf-8") as f:
        return f.read()


def build_index(gold_db_path, index_db_path):
    """
    Build (or rebuild) the few-shot vector index from gold DB training entries.

    Recreates the virtual table so schema changes (such as the distance metric)
    take effect before repopulating the training vectors.
    """
    nlp = _get_nlp()

    conn = _open_vec_conn(index_db_path)
    try:
        print(f"Recreating {_TABLE} (vector_dim={nlp.vocab.vectors_length})")
        conn.executescript(_load_schema())

        if gold_db_path == index_db_path:
            rows = conn.execute(
                "SELECT id, db_id, source, question, simplified_ddl "
                "FROM gold_dataset WHERE source = 'train'"
            ).fetchall()
        else:
            gold_conn = sqlite3.connect(gold_db_path)
            try:
                rows = gold_conn.execute(
                    "SELECT id, db_id, source, question, simplified_ddl "
                    "FROM gold_dataset WHERE source = 'train'"
                ).fetchall()
            finally:
                gold_conn.close()

        print(f"Indexing {len(rows)} training entries...")
        for rowid, (id_, db_id, source, question, simplified_ddl) in enumerate(rows, start=1):
            skeleton = get_question_skeleton(question, simplified_ddl)
            vector_bytes = nlp(skeleton).vector.astype('float32').tobytes()
            conn.execute(
                f"INSERT INTO {_TABLE}(rowid, vector, id, db_id, source, question, skeleton_question) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rowid, vector_bytes, id_, db_id, source, question, skeleton),
            )
            if rowid % 500 == 0:
                print(f"  {rowid}/{len(rows)} indexed...")

        conn.commit()
        print(f"✓ Index built: {len(rows)} vectors saved to {index_db_path}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()
    ROOT_PATH = os.environ.get("ROOT_PATH")
    if not ROOT_PATH:
        raise ValueError("ROOT_PATH not set. Add it to your .env file.")

    parser = argparse.ArgumentParser(description="Build the few-shot vector index from the gold DB.")
    parser.parse_args()

    db = f"{ROOT_PATH}/database/OpenText2SQL.db"

    if not os.path.exists(db):
        raise FileNotFoundError(f"Database not found at {db}. Run src/pipeline/ingest.py first.")

    build_index(db, db)
