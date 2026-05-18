import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

SILVER_DB = f"{ROOT_PATH}/database/silver/silver.sqlite"
GOLD_DB = f"{ROOT_PATH}/database/gold/gold.sqlite"
SCHEMA_FILE = f"{ROOT_PATH}/database/gold/schema.sql"


def main():
    os.makedirs(os.path.dirname(GOLD_DB), exist_ok=True)

    if not os.path.exists(SCHEMA_FILE):
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")

    conn_gold = sqlite3.connect(GOLD_DB)
    try:
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            conn_gold.executescript(f.read())
        cursor_gold = conn_gold.cursor()

        conn_silver = sqlite3.connect(SILVER_DB)
        try:
            cursor_silver = conn_silver.cursor()
            cursor_silver.execute("""
                SELECT d.id, d.db_id, d.source, d.question, d.query, d.is_valid, d.notes,
                       d.simplified_ddl, d.full_ddl, d.foreign_keys, d.difficulty, d.natsql
                FROM silver_dataset d
            """)
            rows = cursor_silver.fetchall()
        finally:
            conn_silver.close()

        error_count = 0
        for row_id, db_id, source, question, query, is_valid, notes, simplified_ddl, full_ddl, foreign_keys, difficulty, natsql in rows:
            try:
                cursor_gold.execute(
                    """
                    INSERT INTO gold_dataset (
                        id, db_id, source, question, query, is_valid, notes,
                        simplified_ddl, full_ddl, foreign_keys, difficulty, natsql
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_id,
                        db_id,
                        source,
                        question,
                        query,
                        is_valid,
                        notes,
                        simplified_ddl,
                        full_ddl,
                        foreign_keys,
                        difficulty,
                        natsql
                    )
                )
            except Exception as e:
                print(f"❌ Error processing db_id={db_id}: {e}")
                error_count += 1

        if error_count:
            print(f"⚠️ Completed with {error_count} row error(s). Committing successful rows.")
        conn_gold.commit()
    except Exception:
        conn_gold.rollback()
        raise
    finally:
        conn_gold.close()


if __name__ == "__main__":
    main()
