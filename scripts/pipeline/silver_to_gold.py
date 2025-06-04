import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ["ROOT_PATH"]

SILVER_DB = f"{ROOT_PATH}/database/silver/silver.sqlite"
GOLD_DB = f"{ROOT_PATH}/database/gold/gold.sqlite"
SCHEMA_FILE = f"{ROOT_PATH}/database/gold/schema.sql"

os.makedirs(os.path.dirname(GOLD_DB), exist_ok=True)

# -- Load schema and initialize gold DB --
conn_gold = sqlite3.connect(GOLD_DB)
with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
    conn_gold.executescript(f.read())
cursor_gold = conn_gold.cursor()

# -- Connect to silver DB --
conn_silver = sqlite3.connect(SILVER_DB)
cursor_silver = conn_silver.cursor()


cursor_silver.execute("""
    SELECT d.id, d.db_id, d.source, d.question, d.query, d.is_valid, d.notes, d.simplified_ddl, d.full_ddl, d.foreign_keys, d.difficulty, d.natsql
    FROM silver_dataset d
""")
rows = cursor_silver.fetchall()

# -- Process and insert --
for id, db_id, source, question, query, is_valid, notes, simplified_ddl, full_ddl, foreign_keys, difficulty, natsql in rows:
    try:
        cursor_gold.execute(
            """
            INSERT INTO gold_dataset (
                id, db_id, source, question, query, is_valid, notes, simplified_ddl, full_ddl, foreign_keys, difficulty, natsql
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
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

conn_gold.commit()
conn_gold.close()
