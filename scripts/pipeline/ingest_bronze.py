import os
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

TMP_DIR = os.environ.get("TMP_DIR")
if not TMP_DIR:
    raise ValueError("TMP_DIR environment variable not set. Please set it in your .env file.")

SPIDER_DIR = f"{TMP_DIR}/spider_data"
NATSQL_DIR = f"{TMP_DIR}/NatSQL/NatSQLv1_6"
OUT_DB = f"{ROOT_PATH}/database/bronze/bronze.sqlite"
SCHEMA_FILE = f"{ROOT_PATH}/database/bronze/schema.sql"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(os.path.dirname(OUT_DB), exist_ok=True)

    if not os.path.exists(SCHEMA_FILE):
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")

    conn = sqlite3.connect(OUT_DB)
    try:
        cursor = conn.cursor()

        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        cursor.executescript(schema_sql)

        # Each source file's records are identified by (id, source) where id is the
        # 0-based index within that source file. The JOIN in bronze_to_silver.py uses
        # both columns, so per-source id restart is intentional.
        def load_and_insert_json(file_name, source_label):
            data = load_json(os.path.join(SPIDER_DIR, file_name))
            for idx, record in enumerate(data):
                cursor.execute(
                    """
                    INSERT INTO spider_dataset
                    (id, db_id, source, question, question_toks, query, query_toks, query_toks_no_value, sql_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        idx,
                        record["db_id"],
                        source_label,
                        record["question"],
                        json.dumps(record.get("question_toks", [])),
                        record["query"],
                        json.dumps(record.get("query_toks", [])),
                        json.dumps(record.get("query_toks_no_value", [])),
                        json.dumps(record.get("sql", {}))
                    )
                )

        def load_and_insert_natsql(file_name, source_label):
            data = load_json(os.path.join(NATSQL_DIR, file_name))
            for idx, record in enumerate(data):
                cursor.execute(
                    """
                    INSERT INTO spider_natsql (id, source, natsql)
                    VALUES (?, ?, ?)
                    """,
                    (
                        idx,
                        source_label,
                        record["NatSQL"]
                    )
                )

        def load_and_insert_table_schema(schema_path, source_label):
            schema_list = load_json(schema_path)
            for schema in schema_list:
                cursor.execute(
                    """
                    INSERT INTO spider_tables
                    (db_id, source, table_names, table_names_original,
                     column_names, column_names_original,
                     column_types, primary_keys, foreign_keys)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        schema["db_id"],
                        source_label,
                        json.dumps(schema.get("table_names", [])),
                        json.dumps(schema.get("table_names_original", [])),
                        json.dumps(schema.get("column_names", [])),
                        json.dumps(schema.get("column_names_original", [])),
                        json.dumps(schema.get("column_types", [])),
                        json.dumps(schema.get("primary_keys", [])),
                        json.dumps(schema.get("foreign_keys", [])),
                    )
                )

        load_and_insert_json("train_spider.json", "train")
        # train_others.json is intentionally excluded (out-of-domain data not used in this thesis)
        load_and_insert_json("dev.json", "dev")
        load_and_insert_json("test.json", "test")

        load_and_insert_natsql("train_spider-natsql.json", "train")
        load_and_insert_natsql("dev-natsql.json", "dev")

        load_and_insert_table_schema(os.path.join(SPIDER_DIR, "tables.json"), "train_dev")
        load_and_insert_table_schema(os.path.join(SPIDER_DIR, "test_tables.json"), "test")

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
