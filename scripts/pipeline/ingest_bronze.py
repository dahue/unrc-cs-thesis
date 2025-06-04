import os
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ["ROOT_PATH"]
TMP_DIR = os.environ["TMP_DIR"]

SPIDER_DIR = f"{TMP_DIR}/spider_data"
NATSQL_DIR = f"{TMP_DIR}/NatSQL/NatSQLv1_6"
OUT_DB = f"{ROOT_PATH}/database/bronze/bronze.sqlite"
SCHEMA_FILE = f"{ROOT_PATH}/database/bronze/schema.sql"
os.makedirs(os.path.dirname(OUT_DB), exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Connect to SQLite
conn = sqlite3.connect(OUT_DB)
cursor = conn.cursor()

# Load schema.sql and apply
with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
    schema_sql = f.read()
cursor.executescript(schema_sql)

# Insert Spider dataset
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
# load_and_insert_json("train_others.json", "train")
load_and_insert_json("dev.json", "dev")
load_and_insert_json("test.json", "test")

load_and_insert_natsql("train_spider-natsql.json", "train")
load_and_insert_natsql("dev-natsql.json", "dev")

# Insert Spider schema
load_and_insert_table_schema(os.path.join(SPIDER_DIR, "tables.json"), "train_dev")
load_and_insert_table_schema(os.path.join(SPIDER_DIR, "test_tables.json"), "test")

conn.commit()
conn.close()
