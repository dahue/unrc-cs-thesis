import os
import sqlite3
import json
import re
from spider import evaluation, process_sql
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

SPIDER_DB_PATH = f"{ROOT_PATH}/database/spider"
BRONZE_DB = f"{ROOT_PATH}/database/bronze/bronze.sqlite"
SILVER_DB = f"{ROOT_PATH}/database/silver/silver.sqlite"
SCHEMA_FILE = f"{ROOT_PATH}/database/silver/schema.sql"


def get_query_difficulty(sql_str, db_id):
    try:
        evaluator = evaluation.Evaluator()
        schema_path = os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")
        schema = process_sql.Schema(process_sql.get_schema(schema_path))
        parsed_sql = process_sql.get_sql(schema, sql_str)
        return evaluator.eval_hardness(parsed_sql)
    except Exception as e:
        print(f"⚠️ Failed to evaluate difficulty for {db_id}: {e}")
        return None


def clean_question(text):
    return text.strip().lower()


def normalize_sql(sql):
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def get_schema_context(conn, db_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM spider_tables WHERE db_id = ?", (db_id,))
    row = cursor.fetchone()
    if not row:
        return "", "", ""

    (
        _db_id,
        _source,
        table_names,
        table_names_original,
        column_names,
        column_names_original,
        column_types,
        primary_keys,
        foreign_keys
    ) = row[0:9]

    table_names = json.loads(table_names_original)
    column_names = json.loads(column_names_original)
    column_types = json.loads(column_types)
    primary_keys = json.loads(primary_keys)
    foreign_keys = json.loads(foreign_keys)

    table_columns = {t: [] for t in table_names}
    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx >= 0:
            col_type = column_types[idx]
            table_columns[table_names[table_idx]].append((col_name, col_type))

    simplified_ddl = [
        f"{table}({', '.join(col for col, _ in cols)})"
        for table, cols in table_columns.items()
    ]

    full_ddl = [
        f"CREATE TABLE {table}({', '.join(f'{col} {typ}' for col, typ in cols)});"
        for table, cols in table_columns.items()
    ]

    fk_list = []
    for i, j in foreign_keys:
        src_table_idx, src_col = column_names[i]
        tgt_table_idx, tgt_col = column_names[j]
        src_table = table_names[src_table_idx]
        tgt_table = table_names[tgt_table_idx]
        fk_list.append(f"{src_table}({src_col}) REFERENCES {tgt_table}({tgt_col})")

    return json.dumps(simplified_ddl), json.dumps(full_ddl), json.dumps(fk_list)


def main():
    os.makedirs(os.path.dirname(SILVER_DB), exist_ok=True)

    conn_silver = sqlite3.connect(SILVER_DB)
    try:
        if not os.path.exists(SCHEMA_FILE):
            raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            conn_silver.executescript(f.read())
        cursor_silver = conn_silver.cursor()

        conn_bronze = sqlite3.connect(BRONZE_DB)
        try:
            cursor_bronze = conn_bronze.cursor()
            cursor_bronze.execute("""
                SELECT d.id, d.db_id, d.source, d.question, d.query, d.query_toks_no_value, d.sql_json, n.natsql
                FROM spider_dataset d
                LEFT JOIN spider_natsql n ON d.id = n.id AND d.source = n.source
            """)
            rows = cursor_bronze.fetchall()

            error_count = 0
            for row_id, db_id, source, question, query, query_toks_no_value, sql_json, natsql in rows:
                try:
                    cleaned_q = clean_question(question)
                    norm_sql = normalize_sql(query)
                    simplified_ddl, full_ddl, foreign_keys = get_schema_context(conn_bronze, db_id)
                    difficulty = get_query_difficulty(query, db_id)

                    cursor_silver.execute(
                        """
                        INSERT INTO silver_dataset (
                            id, db_id, source, question, query, query_toks_no_value, sql_json,
                            simplified_ddl, full_ddl, foreign_keys, difficulty, natsql
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row_id,
                            db_id,
                            source,
                            cleaned_q,
                            norm_sql,
                            query_toks_no_value,
                            sql_json,
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
        finally:
            conn_bronze.close()

        if error_count:
            print(f"⚠️ Completed with {error_count} row error(s). Committing successful rows.")
        conn_silver.commit()
    except Exception:
        conn_silver.rollback()
        raise
    finally:
        conn_silver.close()


if __name__ == "__main__":
    main()
