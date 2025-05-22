import os
import json
import sqlite3
import string
import pandas as pd

ROOT_PATH = "/Users/atissera/Developer/repos/unrc-cs-thesis"
GOLD_DB = f"{ROOT_PATH}/database/gold/gold.sqlite"
MODEL_TYPE = 'nl2SQL'

def get_schema_ddl(entry):
    schema_ddl = json.loads(entry["simplified_ddl"])
    formatted_schema_ddl = []
    for table in schema_ddl:
        formatted_schema_ddl.append(f"# {table}")
    return "\n".join(formatted_schema_ddl)

def get_cell_values_sample(entry, max_samples=3):
    db_id = entry["db_id"]
    db_path = os.path.join(ROOT_PATH, 'tmp', 'spider_data', 'database', db_id, db_id + '.sqlite')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    formatted_tables = []

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [row[1] for row in cursor.fetchall()]  # row[1] is column name

        cursor.execute(f"SELECT * FROM {table} LIMIT {max_samples};")
        rows = cursor.fetchall()

        col_samples = list(zip(*rows)) if rows else [[] for _ in columns]

        col_strs = []
        for col, vals in zip(columns, col_samples):
            val_list = ", ".join(str(v) for v in vals[:max_samples])
            col_strs.append(f"{col}[{val_list}]")
        formatted = f"# {table}(" + ", ".join(col_strs) + ")"
        formatted_tables.append(formatted)

    conn.close()
    return "\n".join(formatted_tables)

def get_foreign_keys(entry):
    foreign_keys = json.loads(entry["foreign_keys"])
    formatted_foreign_keys = []
    for fk in foreign_keys:
        formatted_foreign_keys.append(f"# {fk}")
    return "\n".join(formatted_foreign_keys)

def create_prompts(entries, template, template_variables='None'):
    prompts = []
    for entry in entries:
        variables = {
            "schema_ddl": get_schema_ddl(entry),
            "data_samples": get_cell_values_sample(entry),
            "foreign_keys": get_foreign_keys(entry),
            "question": entry["question"],
        }
        prompts.append(template.format(**variables))
    return prompts

def create_completions(entries, completion_column="query"):
    completions = []
    for entry in entries:
        completions.append(entry[completion_column])
    return completions

def create_dataset(entries, template, template_variables='None', completion_column="query"):
    processed_data = []
    prompts = create_prompts(entries, template)
    completions = create_completions(entries, completion_column)
    for prompt, completion in zip(prompts, completions):
        processed_data.append({"prompt": prompt, "completion": completion})
    return processed_data


conn_gold = sqlite3.connect(GOLD_DB)
cursor_gold = conn_gold.cursor()

table_columns = cursor_gold.execute("PRAGMA table_info(gold_dataset)").fetchall()
columns = [column[1] for column in table_columns]

cursor_gold.execute(f"SELECT {', '.join(columns)} FROM gold_dataset WHERE source = 'train'")
train_rows = cursor_gold.fetchall()
train_entries = [dict(zip(columns, row)) for row in train_rows]

cursor_gold.execute(f"SELECT {', '.join(columns)} FROM gold_dataset WHERE source = 'dev'")
valid_rows = cursor_gold.fetchall()
valid_entries = [dict(zip(columns, row)) for row in valid_rows]

cursor_gold.execute(f"SELECT {', '.join(columns)} FROM gold_dataset WHERE source = 'test'")
test_rows = cursor_gold.fetchall()
test_entries = [dict(zip(columns, row)) for row in test_rows]


# Load Prompt Template
# nl2SQL
with open(os.path.join(ROOT_PATH, "data", "template", MODEL_TYPE, "prompt_template.md"), "r", encoding="utf-8") as f:
    template = f.read()

formatter = string.Formatter()
template_variables = [field_name for _, field_name, _, _ in formatter.parse(template) if field_name]


train_data = create_dataset(train_entries, template, completion_column="query")
valid_data = create_dataset(valid_entries, template, completion_column="query")
test_data = create_dataset(test_entries, template, completion_column="query")


# Write to JSONL files
def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

folder_prefix = f"{ROOT_PATH}/data/training/{MODEL_TYPE}/"
write_jsonl(train_data, folder_prefix+'train.jsonl')
write_jsonl(test_data, folder_prefix+'test.jsonl')
write_jsonl(valid_data, folder_prefix+'valid.jsonl')

print(f"Dataset split and saved: train ({len(train_data)}), test ({len(test_data)}), valid ({len(valid_data)})")

# Verify file contents
def count_lines(filename):
    with open(folder_prefix+filename, 'r') as f:
        return sum(1 for _ in f)

print("\nVerifying file contents:")
print(f"train.jsonl: {count_lines('train.jsonl')} lines")
print(f"test.jsonl: {count_lines('test.jsonl')} lines")
print(f"valid.jsonl: {count_lines('valid.jsonl')} lines")