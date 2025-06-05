import os
import json
import sqlite3
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ["ROOT_PATH"]
SPIDER_DB_PATH = f"{ROOT_PATH}/database/spider"

GOLD_DB = f"{ROOT_PATH}/database/gold/gold.sqlite"

def get_full_ddl(entry):
    full_ddl = json.loads(entry["full_ddl"])
    formatted_full_ddl = []
    for table in full_ddl:
        formatted_full_ddl.append(table)
    return "\n".join(formatted_full_ddl)

def get_simplified_ddl(entry):
    simplified_ddl = json.loads(entry["simplified_ddl"])
    formatted_simplified_ddl = []
    for table in simplified_ddl:
        formatted_simplified_ddl.append(table)
    return "\n".join(formatted_simplified_ddl)

def get_cell_values(entry, max_samples=3):
    db_id = entry["db_id"]
    db_path = os.path.join(SPIDER_DB_PATH, db_id, db_id + '.sqlite')
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
        formatted = f"{table}(" + ", ".join(col_strs) + ")"
        formatted_tables.append(formatted)

    conn.close()
    return "\n".join(formatted_tables)

def get_foreign_keys(entry):
    foreign_keys = json.loads(entry["foreign_keys"])
    formatted_foreign_keys = []
    for fk in foreign_keys:
        formatted_foreign_keys.append(fk)
    return "\n".join(formatted_foreign_keys)

def get_few_shot():
    few_shot = """How many farms are there?\nSELECT count(*) FROM farm\nWhat is the average, minimum, and maximum age for all French singers?\nSELECT avg(age), min(age), max(age) FROM singer WHERE country = 'France'\nShow the ID of the high schooler named Kyle.\nSELECT ID FROM Highschooler WHERE name = 'Kyle'"""
    return few_shot

def create_prompts(entries, template, query_type="sql"):
    params = {}
    prompts = []
    for entry in entries:
        params["query_type"] = query_type
        params["question"] = entry["question"]
        params["full_ddl"] = get_full_ddl(entry)
        params["simplified_ddl"] = get_simplified_ddl(entry)
        params["foreign_keys"] = get_foreign_keys(entry)
        params["cell_values"] = get_cell_values(entry)
        params["few_shot"] = get_few_shot()
        prompt = template.render(params)
        prompts.append(prompt)
    return prompts

def create_completions(entries, query_type="sql"):
    completions = []
    for entry in entries:
        completions.append(entry["query"] if query_type == "sql" else entry[query_type])
    return completions

def create_dataset(entries, template, model_type):
    query_type = "sql" if model_type == "nl2SQL" else "natsql"
    processed_data = []
    prompts = create_prompts(entries, template, query_type)
    completions = create_completions(entries, query_type)
    for prompt, completion in zip(prompts, completions):
        processed_data.append({"prompt": prompt, "completion": completion})
    return processed_data

def create_sql_dataset(entries):
    processed_data = []
    for entry in entries:
        processed_data.append(f'{entry["query"]}\t{entry["db_id"]}')
    return processed_data

def main(model_type, template_name):
    """
    Create dataset for either nl2SQL or nl2NatSQL models using specified template.
    
    Args:
        model_type (str): Either 'nl2SQL' or 'nl2NatSQL'
        template_name (str): Template name (e.g., 'template_00.j2')
    """
    if model_type not in ['nl2SQL', 'nl2NatSQL']:
        raise ValueError("model_type must be either 'nl2SQL' or 'nl2NatSQL'")

    # Connect to gold database
    conn_gold = sqlite3.connect(GOLD_DB)
    cursor_gold = conn_gold.cursor()

    # Get table columns
    table_columns = cursor_gold.execute("PRAGMA table_info(gold_dataset)").fetchall()
    columns = [column[1] for column in table_columns]

    # Get train, valid, and test entries
    cursor_gold.execute(f"SELECT {', '.join(columns)} FROM gold_dataset WHERE source = 'train'")
    train_rows = cursor_gold.fetchall()
    train_entries = [dict(zip(columns, row)) for row in train_rows]

    cursor_gold.execute(f"SELECT {', '.join(columns)} FROM gold_dataset WHERE source = 'dev'")
    valid_rows = cursor_gold.fetchall()
    valid_entries = [dict(zip(columns, row)) for row in valid_rows]

    cursor_gold.execute(f"SELECT {', '.join(columns)} FROM gold_dataset WHERE source = 'test'")
    test_rows = cursor_gold.fetchall()
    test_entries = [dict(zip(columns, row)) for row in test_rows]

    # Load template
    env = Environment(loader=FileSystemLoader(f'{ROOT_PATH}/data/templates/{model_type}'))
    template = env.get_template(template_name)

    # Create datasets
    train_data = create_dataset(train_entries, template, model_type=model_type)
    valid_data = create_dataset(valid_entries, template, model_type=model_type)
    test_data = create_dataset(test_entries, template, model_type=model_type)

    train_sql_data = create_sql_dataset(train_entries)
    valid_sql_data = create_sql_dataset(valid_entries)
    test_sql_data = create_sql_dataset(test_entries)

    # Write to JSONL files
    def write_jsonl(data, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    def write_sql(data, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            for query in data:
                f.write(query + "\n")

    folder_prefix = f"{ROOT_PATH}/data/training/{model_type}/{template_name.strip('.j2')}/"
    write_jsonl(train_data, folder_prefix+'train.jsonl')
    write_jsonl(test_data, folder_prefix+'test.jsonl')
    write_jsonl(valid_data, folder_prefix+'valid.jsonl')

    write_sql(train_sql_data, folder_prefix+'train.sql')
    write_sql(test_sql_data, folder_prefix+'test.sql')
    write_sql(valid_sql_data, folder_prefix+'valid.sql')

    # Close database connection
    conn_gold.close()

    print(f"Dataset split and saved to {folder_prefix}: train ({len(train_data)} lines), test ({len(test_data)} lines), valid ({len(valid_data)} lines)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create dataset for nl2SQL or nl2NatSQL')
    parser.add_argument('--model-type', type=str, required=True, choices=['nl2SQL', 'nl2NatSQL'],
                       help='Type of model to create dataset for')
    parser.add_argument('--template-name', type=str, required=True,
                       help='Name of the template file')
    args = parser.parse_args()
    main(args.model_type, args.template_name)
