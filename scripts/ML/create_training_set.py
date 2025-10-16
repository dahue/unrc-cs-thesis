import sys
import os
import json
import sqlite3
import json
import re
import nltk
import spacy
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
load_dotenv()


try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

spacy.prefer_gpu()
model = "en_core_web_md"
try:
    nlp = spacy.load(model)
except OSError:
    print(f"Downloading {model}...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
    nlp = spacy.load(model)

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


def create_prompts(entries, template, query_type="sql", skeleton_dataset=None, max_workers=None):
    """
    Create prompts for all entries using parallel processing while maintaining order.
    
    Args:
        entries: List of database entries
        template: Jinja2 template object
        query_type: Type of query ("sql" or "natsql")
        skeleton_dataset: Optional skeleton dataset for few-shot learning
        max_workers: Maximum number of worker processes (default: min(32, os.cpu_count() + 4))
    
    Returns:
        List of prompts in the same order as input entries
    """
    if not entries:
        return []
    
    start_time = time.time()
    print(f"Creating prompts for {len(entries)} entries...")
    
    # For small datasets or when max_workers is 1, use sequential processing
    if max_workers == 1 or len(entries) < 10:
        prompts = []
        for entry in entries:
            params = {}
            params["query_type"] = query_type
            params["question"] = entry["question"]
            params["full_ddl"] = get_full_ddl(entry)
            params["simplified_ddl"] = get_simplified_ddl(entry)
            params["foreign_keys"] = get_foreign_keys(entry)
            params["cell_values"] = get_cell_values(entry)
            params["few_shot"] = get_few_shot(entry["question"], entry["simplified_ddl"], skeleton_dataset)
            prompt = template.render(params)
            prompts.append(prompt)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Prompts created in {elapsed_time:.2f} seconds (sequential processing)")
        return prompts
    
    # Use parallel processing for larger datasets
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    print(f"Using {max_workers} worker threads for parallel processing...")
    
    # Process entries in parallel using ThreadPoolExecutor for I/O bound tasks
    def process_entry(entry):
        params = {}
        params["query_type"] = query_type
        params["question"] = entry["question"]
        params["full_ddl"] = get_full_ddl(entry)
        params["simplified_ddl"] = get_simplified_ddl(entry)
        params["foreign_keys"] = get_foreign_keys(entry)
        params["cell_values"] = get_cell_values(entry)
        params["few_shot"] = get_few_shot(entry["question"], entry["simplified_ddl"], skeleton_dataset)
        prompt = template.render(params)
        return prompt
    
    prompts = [None] * len(entries)  # Pre-allocate list to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and keep track of their indices
        future_to_index = {
            executor.submit(process_entry, entry): i 
            for i, entry in enumerate(entries)
        }
        
        # Collect results as they complete, but maintain order
        completed_count = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                prompt = future.result()
                prompts[index] = prompt
                completed_count += 1
                if completed_count % 100 == 0:  # Progress indicator
                    print(f"  Processed {completed_count}/{len(entries)} entries...")
            except Exception as exc:
                print(f'Entry {index} generated an exception: {exc}')
                # Fallback to original processing for this entry
                entry = entries[index]
                params = {}
                params["query_type"] = query_type
                params["question"] = entry["question"]
                params["full_ddl"] = get_full_ddl(entry)
                params["simplified_ddl"] = get_simplified_ddl(entry)
                params["foreign_keys"] = get_foreign_keys(entry)
                params["cell_values"] = get_cell_values(entry)
                params["few_shot"] = get_few_shot(entry["question"], entry["simplified_ddl"], skeleton_dataset)
                prompt = template.render(params)
                prompts[index] = prompt
                completed_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"✓ Prompts created in {elapsed_time:.2f} seconds (parallel processing with {max_workers} workers)")
    return prompts

def create_completions(entries, query_type="sql"):
    completions = []
    for entry in entries:
        completions.append(entry["query"] if query_type == "sql" else entry[query_type])
    return completions

def create_dataset(entries, template, strategy, skeleton_dataset=None, max_workers=None):
    query_type = "sql" if strategy == "nl2SQL" else "natsql"
    processed_data = []
    prompts = create_prompts(entries, template, query_type, skeleton_dataset, max_workers)
    completions = create_completions(entries, query_type)
    for prompt, completion in zip(prompts, completions):
        processed_data.append({"prompt": prompt, "completion": completion})
    return processed_data

def create_sql_dataset(entries):
    processed_data = []
    for entry in entries:
        processed_data.append(f'{entry["query"]}\t{entry["db_id"]}')
    return processed_data

def get_question_skeleton(question, schema):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Parse the schema to extract table and column names
    try:
        schema_data = json.loads(schema)
    except json.JSONDecodeError:
        return question  # Return original if schema parsing fails
    
    # Extract all table names and column names from schema
    domain_tokens = set()
    
    for table_info in schema_data:
        # Extract table name (before the opening parenthesis)
        table_name = table_info.split('(')[0].strip()
        domain_tokens.add(table_name.lower())
        
        # Extract column names (inside parentheses)
        columns_part = table_info.split('(')[1].split(')')[0]
        columns = [col.strip().split()[0] for col in columns_part.split(',')]
        for col in columns:
            domain_tokens.add(col.lower())
    
    
    # Find and replace quoted strings (single or double quotes) with placeholders
    quoted_strings = []
    quote_pattern = r"'([^']*)'|\"([^\"]*)\""
    
    def replace_quoted(match):
        quoted_strings.append(match.group(0))
        return f"__QUOTED_STRING_{len(quoted_strings)-1}__"
    
    question_with_placeholders = re.sub(quote_pattern, replace_quoted, question)
    
    # Tokenize the question
    question_tokens = word_tokenize(question_with_placeholders.lower())
    
    # Create skeleton by replacing domain tokens with <mask>
    skeleton_tokens = []
    for token in question_tokens:
        # Check if it's a quoted string placeholder
        if token.startswith("__quoted_string_") and token.endswith("__"):
            skeleton_tokens.append('<mask>')
        
        # Check for numeric values
        elif token.isdigit() or (token.replace('.', '').replace(',', '').isdigit()):
            skeleton_tokens.append('<mask>')
        
        # Check for domain tokens (table/column names)
        else:
            lemmatized_token = lemmatizer.lemmatize(token)
            if lemmatized_token in domain_tokens or token in domain_tokens:
                skeleton_tokens.append('<mask>')
            else:
                skeleton_tokens.append(token)
    
    # Join tokens back into a sentence
    skeleton = ' '.join(skeleton_tokens)
    
    return skeleton

def create_skeleton_dataset(entries):
    processed_data = []
    for entry in entries:
        processed_data.append(f'{get_question_skeleton(entry["question"], entry["simplified_ddl"])}\t{entry["question"]}\t{entry["query"]}')
    return processed_data

def get_similar_skeletons(query_skeleton, train_skeleton_data, top_k=3):
    query_doc = nlp(query_skeleton)
    similarities = []
    for i, line in enumerate(train_skeleton_data):
        line = line.strip()
        skeleton_question, full_question, sql_query = line.split('\t', 2)
        similarity = query_doc.similarity(nlp(skeleton_question))
        similarities.append((i, similarity, skeleton_question, full_question, sql_query))
    
    # Sort by similarity (descending) and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def get_few_shot(question, schema, train_skeleton_data=None):
    few_shot = ""
    if train_skeleton_data is None:
        return few_shot
    else:
        skeleton_query = get_question_skeleton(question, schema)
        similar_skeletons = get_similar_skeletons(skeleton_query, train_skeleton_data, top_k=3)
        for i, (idx, score, skeleton_question, full_question, sql_query) in enumerate(similar_skeletons, 1):
            few_shot += f"{full_question}\n{sql_query}\n"
        return few_shot

def main(strategy, template_name, difficulties=None, test_limit=None, max_workers=None):
    """
    Create dataset for either nl2SQL or nl2NatSQL models using specified template.
    
    Args:
        strategy (str): Either 'nl2SQL' or 'nl2NatSQL'
        template_name (str): Template name (e.g., 'template_00' or 'template_00.j2')
        difficulties (list): Optional list of difficulty levels to filter by ('easy', 'medium', 'hard', 'extra')
        test_limit (int): Optional limit for number of test records to include
        max_workers (int): Optional maximum number of worker processes for parallel processing
    """
    script_start_time = time.time()
    print(f"Starting dataset creation for {strategy} with template {template_name}")
    if max_workers:
        print(f"Using {max_workers} worker threads for parallel processing")
    print("-" * 60)
    if strategy not in ['nl2SQL', 'nl2NatSQL']:
        raise ValueError("strategy must be either 'nl2SQL' or 'nl2NatSQL'")
    
    # Validate difficulties parameter
    valid_difficulties = ['easy', 'medium', 'hard', 'extra']
    if difficulties is not None:
        if not isinstance(difficulties, list):
            raise ValueError("difficulties must be a list")
        for difficulty in difficulties:
            if difficulty not in valid_difficulties:
                raise ValueError(f"Invalid difficulty '{difficulty}'. Must be one of: {valid_difficulties}")
        print(f"Filtering dataset by difficulties: {', '.join(difficulties)}")
    
    # Validate test_limit parameter
    if test_limit is not None:
        if not isinstance(test_limit, int) or test_limit <= 0:
            raise ValueError("test_limit must be a positive integer")
        print(f"Limiting test dataset to {test_limit} records")
    
    # Validate max_workers parameter
    if max_workers is not None:
        if not isinstance(max_workers, int) or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        print(f"Using {max_workers} worker processes for parallel processing")

    # Connect to gold database
    conn_gold = sqlite3.connect(GOLD_DB)
    cursor_gold = conn_gold.cursor()

    # Get table columns
    table_columns = cursor_gold.execute("PRAGMA table_info(gold_dataset)").fetchall()
    columns = [column[1] for column in table_columns]

    # Build difficulty filter clause if difficulties are specified
    difficulty_clause = ""
    if difficulties is not None:
        placeholders = ",".join(["?" for _ in difficulties])
        difficulty_clause = f" AND difficulty IN ({placeholders})"
    
    # Get train, valid, and test entries
    train_query = f"SELECT {', '.join(columns)} FROM gold_dataset WHERE source = 'train'{difficulty_clause}"
    if difficulties is not None:
        cursor_gold.execute(train_query, difficulties)
    else:
        cursor_gold.execute(train_query)
    train_rows = cursor_gold.fetchall()
    train_entries = [dict(zip(columns, row)) for row in train_rows]

    valid_query = f"SELECT {', '.join(columns)} FROM gold_dataset WHERE source = 'dev'{difficulty_clause}"
    if difficulties is not None:
        cursor_gold.execute(valid_query, difficulties)
    else:
        cursor_gold.execute(valid_query)
    valid_rows = cursor_gold.fetchall()
    valid_entries = [dict(zip(columns, row)) for row in valid_rows]

    test_query = f"SELECT {', '.join(columns)} FROM gold_dataset WHERE source = 'test'{difficulty_clause}"
    if difficulties is not None:
        cursor_gold.execute(test_query, difficulties)
    else:
        cursor_gold.execute(test_query)
    test_rows = cursor_gold.fetchall()
    test_entries = [dict(zip(columns, row)) for row in test_rows]

    # Ensure template name has .j2 extension for loading
    template_file = template_name if template_name.endswith('.j2') else f"{template_name}.j2"
    
    # Load template
    env = Environment(loader=FileSystemLoader(f'{ROOT_PATH}/data/templates/{strategy}'))
    template = env.get_template(template_file)

    # Apply test_limit if specified
    if test_limit is not None:
        test_entries = test_entries[:test_limit]

    # Create datasets
    print(f"Creating skeleton dataset from {len(train_entries)} training entries...")
    skeleton_start_time = time.time()
    test_skeleton_data = create_skeleton_dataset(train_entries)
    skeleton_time = time.time() - skeleton_start_time
    print(f"✓ Skeleton dataset created in {skeleton_time:.2f} seconds")

    print(f"\nCreating training dataset ({len(train_entries)} entries)...")
    train_data = create_dataset(train_entries, template, strategy=strategy, max_workers=max_workers)
    
    print(f"\nCreating validation dataset ({len(valid_entries)} entries)...")
    valid_data = create_dataset(valid_entries, template, strategy=strategy, max_workers=max_workers)
    
    print(f"\nCreating test dataset ({len(test_entries)} entries)...")
    test_data = create_dataset(test_entries, template, strategy=strategy, skeleton_dataset=test_skeleton_data, max_workers=max_workers)

    train_sql_data = create_sql_dataset(train_entries)
    valid_sql_data = create_sql_dataset(valid_entries)
    test_sql_data = create_sql_dataset(test_entries)

    # Write to JSONL files
    print(f"\nWriting datasets to files...")
    write_start_time = time.time()
    
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

    folder_prefix = f"{ROOT_PATH}/data/training/{strategy}/{template_name.removesuffix('.j2')}/"
    write_jsonl(train_data, folder_prefix+'train.jsonl')
    write_jsonl(test_data, folder_prefix+'test.jsonl')
    write_jsonl(valid_data, folder_prefix+'valid.jsonl')

    write_sql(train_sql_data, folder_prefix+'train.sql')
    write_sql(test_sql_data, folder_prefix+'test.sql')
    write_sql(valid_sql_data, folder_prefix+'valid.sql')

    write_time = time.time() - write_start_time
    print(f"✓ Files written in {write_time:.2f} seconds")

    # Close database connection
    conn_gold.close()

    # Final timing summary
    total_time = time.time() - script_start_time
    print("-" * 60)
    print(f"✓ Dataset creation completed successfully!")
    print(f"  - Training: {len(train_data)} entries")
    print(f"  - Validation: {len(valid_data)} entries") 
    print(f"  - Test: {len(test_data)} entries")
    print(f"  - Total time: {total_time:.2f} seconds")
    print(f"  - Saved to: {folder_prefix}")
    print("-" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create dataset for nl2SQL or nl2NatSQL')
    parser.add_argument('--strategy', type=str, required=True, choices=['nl2SQL', 'nl2NatSQL'],
                       help='Type of model to create dataset for')
    parser.add_argument('--template', type=str, required=True,
                       help='Name of the template file without .j2 extension')
    parser.add_argument('--difficulty', type=str, nargs='*', choices=['easy', 'medium', 'hard', 'extra'],
                       help='Optional list of difficulty levels to filter by. Can specify multiple values.')
    parser.add_argument('--test-limit', type=int, metavar='N',
                       help='Optional limit for number of test records to include (takes first N records)')
    parser.add_argument('--max-workers', type=int, metavar='N',
                       help='Optional maximum number of worker processes for parallel processing (default: min(32, cpu_count + 4))')
    args = parser.parse_args()
    main(args.strategy, args.template, args.difficulty, args.test_limit, args.max_workers)
