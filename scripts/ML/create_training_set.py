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
import hashlib
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
CACHE_DIR = f"{ROOT_PATH}/data/cache/few_shot"

def generate_cache_key(difficulties, test_limit):
    """
    Generate a cache key based on difficulty and test_limit parameters.
    
    Args:
        difficulties: List of difficulty levels or None
        test_limit: Integer limit or None
    
    Returns:
        String cache key
    """
    # Sort difficulties to ensure consistent key generation
    difficulty_str = "_".join(sorted(difficulties)) if difficulties else "all"
    test_limit_str = str(test_limit) if test_limit else "all"
    key_string = f"difficulty_{difficulty_str}_testlimit_{test_limit_str}"
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cache_path(cache_key):
    """Get the full path to the cache file for a given cache key."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"few_shot_{cache_key}.json")

def load_few_shot_cache(cache_key, test_entries):
    """
    Load few-shot examples from cache if available and valid.
    
    Args:
        cache_key: Cache key string
        test_entries: List of test entries to validate cache against
    
    Returns:
        List of few-shot strings (one per test entry) or None if cache invalid/missing
    """
    cache_path = get_cache_path(cache_key)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Validate cache: check if it matches the current test entries
        cached_entries = cache_data.get("entries", [])
        if len(cached_entries) != len(test_entries):
            print(f"Cache invalid: entry count mismatch ({len(cached_entries)} vs {len(test_entries)})")
            return None
        
        # Validate by checking question + schema for each entry
        for i, (cached_entry, test_entry) in enumerate(zip(cached_entries, test_entries)):
            if cached_entry.get("question") != test_entry["question"] or \
               cached_entry.get("simplified_ddl") != test_entry["simplified_ddl"]:
                print(f"Cache invalid: entry {i} mismatch")
                return None
        
        print(f"✓ Loaded few-shot cache from {cache_path}")
        return cache_data.get("few_shots", [])
    
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error loading cache: {e}")
        return None

def save_few_shot_cache(cache_key, test_entries, few_shots):
    """
    Save few-shot examples to cache.
    
    Args:
        cache_key: Cache key string
        test_entries: List of test entries
        few_shots: List of few-shot strings (one per test entry)
    """
    cache_path = get_cache_path(cache_key)
    
    # Store entries metadata for validation
    entries_metadata = [
        {"question": entry["question"], "simplified_ddl": entry["simplified_ddl"]}
        for entry in test_entries
    ]
    
    cache_data = {
        "entries": entries_metadata,
        "few_shots": few_shots
    }
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"✓ Saved few-shot cache to {cache_path}")
    except Exception as e:
        print(f"Error saving cache: {e}")

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


def create_prompts(entries, template, query_type="sql", skeleton_dataset=None, max_workers=None, few_shots_list=None):
    """
    Create prompts for all entries using parallel processing while maintaining order.
    
    Args:
        entries: List of database entries
        template: Jinja2 template object
        query_type: Type of query ("sql" or "natsql")
        skeleton_dataset: Optional skeleton dataset for few-shot learning
        max_workers: Maximum number of worker processes (default: min(32, os.cpu_count() + 4))
        few_shots_list: Optional pre-computed list of few-shot examples (one per entry)
    
    Returns:
        List of prompts in the same order as input entries
    """
    if not entries:
        return []
    
    # Validate few_shots_list length if provided
    if few_shots_list is not None and len(few_shots_list) != len(entries):
        raise ValueError(f"few_shots_list length ({len(few_shots_list)}) must match entries length ({len(entries)})")
    
    start_time = time.time()
    print(f"Creating prompts for {len(entries)} entries...")
    
    # For small datasets or when max_workers is 1, use sequential processing
    if max_workers == 1 or len(entries) < 10:
        prompts = []
        for i, entry in enumerate(entries):
            params = {}
            params["query_type"] = query_type
            params["question"] = entry["question"]
            params["full_ddl"] = get_full_ddl(entry)
            params["simplified_ddl"] = get_simplified_ddl(entry)
            params["foreign_keys"] = get_foreign_keys(entry)
            params["cell_values"] = get_cell_values(entry)
            if few_shots_list is not None:
                params["few_shot"] = few_shots_list[i]
            else:
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
    def process_entry(entry_index):
        entry = entries[entry_index]
        params = {}
        params["query_type"] = query_type
        params["question"] = entry["question"]
        params["full_ddl"] = get_full_ddl(entry)
        params["simplified_ddl"] = get_simplified_ddl(entry)
        params["foreign_keys"] = get_foreign_keys(entry)
        params["cell_values"] = get_cell_values(entry)
        if few_shots_list is not None:
            params["few_shot"] = few_shots_list[entry_index]
        else:
            params["few_shot"] = get_few_shot(entry["question"], entry["simplified_ddl"], skeleton_dataset)
        prompt = template.render(params)
        return prompt
    
    prompts = [None] * len(entries)  # Pre-allocate list to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and keep track of their indices
        future_to_index = {
            executor.submit(process_entry, i): i 
            for i in range(len(entries))
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
                if few_shots_list is not None:
                    params["few_shot"] = few_shots_list[index]
                else:
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

def create_dataset(
    entries,
    template,
    strategy,
    skeleton_dataset=None,
    max_workers=None,
    few_shots_list=None,
    include_metadata: bool = False,
):
    query_type = "sql" if strategy == "nl2SQL" else "natsql"
    processed_data = []
    prompts = create_prompts(entries, template, query_type, skeleton_dataset, max_workers, few_shots_list)
    completions = create_completions(entries, query_type)
    for entry, prompt, completion in zip(entries, prompts, completions):
        item = {"prompt": prompt, "completion": completion}
        if include_metadata:
            # Include minimal metadata needed for downstream schema refinement and evaluation.
            # Store rendered schema blocks (same representation used in prompts).
            item.update(
                {
                    "question": entry.get("question"),
                    "db_id": entry.get("db_id"),
                    "simplified_ddl": get_simplified_ddl(entry),
                    "foreign_keys": get_foreign_keys(entry),
                }
            )
        processed_data.append(item)
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

def compute_few_shots_for_entries(entries, train_skeleton_data, max_workers=None):
    """
    Compute few-shot examples for all entries.
    
    Args:
        entries: List of database entries
        train_skeleton_data: Training skeleton dataset for few-shot learning
        max_workers: Maximum number of worker processes
    
    Returns:
        List of few-shot strings (one per entry)
    """
    if train_skeleton_data is None:
        return [""] * len(entries)
    
    start_time = time.time()
    print(f"Computing few-shot examples for {len(entries)} entries...")
    
    # For small datasets or when max_workers is 1, use sequential processing
    if max_workers == 1 or len(entries) < 10:
        few_shots = []
        for entry in entries:
            few_shot = get_few_shot(entry["question"], entry["simplified_ddl"], train_skeleton_data)
            few_shots.append(few_shot)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Few-shot examples computed in {elapsed_time:.2f} seconds (sequential processing)")
        return few_shots
    
    # Use parallel processing for larger datasets
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    print(f"Using {max_workers} worker threads for parallel few-shot computation...")
    
    def compute_few_shot(entry_index):
        entry = entries[entry_index]
        return get_few_shot(entry["question"], entry["simplified_ddl"], train_skeleton_data)
    
    few_shots = [None] * len(entries)  # Pre-allocate list to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and keep track of their indices
        future_to_index = {
            executor.submit(compute_few_shot, i): i 
            for i in range(len(entries))
        }
        
        # Collect results as they complete, but maintain order
        completed_count = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                few_shot = future.result()
                few_shots[index] = few_shot
                completed_count += 1
                if completed_count % 100 == 0:  # Progress indicator
                    print(f"  Computed {completed_count}/{len(entries)} few-shot examples...")
            except Exception as exc:
                print(f'Entry {index} generated an exception: {exc}')
                # Fallback to original processing for this entry
                entry = entries[index]
                few_shot = get_few_shot(entry["question"], entry["simplified_ddl"], train_skeleton_data)
                few_shots[index] = few_shot
                completed_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"✓ Few-shot examples computed in {elapsed_time:.2f} seconds (parallel processing with {max_workers} workers)")
    return few_shots

def main(strategy, template_name, difficulties=None, test_limit=None, max_workers=None, include_metadata: bool = False):
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
    if include_metadata:
        print("Including metadata fields in JSONL outputs (question, db_id, simplified_ddl, foreign_keys)")
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
    train_data = create_dataset(
        train_entries, template, strategy=strategy, max_workers=max_workers, include_metadata=include_metadata
    )
    
    print(f"\nCreating validation dataset ({len(valid_entries)} entries)...")
    valid_data = create_dataset(
        valid_entries, template, strategy=strategy, max_workers=max_workers, include_metadata=include_metadata
    )
    
    # Handle few-shot caching for test dataset
    print(f"\nCreating test dataset ({len(test_entries)} entries)...")
    cache_key = generate_cache_key(difficulties, test_limit)
    cached_few_shots = load_few_shot_cache(cache_key, test_entries)
    
    if cached_few_shots is None:
        # Compute few-shots and cache them
        few_shots_list = compute_few_shots_for_entries(test_entries, test_skeleton_data, max_workers)
        save_few_shot_cache(cache_key, test_entries, few_shots_list)
    else:
        few_shots_list = cached_few_shots
    
    test_data = create_dataset(
        test_entries,
        template,
        strategy=strategy,
        max_workers=max_workers,
        few_shots_list=few_shots_list,
        include_metadata=include_metadata,
    )

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
    parser.add_argument(
        '--include-metadata',
        action='store_true',
        help='Include extra fields (question, db_id, simplified_ddl, foreign_keys) in JSONL outputs for downstream processing.',
    )
    args = parser.parse_args()
    main(args.strategy, args.template, args.difficulty, args.test_limit, args.max_workers, args.include_metadata)
