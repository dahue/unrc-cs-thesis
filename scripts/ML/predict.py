import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import mlx.core as mx
from mlx_lm import load, batch_generate, generate
from mlx_lm.sample_utils import make_sampler
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from scripts.ML.schema_refinement import refine_schema_from_sql

load_dotenv()
ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

PRED_PATH = f"{ROOT_PATH}/data/predictions"


def parse_model_spec(model_spec: str) -> tuple[str, bool]:
    """
    Parse a model specification string.
    
    Args:
        model_spec: Model specification in format "model_name" or "model_name:fine-tuned"
        
    Returns:
        Tuple of (model_name, use_adapter)
        
    Raises:
        ValueError: If model_spec is empty or has invalid format
    """
    if not model_spec or not model_spec.strip():
        raise ValueError("Model specification cannot be empty")
    
    model_spec = model_spec.strip()
    
    # Split by colon to check for :fine-tuned suffix
    parts = model_spec.split(':')
    
    if len(parts) == 1:
        # No colon, base model
        return parts[0], False
    elif len(parts) == 2:
        # Has colon, check if second part is "fine-tuned"
        model_name = parts[0].strip()
        suffix = parts[1].strip().lower()
        
        if not model_name:
            raise ValueError(f"Model name cannot be empty in specification: {model_spec}")
        
        if suffix == "fine-tuned":
            return model_name, True
        else:
            # Has colon but not :fine-tuned, treat as base model but warn?
            # Actually, let's be strict - if there's a colon, it must be :fine-tuned
            raise ValueError(f"Invalid model specification suffix: '{suffix}'. Expected 'fine-tuned' or no suffix.")
    else:
        # Multiple colons - invalid
        raise ValueError(f"Invalid model specification format: {model_spec}. Use 'model_name' or 'model_name:fine-tuned'")


def load_model(model_path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path: str = None):
    """Load the MLX model and tokenizer, optionally with LoRA adapter"""
    print(f"Loading model: {model_path}")
    
    if adapter_path:
        print(f"Loading with adapter: {adapter_path}")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        print("Model and adapter loaded successfully!")
    else:
        model, tokenizer = load(model_path)
        print("Model loaded successfully!")
    
    return model, tokenizer

def process_batch(
    prompts: List[str], 
    model, 
    tokenizer,
    max_tokens: int = 512,
    batch_size: int = None
) -> List[Dict[str, Any]]:
    """
    Process a batch of prompts using MLX batch_generate for improved efficiency.
    Maintains the same interface and output format as the original function.
    
    Args:
        prompts: List of prompt strings to process
        model: Loaded MLX model
        tokenizer: Loaded MLX tokenizer
        max_tokens: Maximum tokens to generate per prompt
        batch_size: Maximum number of prompts to process in a single batch.
                   If None, processes all prompts at once. Use smaller values
                   to avoid memory issues.
        
    Returns:
        List of dictionaries containing results for each prompt
    """
    # Set default batch size if not specified
    if batch_size is None:
        batch_size = len(prompts)
    
    print(f"Processing {len(prompts)} prompts using MLX batch_generate (batch_size={batch_size})")
    
    start_time = time.time()
    all_results = []
    
    # Process prompts in chunks
    for chunk_start in range(0, len(prompts), batch_size):
        chunk_end = min(chunk_start + batch_size, len(prompts))
        chunk_prompts = prompts[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_start//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size} "
              f"(prompts {chunk_start+1}-{chunk_end})")
        
        try:
            # Apply chat template to prompts in this chunk
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                for prompt in chunk_prompts
            ]
            
            # Use batch_generate for efficient processing of this chunk
            batch_result = batch_generate(
                model, 
                tokenizer, 
                formatted_prompts, 
                verbose=False, 
                max_tokens=max_tokens
            )
            
            # Process results for this chunk
            chunk_results = []
            for i, (prompt, response_text) in enumerate(zip(chunk_prompts, batch_result.texts)):
                # Normalize the response using the existing function
                normalized_response = normalize_response(response_text)
                
                result = {
                    "prompt_index": chunk_start + i,  # Global index
                    "prompt": prompt,
                    "response": normalized_response,
                    "generation_time": 0,  # Will be calculated below
                    "status": "success"
                }
                chunk_results.append(result)
            
            all_results.extend(chunk_results)
            print(f"Chunk completed successfully")
            
        except Exception as e:
            print(f"Error processing chunk {chunk_start//batch_size + 1}: {e}")
            # If chunk processing fails, create error results for all prompts in this chunk
            for i, prompt in enumerate(chunk_prompts):
                result = {
                    "prompt_index": chunk_start + i,
                    "prompt": prompt,
                    "response": None,
                    "generation_time": 0,
                    "status": "error",
                    "error": str(e)
                }
                all_results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Update generation_time for each result (approximate per-prompt time)
    if all_results:
        avg_time_per_prompt = total_time / len(prompts)
        for result in all_results:
            result["generation_time"] = avg_time_per_prompt
    
    print(f"Batch processing completed in {total_time:.2f}s")
    if prompts:
        print(f"Average time per prompt: {total_time/len(prompts):.2f}s")

    return all_results

def process_batch_with_sampling(
    prompts: List[str], 
    model, 
    tokenizer,
    max_tokens: int = 512,
    batch_size: int = None,
    num_samples: int = 1,
    temperature: float = 0.0
) -> List[List[Dict[str, Any]]]:
    """
    Process a batch of prompts with multiple samples per prompt using different seeds.
    
    Args:
        prompts: List of prompt strings to process
        model: Loaded MLX model
        tokenizer: Loaded MLX tokenizer
        max_tokens: Maximum tokens to generate per prompt
        batch_size: Maximum number of prompts to process in a single batch
        num_samples: Number of samples to generate per prompt
        temperature: Temperature for sampling (0.0 = deterministic, >0.0 = stochastic)
        
    Returns:
        List where each element is a list of sample results for one prompt
    """
    print(f"Processing {len(prompts)} prompts with {num_samples} samples each (temperature={temperature})")
    
    # Set default batch size if not specified
    if batch_size is None:
        batch_size = len(prompts)
    
    all_samples_per_prompt = []
    
    # Generate samples for each prompt
    for prompt_idx, prompt in enumerate(prompts):
        print(f"Processing prompt {prompt_idx + 1}/{len(prompts)} with {num_samples} samples")
        prompt_samples = []
        
        # Generate multiple samples with different seeds
        for sample_idx in range(num_samples):
            seed = 42 + sample_idx  # Different seed for each sample
            
            try:
                # Apply chat template
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                
                # Generate with specific seed and temperature
                mx.random.seed(seed)
                sampler = make_sampler(temp=temperature)
                response = generate(
                    model,
                    tokenizer,
                    formatted_prompt,
                    verbose=False,
                    max_tokens=max_tokens,
                    sampler=sampler
                )
                
                # Normalize the response
                normalized_response = normalize_response(response)
                
                result = {
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "response": normalized_response,
                    "generation_time": 0,  # Individual timing not tracked
                    "status": "success",
                    "sample_idx": sample_idx,
                    "seed": seed
                }
                prompt_samples.append(result)
                
            except Exception as e:
                print(f"Error generating sample {sample_idx} for prompt {prompt_idx}: {e}")
                result = {
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "response": None,
                    "generation_time": 0,
                    "status": "error",
                    "error": str(e),
                    "sample_idx": sample_idx,
                    "seed": seed
                }
                prompt_samples.append(result)
        
        all_samples_per_prompt.append(prompt_samples)
    
    return all_samples_per_prompt

def process_self_consistent(
    prompts: List[str], 
    model, 
    tokenizer,
    max_tokens: int = 512,
    batch_size: int = None,
    num_samples: int = 5,
    temperature: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Process prompts with self-consistency (same model, multiple samples).
    
    Args:
        prompts: List of prompt strings to process
        model: Loaded MLX model
        tokenizer: Loaded MLX tokenizer
        max_tokens: Maximum tokens to generate per prompt
        batch_size: Maximum number of prompts to process in a single batch
        num_samples: Number of samples per prompt
        temperature: Temperature for sampling
        
    Returns:
        List of aggregated results, one per prompt
    """
    print(f"Starting self-consistency processing with {num_samples} samples per prompt")
    
    # Generate multiple samples for each prompt
    samples_per_prompt = process_batch_with_sampling(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        batch_size=batch_size,
        num_samples=num_samples,
        temperature=temperature
    )
    
    # Aggregate results using majority voting
    aggregated_results = aggregate_results_majority_vote(samples_per_prompt)
    
    # Update consistency mode
    for result in aggregated_results:
        result["consistency_mode"] = "self"
    
    return aggregated_results

def process_cross_consistent(
    prompts: List[str],
    model_specs: List[str],
    template: str,
    max_tokens: int = 512,
    batch_size: int = None,
    temperature: float = 0.0,
    input_file: str = None,
    db_ids: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process prompts with cross-consistency (multiple models, one sample each).
    Uses semantic comparison (execution results) for aggregation.
    
    Args:
        prompts: List of prompt strings to process
        model_specs: List of model specifications (can include :fine-tuned suffix)
        template: Template name (used for adapter paths)
        max_tokens: Maximum tokens to generate per prompt
        batch_size: Maximum number of prompts to process in a single batch
        temperature: Temperature for generation (can be 0.0 for deterministic)
        input_file: Path to the input JSONL file (used to load database IDs if db_ids is not provided)
        db_ids: Optional list of database IDs corresponding to each prompt
        
    Returns:
        List of aggregated results, one per prompt
    """
    import gc
    
    print(f"Starting cross-consistency processing with {len(model_specs)} models")
    
    template_folder = template.removesuffix('.j2')

    # Store results from each model
    all_model_results = []
    model_names_list = []
    
    # Process each model sequentially
    for model_idx, model_spec in enumerate(model_specs):
        print(f"\n{'='*60}")
        print(f"Processing model {model_idx + 1}/{len(model_specs)}: {model_spec}")
        print(f"{'='*60}")
        
        # Parse model spec and load model
        model_name, use_adapter = parse_model_spec(model_spec)
        model_names_list.append(model_name)
        
        adapter_path = None
        if use_adapter:
            adapter_path = f"{ROOT_PATH}/data/adapters/nl2SQL/{template_folder}/{get_model_dir_name(model_name)}"
        
        model, tokenizer = load_model(model_name, adapter_path)
        
        # Process all prompts with this model
        model_results = process_batch(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            batch_size=batch_size
        )
        
        # Add model identifier to each result
        for result in model_results:
            result['model_name'] = model_name
            result['model_spec'] = model_spec
        
        all_model_results.append(model_results)
        
        # Unload model (set to None to help garbage collection)
        print(f"Unloading model {model_idx + 1}/{len(model_specs)}")
        model = None
        tokenizer = None
        gc.collect()  # Force garbage collection to free memory
    
    print(f"\n{'='*60}")
    print(f"All models processed. Aggregating results using semantic comparison...")
    print(f"{'='*60}")
    
    # Reorganize results: List[Dict] per model → List[List[Dict]] per prompt
    # Each prompt should have a list of results (one from each model)
    samples_per_prompt = []
    for prompt_idx in range(len(prompts)):
        prompt_samples = []
        for model_results in all_model_results:
            if prompt_idx < len(model_results):
                prompt_samples.append(model_results[prompt_idx])
        samples_per_prompt.append(prompt_samples)
    
    # Load database IDs if not explicitly provided
    if db_ids is None and input_file:
        try:
            input_file_clean = input_file.removesuffix('.jsonl')
            data_path = f"{ROOT_PATH}/data/training/nl2SQL/{template_folder}/{input_file_clean+'.jsonl'}"
            _, db_ids = load_prompts_with_db_ids(data_path)
            print(f"Loaded database IDs for {len([d for d in db_ids if d])} prompts")
        except Exception as e:
            print(f"Warning: Could not load database IDs from {input_file}: {e}")
            print("Falling back to extracting db_id from samples or using syntactic comparison")
    
    # Aggregate results using semantic majority voting (execution results)
    aggregated_results = aggregate_results_semantic_majority_vote(
        samples_per_prompt=samples_per_prompt,
        db_ids=db_ids
    )
    
    # Update consistency mode and models_used for cross-consistency
    for result in aggregated_results:
        result["consistency_mode"] = "cross"
        result["models_used"] = model_names_list
    
    return aggregated_results

def normalize_response(text: str) -> str:
    """
    Normalize the LLM response to fit in a single line.
    Removes excessive whitespace, newlines, and tabs.

    Args:
        text (str): The raw response string from the LLM.

    Returns:
        str: A normalized, single-line string.
    """
    return " ".join(text.split())

def post_process_sql(sql_text: str) -> str:
    """
    Post-process SQL output to remove markdown formatting and other verbosity.
    Handles chain-of-thought reasoning tags and markdown code blocks.
    
    Args:
        sql_text (str): The raw SQL text that may contain markdown formatting.
        
    Returns:
        str: Clean SQL query without markdown formatting.
    """
    if not sql_text:
        return ""
    
    import re
    
    # Step 1: If there's a </think> tag, extract everything after it
    if '</think>' in sql_text:
        parts = sql_text.split('</think>', 1)
        if len(parts) > 1:
            sql_text = parts[1].strip()
    
    # Step 2: Extract SQL from markdown code blocks (```sql ... ```)
    # Use multiline and dotall flags to match across newlines
    sql_block_pattern = r'```sql\s*(.*?)\s*```'
    match = re.search(sql_block_pattern, sql_text, re.IGNORECASE | re.DOTALL)
    if match:
        sql_text = match.group(1).strip()
    else:
        # Fallback: try to match generic code blocks (``` ... ```)
        generic_block_pattern = r'```\s*(.*?)\s*```'
        match = re.search(generic_block_pattern, sql_text, re.DOTALL)
        if match:
            sql_text = match.group(1).strip()
        else:
            # If no code block found, remove any leading/trailing ```sql or ```
            sql_text = re.sub(r'^```sql\s*', '', sql_text, flags=re.IGNORECASE)
            sql_text = re.sub(r'^```\s*', '', sql_text)
            sql_text = re.sub(r'\s*```\s*$', '', sql_text)
    
    # Remove any remaining backticks
    sql_text = sql_text.replace('`', '')
    
    # Clean up whitespace - normalize to single spaces and remove leading/trailing whitespace
    sql_text = " ".join(sql_text.split())
    
    # Remove common prefixes that models sometimes add
    prefixes_to_remove = [
        "here's the sql query:",
        "here is the sql query:",
        "the sql query is:",
        "sql query:",
        "query:",
        "sql:",
    ]
    
    sql_text_lower = sql_text.lower()
    for prefix in prefixes_to_remove:
        if sql_text_lower.startswith(prefix):
            sql_text = sql_text[len(prefix):].strip()
            break
    
    # Remove trailing dots that are not part of SQL
    sql_text = sql_text.rstrip('.')

    # Remove semicolons from the SQL text
    sql_text = sql_text.replace(';', '')
    
    return sql_text.lower()

def aggregate_results_majority_vote(samples_per_prompt: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Aggregate multiple samples per prompt using majority voting with random tie-breaking.
    
    Args:
        samples_per_prompt: List where each element is a list of sample results for one prompt
        
    Returns:
        List of aggregated results, one per prompt
    """
    import random
    
    aggregated_results = []
    
    for prompt_idx, samples in enumerate(samples_per_prompt):
        if not samples:
            # No samples for this prompt
            aggregated_results.append({
                "prompt_index": prompt_idx,
                "prompt": "",
                "response": None,
                "all_responses": [],
                "consistency_score": 0.0,
                "response_counts": {},
                "generation_time": 0.0,
                "status": "error",
                "consistency_mode": "self",
                "num_samples": 0,
                "models_used": []
            })
            continue
            
        # Extract responses and normalize them
        responses = []
        total_time = 0.0
        statuses = []
        
        for sample in samples:
            if sample['status'] == 'success' and sample['response']:
                normalized = post_process_sql(sample['response'])
                responses.append(normalized)
                total_time += sample.get('generation_time', 0.0)
                statuses.append('success')
            else:
                statuses.append(sample.get('status', 'error'))
        
        if not responses:
            # All samples failed
            aggregated_results.append({
                "prompt_index": prompt_idx,
                "prompt": samples[0]['prompt'] if samples else "",
                "response": None,
                "all_responses": [],
                "consistency_score": 0.0,
                "response_counts": {},
                "generation_time": total_time,
                "status": "error",
                "consistency_mode": "self",
                "num_samples": len(samples),
                "models_used": []
            })
            continue
        
        # Count frequency of each unique response
        response_counts = {}
        for response in responses:
            response_counts[response] = response_counts.get(response, 0) + 1
        
        # Find the most frequent response(s)
        max_count = max(response_counts.values())
        most_frequent = [resp for resp, count in response_counts.items() if count == max_count]
        
        # Random tie-breaking
        selected_response = random.choice(most_frequent)
        
        # Calculate consistency score
        consistency_score = max_count / len(responses)
        
        # Determine overall status
        success_rate = sum(1 for s in statuses if s == 'success') / len(statuses)
        overall_status = 'success' if success_rate > 0.5 else 'error'
        
        aggregated_results.append({
            "prompt_index": prompt_idx,
            "prompt": samples[0]['prompt'],
            "response": selected_response,
            "all_responses": responses,
            "consistency_score": consistency_score,
            "response_counts": response_counts,
            "generation_time": total_time / len(samples),
            "status": overall_status,
            "consistency_mode": "self",
            "num_samples": len(samples),
            "models_used": []
        })
    
    return aggregated_results

def execute_sql_query(sql_query: str, db_path: str) -> tuple[Any, str]:
    """
    Execute a SQL query against a database and return the result set.
    
    Args:
        sql_query: SQL query string to execute
        db_path: Path to the SQLite database file
        
    Returns:
        Tuple of (result_dataframe, error_message)
        If successful, error_message is None and result_dataframe is a pandas DataFrame
        If failed, error_message contains the error and result_dataframe is None
    """
    import sqlite3
    import pandas as pd
    
    if not sql_query or not sql_query.strip():
        return None, "Empty SQL query"
    
    if not os.path.exists(db_path):
        return None, f"Database file not found: {db_path}"
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(sql_query)

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        if columns:
            df = pd.DataFrame(rows, columns=columns)
            df = df.sort_values(by=columns, axis=0).reset_index(drop=True)
        else:
            df = pd.DataFrame()

        return df, None

    except Exception as e:
        return None, str(e)
    finally:
        if conn is not None:
            conn.close()

def compare_sql_results_semantically(result1: Any, result2: Any) -> bool:
    """
    Compare two SQL execution results semantically.
    
    Args:
        result1: First result (pandas DataFrame or None)
        result2: Second result (pandas DataFrame or None)
        
    Returns:
        True if results are semantically equivalent, False otherwise
    """
    import pandas as pd
    
    # Both None or both empty
    if result1 is None and result2 is None:
        return True
    
    # One is None, other is not
    if result1 is None or result2 is None:
        return False
    
    # Both are DataFrames
    if isinstance(result1, pd.DataFrame) and isinstance(result2, pd.DataFrame):
        # Both empty
        if result1.empty and result2.empty:
            return True
        
        # One empty, other not
        if result1.empty or result2.empty:
            return False
        
        # Compare DataFrames (already sorted)
        try:
            return result1.equals(result2)
        except Exception:
            return False
    
    return False

def aggregate_results_semantic_majority_vote(
    samples_per_prompt: List[List[Dict[str, Any]]],
    db_ids: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Aggregate multiple samples per prompt using majority voting based on semantic equivalence
    (execution results) rather than syntactic string matching.
    
    This function executes each SQL query against its corresponding database and groups
    queries that produce equivalent results, then selects the most frequent query by
    semantic equivalence.
    
    Args:
        samples_per_prompt: List where each element is a list of sample results for one prompt
        db_ids: List of database IDs corresponding to each prompt. If None, will try to
                extract from samples or use a default database path structure.
        
    Returns:
        List of aggregated results, one per prompt
    """
    import random
    import pandas as pd
    
    SPIDER_DB_PATH = f"{ROOT_PATH}/database/spider"
    
    aggregated_results = []
    
    for prompt_idx, samples in enumerate(samples_per_prompt):
        if not samples:
            # No samples for this prompt
            aggregated_results.append({
                "prompt_index": prompt_idx,
                "prompt": "",
                "response": None,
                "all_responses": [],
                "consistency_score": 0.0,
                "response_counts": {},
                "generation_time": 0.0,
                "status": "error",
                "consistency_mode": "cross",
                "num_samples": 0,
                "models_used": [],
                "aggregation_method": "semantic"
            })
            continue
        
        # Get database ID for this prompt
        db_id = None
        if db_ids and prompt_idx < len(db_ids):
            db_id = db_ids[prompt_idx]
        
        # If db_id not provided, try to extract from samples
        if not db_id:
            for sample in samples:
                if 'db_id' in sample:
                    db_id = sample['db_id']
                    break
        
        # Build database path
        db_path = None
        if db_id:
            db_path = os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")
        
        # Extract responses and execute them
        executed_results = []
        total_time = 0.0
        statuses = []
        sql_queries = []
        
        for sample in samples:
            if sample['status'] == 'success' and sample['response']:
                sql_query = post_process_sql(sample['response'])
                sql_queries.append(sql_query)
                total_time += sample.get('generation_time', 0.0)
                statuses.append('success')
                
                # Execute SQL query if we have a database path
                if db_path and os.path.exists(db_path):
                    result_df, error = execute_sql_query(sql_query, db_path)
                    if error:
                        executed_results.append((None, error))
                    else:
                        executed_results.append((result_df, None))
                else:
                    # No database available, fall back to syntactic comparison
                    executed_results.append((sql_query, None))  # Store query string as fallback
            else:
                sql_queries.append(None)
                executed_results.append((None, sample.get('error', 'Unknown error')))
                statuses.append(sample.get('status', 'error'))
        
        if not sql_queries or all(q is None for q in sql_queries):
            # All samples failed
            aggregated_results.append({
                "prompt_index": prompt_idx,
                "prompt": samples[0]['prompt'] if samples else "",
                "response": None,
                "all_responses": [],
                "consistency_score": 0.0,
                "response_counts": {},
                "generation_time": total_time,
                "status": "error",
                "consistency_mode": "cross",
                "num_samples": len(samples),
                "models_used": [],
                "aggregation_method": "semantic"
            })
            continue
        
        # Group queries by semantic equivalence
        # If we have database execution results, use semantic comparison
        # Otherwise, fall back to syntactic comparison
        use_semantic = db_path and os.path.exists(db_path) and all(
            isinstance(r[0], pd.DataFrame) or r[0] is None for r in executed_results
        )
        
        if use_semantic:
            # Semantic grouping: group by equivalent execution results
            # We'll compare each result with existing groups to find equivalent ones
            semantic_groups = []  # List of (representative_result, indices_list)
            
            for i, (result, error) in enumerate(executed_results):
                if error or result is None:
                    # Failed queries: group by error message
                    error_key = error or "Unknown error"
                    found_group = False
                    for group_result, group_indices in semantic_groups:
                        if isinstance(group_result, str) and group_result.startswith("__ERROR__"):
                            if group_result == f"__ERROR__{error_key}":
                                group_indices.append(i)
                                found_group = True
                                break
                    if not found_group:
                        semantic_groups.append((f"__ERROR__{error_key}", [i]))
                else:
                    # Successful queries: compare DataFrames semantically
                    found_group = False
                    for group_result, group_indices in semantic_groups:
                        if isinstance(group_result, pd.DataFrame):
                            if compare_sql_results_semantically(result, group_result):
                                group_indices.append(i)
                                found_group = True
                                break
                    if not found_group:
                        # Create new group with this result as representative
                        semantic_groups.append((result, [i]))
            
            # Find the largest group
            largest_group_size = max(len(indices) for _, indices in semantic_groups)
            largest_groups = [
                (group_result, indices) 
                for group_result, indices in semantic_groups 
                if len(indices) == largest_group_size
            ]
            
            # Random tie-breaking among largest groups
            selected_group_result, selected_indices = random.choice(largest_groups)
            
            # Get the SQL query from the first sample in the selected group
            selected_idx = selected_indices[0]
            selected_response = sql_queries[selected_idx]
            
            # Calculate consistency score
            consistency_score = largest_group_size / len(sql_queries)
            
            # Build response counts (grouped by semantic equivalence).
            # Failed queries use a unique sentinel key to avoid None-key collisions.
            response_counts = {}
            for group_idx, (group_result, indices) in enumerate(semantic_groups):
                rep_idx = indices[0]
                rep_query = sql_queries[rep_idx]
                key = rep_query if rep_query is not None else f"__failed_group_{group_idx}__"
                response_counts[key] = len(indices)
            
        else:
            # Fallback to syntactic comparison if database not available
            print(f"Warning: Database not available for prompt {prompt_idx}, using syntactic comparison")
            response_counts = {}
            for sql_query in sql_queries:
                if sql_query:
                    response_counts[sql_query] = response_counts.get(sql_query, 0) + 1
            
            # Find the most frequent response(s)
            max_count = max(response_counts.values()) if response_counts else 0
            most_frequent = [resp for resp, count in response_counts.items() if count == max_count]
            
            # Random tie-breaking
            selected_response = random.choice(most_frequent) if most_frequent else None
            consistency_score = max_count / len(sql_queries) if sql_queries else 0.0
        
        # Determine overall status
        success_rate = sum(1 for s in statuses if s == 'success') / len(statuses) if statuses else 0.0
        overall_status = 'success' if success_rate > 0.5 else 'error'
        
        aggregated_results.append({
            "prompt_index": prompt_idx,
            "prompt": samples[0]['prompt'],
            "response": selected_response,
            "all_responses": [q for q in sql_queries if q],
            "consistency_score": consistency_score,
            "response_counts": response_counts,
            "generation_time": total_time / len(samples) if samples else 0.0,
            "status": overall_status,
            "consistency_mode": "cross",
            "num_samples": len(samples),
            "models_used": [],
            "aggregation_method": "semantic" if use_semantic else "syntactic_fallback"
        })
    
    return aggregated_results

def save_results(results: List[Dict[str, Any]], output_file: str = "pred.sql"):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for query in results:
            if query['response']:
                # Post-process the SQL to remove markdown formatting
                clean_sql = post_process_sql(query['response'])
                f.write(clean_sql + "\n")
            else:
                # Write empty line for failed queries
                f.write("\n")
    print(f"Results saved to {output_file}")

def save_detailed_results(results: List[Dict[str, Any]], output_file: str = "pred_detailed.jsonl"):
    """Save detailed results including all samples and metadata to JSONL format"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Detailed results saved to {output_file}")

def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a JSONL file. Each line must be a JSON object with a 'prompt' key."""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'prompt' in data:
                    prompts.append(data['prompt'])
                else:
                    print(f"Warning: Line {line_num} missing 'prompt' field, skipping")
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}, skipping: {e}")
    return prompts


def load_jsonl_records(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL records (one JSON object per line). Empty lines are skipped."""
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    print(f"Warning: Line {line_num} is not a JSON object, skipping")
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}, skipping: {e}")
    return records


def get_model_dir_name(name: str) -> str:
    """Extract model directory name, removing common prefixes."""
    if name.startswith("mlx-community/"):
        return name.removeprefix("mlx-community/")
    if "/" in name:
        return name.split("/")[-1]
    return name


def infer_input_base_from_presql_file(presql_file: str) -> str:
    """Infer the {input_file} base used in output naming from a preSQL artifact filename."""
    base = Path(presql_file).name
    # common suffixes we may generate
    suffixes = [
        "_predictions_presql_detailed.jsonl",
        "_predictions_presql.jsonl",
        "_predictions_presql.sql",
        "_presql_detailed.jsonl",
        "_presql.jsonl",
        "_presql.sql",
        ".jsonl",
    ]
    for s in suffixes:
        if base.endswith(s):
            return base[: -len(s)]
    # fallback: drop extension
    return Path(base).stem


def build_predictions_dir(template_folder: str, model_specs: List[str], consistency_mode: str) -> str:
    """
    Output folder policy:
      - cross-consistency: .../{template}/cross{N}models/
      - otherwise: .../{template}/{model_dir}/
    """
    if consistency_mode == "cross":
        return f"{PRED_PATH}/nl2SQL/{template_folder}/cross{len(model_specs)}models"
    model_name, _ = parse_model_spec(model_specs[0])
    return f"{PRED_PATH}/nl2SQL/{template_folder}/{get_model_dir_name(model_name)}"


def render_refined_prompts(
    template: str, records: List[Dict[str, Any]], default_question_key: str = "question"
) -> List[str]:
    """
    Render refined prompts for finSQL generation.
    Requires each record to include:
      - question
      - simplified_ddl
      - foreign_keys
      - presql
    """
    template_base = template.removesuffix(".j2")
    refined_template_file = f"{template_base}_refined.j2"
    env = Environment(loader=FileSystemLoader(f"{ROOT_PATH}/data/templates/nl2SQL"))
    refined_template = env.get_template(refined_template_file)

    prompts: List[str] = []
    for rec in records:
        question = rec.get(default_question_key) or ""
        simplified_ddl = rec.get("simplified_ddl") or ""
        foreign_keys = rec.get("foreign_keys") or ""
        presql = rec.get("presql") or rec.get("preSQL") or ""

        refined = refine_schema_from_sql(presql, simplified_ddl, foreign_keys)

        params = {
            "question": question,
            "refined_simplified_ddl": refined.refined_simplified_ddl,
            "refined_foreign_keys": refined.refined_foreign_keys,
            # optional fields supported by some refined templates
            "refined_cell_values": rec.get("cell_values", ""),
            "refined_few_shot": rec.get("few_shot", ""),
        }
        prompts.append(refined_template.render(params))
    return prompts

def load_prompts_with_db_ids(file_path: str) -> tuple[List[str], List[str]]:
    """
    Load prompts and database IDs from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Tuple of (prompts, db_ids) where db_ids may contain None values
        if db_id is not present in the data
    """
    prompts = []
    db_ids = []
    
    # Try to load from gold database if db_id not in file
    import sqlite3
    BRONZE_DB = f"{ROOT_PATH}/database/bronze/bronze.sqlite"
    db_id_map = {}
    
    try:
        conn = sqlite3.connect(BRONZE_DB)
        cursor = conn.cursor()
        # Get all entries with their prompts and db_ids
        cursor.execute("SELECT question, db_id FROM spider_dataset")
        for question, db_id in cursor.fetchall():
            db_id_map[question.strip().lower()] = db_id
        conn.close()
    except Exception as e:
        print(f"Warning: Could not load database IDs from bronze DB: {e}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'prompt' in data:
                    prompts.append(data['prompt'])
                    # Try to get db_id from data, or from bronze DB
                    if 'db_id' in data:
                        db_ids.append(data['db_id'])
                    elif 'question' in data:
                        # Try to match question to get db_id
                        question_key = data['question'].strip().lower()
                        db_ids.append(db_id_map.get(question_key))
                    else:
                        # Try to extract question from prompt
                        prompt_text = data['prompt']
                        # Simple heuristic: first line after "###" might be the question
                        question_match = None
                        for prompt_line in prompt_text.split('\n'):
                            if prompt_line.strip().startswith('###') and '?' in prompt_line:
                                question_text = prompt_line.replace('###', '').strip()
                                question_key = question_text.lower()
                                db_ids.append(db_id_map.get(question_key))
                                question_match = True
                                break
                        if not question_match:
                            db_ids.append(None)
                else:
                    print(f"Warning: Line {line_num} missing 'prompt' field, skipping")
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}, skipping: {e}")
    
    return prompts, db_ids

def main(
    model_specs: List[str],
    template,
    input_file=None,
    batch_size=None,
    consistency_mode="none",
    num_samples=1,
    temperature=0.7,
    max_tokens=512,
    presql: bool = False,
    finsql: bool = False,
    presql_file: str = None,
):
    """
    Generate predictions using nl2SQL models with optional consistency modes.

    Args:
        model_specs (List[str]): List of model specifications in format "model_name" or "model_name:fine-tuned"
        template (str): Template name to use
        input_file (str): Input file for prompts
        batch_size (int): Batch size for processing
        consistency_mode (str): Consistency strategy (none, self, cross)
        num_samples (int): Number of samples per prompt for consistency
        temperature (float): Temperature for sampling
        max_tokens (int): Maximum tokens to generate per prompt (default: 512)
    """

    template_folder = template.removesuffix('.j2')
    input_base = input_file.removesuffix(".jsonl") if input_file else None

    # Mode selection (manual two-step)
    if presql and finsql:
        raise ValueError("Only one of presql or finsql can be True")

    # ---- preSQL mode ----
    if presql:
        if not input_base:
            raise ValueError("presql mode requires input_file")
        data_path = f"{ROOT_PATH}/data/training/nl2SQL/{template_folder}/{input_base+'.jsonl'}"
        records = load_jsonl_records(data_path)
        prompts = [r["prompt"] for r in records if "prompt" in r]
        if not prompts:
            raise ValueError(f"No prompts found in {data_path}")

        # preSQL always uses only the first model spec (even if user passed multiple models)
        model_spec = model_specs[0]
        model_name, use_adapter = parse_model_spec(model_spec)

        model_dir_name = get_model_dir_name(model_name)
        adapter_path = None
        if use_adapter:
            adapter_path = f"{ROOT_PATH}/data/adapters/nl2SQL/{template_folder}/{model_dir_name}"

        # For preSQL artifacts, pick output folder based on whether user provided a model list for cross runs.
        # If multiple models were passed, store under cross{N}models/ for consistency with the future finSQL run.
        output_consistency_mode = "cross" if len(model_specs) > 1 else "none"
        out_dir = build_predictions_dir(template_folder, model_specs, output_consistency_mode)

        print(f"Starting preSQL generation with {len(prompts)} prompts")
        print(f"Model (preSQL): {model_name}{' (fine-tuned)' if use_adapter else ' (base)'}")
        if len(model_specs) > 1:
            print(f"Note: {len(model_specs)} models provided; preSQL will use only the FIRST model. Output folder: {out_dir}")

        model, tokenizer = load_model(model_name, adapter_path)
        results = process_batch(prompts=prompts, model=model, tokenizer=tokenizer, max_tokens=max_tokens, batch_size=batch_size)

        # Attach preSQL + metadata
        detailed = []
        for rec, res in zip(records, results):
            presql_text = post_process_sql(res.get("response") or "")
            detailed.append(
                {
                    "prompt_index": res.get("prompt_index"),
                    "model_spec": model_spec,
                    "model_name": model_name,
                    "strategy": "nl2SQL",
                    "template": template_folder,
                    "prompt": rec.get("prompt"),
                    "question": rec.get("question"),
                    "db_id": rec.get("db_id"),
                    "simplified_ddl": rec.get("simplified_ddl"),
                    "foreign_keys": rec.get("foreign_keys"),
                    "presql": presql_text,
                    "status": res.get("status"),
                    "error": res.get("error"),
                }
            )

        presql_sql_file = f"{out_dir}/{input_base}_predictions_presql.sql"
        presql_jsonl_file = f"{out_dir}/{input_base}_predictions_presql_detailed.jsonl"

        # Save .sql (optional convenience) and detailed JSONL
        save_results([{"response": d["presql"]} for d in detailed], presql_sql_file)
        save_detailed_results(detailed, presql_jsonl_file)
        return

    # ---- finSQL mode ----
    if finsql:
        if not presql_file:
            raise ValueError("finsql mode requires presql_file")

        records = load_jsonl_records(presql_file)
        if not records:
            raise ValueError(f"No records found in presql_file={presql_file}")

        inferred_base = infer_input_base_from_presql_file(presql_file)
        input_base = input_base or inferred_base

        # Extract db_ids for semantic aggregation (cross-consistency)
        db_ids = [r.get("db_id") for r in records]

        refined_prompts = render_refined_prompts(template=template, records=records)

        print(f"Starting finSQL generation with {len(refined_prompts)} refined prompts")
        print(f"Consistency mode: {consistency_mode}")

        # Validate consistency args for finSQL
        if consistency_mode == "none" and num_samples != 1:
            raise ValueError("num-samples is only valid for consistency-mode 'self' or 'cross'")
        if consistency_mode == "cross":
            if len(model_specs) < 2:
                raise ValueError("For cross mode, at least 2 models are required")
            num_samples = len(model_specs)
        if consistency_mode in ["none", "self"] and len(model_specs) != 1:
            raise ValueError(f"For consistency-mode '{consistency_mode}', exactly 1 model is required")

        # Select output directory (new cross folder convention)
        out_dir = build_predictions_dir(template_folder, model_specs, consistency_mode)

        if consistency_mode in ["none", "self"]:
            model_spec = model_specs[0]
            model_name, use_adapter = parse_model_spec(model_spec)
            model_dir_name = get_model_dir_name(model_name)
            adapter_path = None
            if use_adapter:
                adapter_path = f"{ROOT_PATH}/data/adapters/nl2SQL/{template_folder}/{model_dir_name}"

        if consistency_mode == "none":
            model, tokenizer = load_model(model_name, adapter_path)
            results = process_batch(
                prompts=refined_prompts,
                model=model,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                batch_size=batch_size,
            )
            output_file = f"{out_dir}/{input_base}_predictions.sql"
            detailed_output_file = f"{out_dir}/{input_base}_predictions_finsql_detailed.jsonl"

        elif consistency_mode == "self":
            model, tokenizer = load_model(model_name, adapter_path)
            results = process_self_consistent(
                prompts=refined_prompts,
                model=model,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                batch_size=batch_size,
                num_samples=num_samples,
                temperature=temperature,
            )
            output_file = f"{out_dir}/{input_base}_predictions_self.sql"
            detailed_output_file = f"{out_dir}/{input_base}_predictions_self_finsql_detailed.jsonl"

        elif consistency_mode == "cross":
            results = process_cross_consistent(
                prompts=refined_prompts,
                model_specs=model_specs,
                template=template,
                max_tokens=max_tokens,
                batch_size=batch_size,
                temperature=temperature,
                input_file=input_base,
                db_ids=db_ids,
            )
            num_models = len(model_specs)
            output_file = f"{out_dir}/{input_base}_predictions_cross_{num_models}models.sql"
            detailed_output_file = f"{out_dir}/{input_base}_predictions_cross_{num_models}models_finsql_detailed.jsonl"

        else:
            raise ValueError(f"Unknown consistency mode: {consistency_mode}")

        # Save final SQL in the exact same format as today + detailed jsonl
        save_results(results, output_file)

        # add presql + prompt metadata back into the finSQL detailed file
        detailed = []
        for rec, res in zip(records, results):
            item = dict(rec)
            item["finsql"] = res.get("response")
            item["consistency_mode"] = res.get("consistency_mode", consistency_mode)
            item["status"] = res.get("status")
            item["generation_time"] = res.get("generation_time")
            item["models_used"] = res.get("models_used", [])
            item["consistency_score"] = res.get("consistency_score")
            detailed.append(item)
        save_detailed_results(detailed, detailed_output_file)
        return

    # ---- default behavior (backward compatible) ----
    if not input_base:
        raise ValueError("input_file is required when not running in finsql mode")

    data = f"{ROOT_PATH}/data/training/nl2SQL/{template_folder}/{input_base+'.jsonl'}"
    prompts = load_prompts_from_file(data)
    
    print(f"Starting batch inference with {len(prompts)} prompts")
    print(f"Consistency mode: {consistency_mode}")
    if consistency_mode == 'cross':
        print(f"Number of models: {len(model_specs)}")
        print(f"Number of samples: {num_samples} (one per model)")
        print(f"Models: {', '.join([parse_model_spec(ms)[0] for ms in model_specs])}")
    elif consistency_mode in ['none', 'self']:
        _name, _ft = parse_model_spec(model_specs[0])
        print(f"Model: {_name}{' (fine-tuned)' if _ft else ' (base)'}")
    if consistency_mode == 'self':
        print(f"Number of samples: {num_samples}")
        print(f"Temperature: {temperature}")
    elif consistency_mode == 'cross':
        print(f"Temperature: {temperature}")
    print("=" * 60)

    # Process based on consistency mode
    if consistency_mode == 'none':
        model_spec = model_specs[0]
        model_name, use_adapter = parse_model_spec(model_spec)
        model_dir_name = get_model_dir_name(model_name)
        
        # Build adapter path if needed
        finetuned = 'finetuned' if use_adapter else ''
        adapter_path = None
        if use_adapter:
            adapter_path = f"{ROOT_PATH}/data/adapters/nl2SQL/{template_folder}/{model_dir_name}"
        # Standard processing
        model, tokenizer = load_model(model_name, adapter_path)
        
        results = process_batch(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            batch_size=batch_size
        )
        
        # Set output file names
        if finetuned:
            output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions_{finetuned}.sql"
            detailed_output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions_{finetuned}_detailed.jsonl"
        else:
            output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions.sql"
            detailed_output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions_detailed.jsonl"

    elif consistency_mode == 'self':
        # Self-consistency: same model, multiple samples
        model_spec = model_specs[0]
        model_name, use_adapter = parse_model_spec(model_spec)
        model_dir_name = get_model_dir_name(model_name)

        # Build adapter path if needed
        finetuned = 'finetuned' if use_adapter else ''
        adapter_path = None
        if use_adapter:
            adapter_path = f"{ROOT_PATH}/data/adapters/nl2SQL/{template_folder}/{model_dir_name}"
        
        model, tokenizer = load_model(model_name, adapter_path)
        
        results = process_self_consistent(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            batch_size=batch_size,
            num_samples=num_samples,
            temperature=temperature
        )
        
        # Set output file names
        if finetuned:
            output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions_self_{finetuned}.sql"
            detailed_output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions_self_{finetuned}_detailed.jsonl"
        else:
            output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions_self.sql"
            detailed_output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions_self_detailed.jsonl"

    elif consistency_mode == 'cross':
        # Cross-consistency: multiple models, one sample each
        results = process_cross_consistent(
            prompts=prompts,
            model_specs=model_specs,
            template=template,
            max_tokens=max_tokens,
            batch_size=batch_size,
            temperature=temperature,
            input_file=input_base
        )

        # Cross-consistency output folder convention: cross{N}models/
        model_dir_name = f"cross{len(model_specs)}models"

        # Output file naming for cross-consistency
        num_models = len(model_specs)
        output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions_cross{num_models}models.sql"
        detailed_output_file = f"{PRED_PATH}/nl2SQL/{template_folder}/{model_dir_name}/{input_base}_predictions_cross{num_models}models_detailed.jsonl"
    
    else:
        raise ValueError(f"Unknown consistency mode: {consistency_mode}")
    
    # Save results
    save_results(results, output_file)
    save_detailed_results(results, detailed_output_file)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    total_time = sum(r['generation_time'] for r in results)
    
    # Calculate consistency statistics if applicable
    consistency_stats = ""
    if consistency_mode in ['self', 'cross']:
        avg_consistency = sum(r.get('consistency_score', 0) for r in results) / len(results) if results else 0
        high_consistency = sum(1 for r in results if r.get('consistency_score', 0) > 0.8)
        consistency_stats = f"\nConsistency Statistics:"
        consistency_stats += f"\nAverage consistency score: {avg_consistency:.3f}"
        consistency_stats += f"\nHigh consistency (>0.8): {high_consistency}/{len(results)}"
        
        if consistency_mode == 'cross' and results:
            models_used = results[0].get('models_used', [])
            if models_used:
                consistency_stats += f"\nModels used: {len(models_used)} ({', '.join([get_model_dir_name(m) for m in models_used])})"
    
    print("=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    if prompts:
        print(f"Average time per prompt: {total_time/len(prompts):.2f}s")
    print(f"Results saved to {output_file}")
    print(f"Detailed results saved to {detailed_output_file}")
    print(consistency_stats)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Batch inference using nl2SQL models')
    
    # Define valid base model names (without :fine-tuned suffix)
    valid_models = [
        'mlx-community/Qwen3-14B-4bit',                    # 14B
        'mlx-community/phi-4-4bit',                        # 14B
        'mlx-community/Ministral-3-14B-Instruct-2512-4bit',# 14B
        'mlx-community/Ministral-8B-Instruct-2410-4bit',   # 8B
        'mlx-community/Meta-Llama-3.1-8B-Instruct-4bit',   # 8B
        'mlx-community/Olmo-3-7B-Instruct-4bit',           # 7B
        'mlx-community/Llama-3.2-3B-Instruct-4bit',        # 3B

        'mlx-community/Phi-4-reasoning-plus-4bit',         # x 14B
        'mlx-community/Phi-4-reasoning-4bit',              # x 14B
        'mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit', # x 14B
        'Qwen/Qwen3-8B-MLX-4bit',                          # x 8B
        'mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit',  # x 7B
        'Qwen/Qwen3-4B-MLX-4bit',                          # x 4B
        'mlx-community/Phi-4-mini-reasoning-4bit',         # x 3.8B
        'mlx-community/Llama-3.2-1B-Instruct-4bit',        # x 1B
    ]
    
    def validate_model_spec(value: str) -> str:
        """Validate model specification format"""
        try:
            model_name, _ = parse_model_spec(value)
            # Check if base model name is in valid list
            if model_name not in valid_models:
                raise argparse.ArgumentTypeError(
                    f"Model '{model_name}' not in valid models list. "
                    f"Valid models: {', '.join(valid_models)}"
                )
            return value
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))
    
    def validate_models_list(value):
        """Validate each model in the list"""
        # This will be called for each model in nargs='+'
        return validate_model_spec(value)
    
    parser.add_argument('--models', type=validate_models_list, nargs='+', required=True,
                       help='Model(s) to use. Format: "model_name" or "model_name:fine-tuned". '
                            'For single model: --models model1 '
                            'For multiple models: --models model1 model2 model3. '
                            'Examples: --models mlx-community/Llama-3.2-3B-Instruct-4bit or '
                            '--models mlx-community/Llama-3.2-1B-Instruct-4bit mlx-community/Llama-3.2-3B-Instruct-4bit')
    parser.add_argument('--template', type=str, default='template_12',
                       help='Template name to use for training data (default: template_12)')
    parser.add_argument('--input-file', type=str, required=False,
                       help='Input file for prompts (training jsonl name). Required for normal mode and --presql.')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--presql', action='store_true', help='Generate intermediate preSQL and store JSONL artifacts.')
    mode_group.add_argument('--finsql', action='store_true', help='Generate finSQL from a preSQL JSONL artifact.')
    parser.add_argument('--presql-file', type=str, default=None,
                       help='Path to the preSQL JSONL artifact to consume (required for --finsql).')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for processing prompts. If not specified, processes all prompts at once. Use smaller values to avoid memory issues.')
    
    # Self-consistency arguments
    parser.add_argument('--consistency-mode', type=str, default='none',
                       choices=['none', 'self', 'cross'],
                       help='Consistency strategy (default: none). '
                            'num-samples is only valid for "self" mode. '
                            'For "cross" mode, num-samples is automatically set to number of models.')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples per prompt for consistency (default: 1). '
                            'Only valid for consistency-mode "self". '
                            'For "cross" mode, automatically set to number of models (one sample per model).')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for sampling in consistency mode (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate per prompt (default: 512)')
    
    args = parser.parse_args()

    # Mode-specific argument validation
    if args.presql:
        if not args.input_file:
            parser.error("--presql requires --input-file")
        # preSQL generation always uses only the first model; allow users to pass multiple models
        # (intended for later cross-consistency finSQL).
    elif args.finsql:
        if not args.presql_file:
            parser.error("--finsql requires --presql-file")
        # finSQL obeys the existing consistency-mode rules
        if args.consistency_mode == "none" and args.num_samples != 1:
            parser.error("--num-samples is only valid for consistency-mode 'self' or 'cross'")
        if args.consistency_mode == "cross":
            if len(args.models) < 2:
                parser.error("For --consistency-mode cross, at least 2 models are required")
            args.num_samples = len(args.models)
        if args.consistency_mode in ["none", "self"] and len(args.models) != 1:
            parser.error(f"For consistency-mode '{args.consistency_mode}', exactly 1 model is required")
    else:
        # Backward compatible behavior
        if not args.input_file:
            parser.error("Normal mode requires --input-file")
        if args.consistency_mode == "none" and args.num_samples != 1:
            parser.error("--num-samples is only valid for consistency-mode 'self' or 'cross'")
        if args.consistency_mode == "cross":
            if len(args.models) < 2:
                parser.error("For --consistency-mode cross, at least 2 models are required")
            args.num_samples = len(args.models)
        if args.consistency_mode in ["none", "self"] and len(args.models) != 1:
            parser.error(f"For consistency-mode '{args.consistency_mode}', exactly 1 model is required")

    main(
        args.models,
        args.template,
        args.input_file,
        args.batch_size,
        args.consistency_mode,
        args.num_samples,
        args.temperature,
        args.max_tokens,
        presql=args.presql,
        finsql=args.finsql,
        presql_file=args.presql_file,
    )
