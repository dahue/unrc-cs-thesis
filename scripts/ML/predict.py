import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from mlx_lm import load, batch_generate
from dotenv import load_dotenv

load_dotenv()
ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH:
    raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

PRED_PATH = f"{ROOT_PATH}/data/predictions"


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
                import mlx.core as mx
                mx.random.seed(seed)
                
                # Use generate instead of batch_generate for individual samples
                from mlx_lm import generate
                from mlx_lm.sample_utils import make_sampler
                sampler = make_sampler(
                    temp=temperature
                )
                response = generate(
                    model, 
                    tokenizer, 
                    formatted_prompt, 
                    verbose=False, 
                    max_tokens=max_tokens,
                    # temp=temperature
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
    
    Args:
        sql_text (str): The raw SQL text that may contain markdown formatting.
        
    Returns:
        str: Clean SQL query without markdown formatting.
    """
    if not sql_text:
        return ""
    
    # Remove markdown code blocks (```sql ... ``` or ``` ... ```)
    import re
    
    # Remove ```sql at the beginning and ``` at the end
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
    
    # Remove trailing punctuation that's not part of SQL
    sql_text = sql_text.rstrip('.;')
    
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
    """Load prompts from a text file, JSON file, or JSONL file"""
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

def main(model_name, strategy, template, use_adapter, input_file, batch_size=None, 
         consistency_mode='self', num_samples=1, temperature=0.7):
    """
    Generate predictions using nl2SQL or nl2NatSQL models with optional self-consistency.
    
    Args:
        model_name (str): Model to use for inference
        strategy (str): Strategy used (nl2SQL or nl2NatSQL)
        template (str): Template name to use
        use_adapter (bool): Whether to use adapter for inference
        input_file (str): Input file for prompts
        batch_size (int): Batch size for processing
        consistency_mode (str): Consistency strategy (none, self)
        num_samples (int): Number of samples per prompt for consistency
        temperature (float): Temperature for sampling
    """
    MAX_TOKENS = 512

    template_folder = template.removesuffix('.j2')
    input_file = input_file.removesuffix('.jsonl')
    data = f"{ROOT_PATH}/data/training/{strategy}/{template_folder}/{input_file+'.jsonl'}"

    prompts = load_prompts_from_file(data)
    
    print(f"Starting batch inference with {len(prompts)} prompts")
    print(f"Consistency mode: {consistency_mode}")
    if consistency_mode == 'self':
        print(f"Number of samples: {num_samples}")
        print(f"Temperature: {temperature}")
    print("=" * 60)

    # Process based on consistency mode
    if consistency_mode == 'none':
        # Standard processing
        finetuned = ''
        if use_adapter:
            finetuned = 'finetuned'
            adapter = f"{ROOT_PATH}/data/adapters/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}"
            model, tokenizer = load_model(model_name, adapter)
        else:
            model, tokenizer = load_model(model_name)
        
        results = process_batch(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            batch_size=batch_size
        )
        
        # Set output file names
        if finetuned:
            output_file = f"{PRED_PATH}/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}/{input_file}_predictions_{finetuned}.sql"
            detailed_output_file = f"{PRED_PATH}/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}/{input_file}_predictions_{finetuned}_detailed.jsonl"
        else:
            output_file = f"{PRED_PATH}/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}/{input_file}_predictions.sql"
            detailed_output_file = f"{PRED_PATH}/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}/{input_file}_predictions_detailed.jsonl"
    
    elif consistency_mode == 'self':
        # Self-consistency: same model, multiple samples
        finetuned = ''
        if use_adapter:
            finetuned = 'finetuned'
            adapter = f"{ROOT_PATH}/data/adapters/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}"
            model, tokenizer = load_model(model_name, adapter)
        else:
            model, tokenizer = load_model(model_name)
        
        results = process_self_consistent(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            batch_size=batch_size,
            num_samples=num_samples,
            temperature=temperature
        )
        
        # Set output file names
        if finetuned:
            output_file = f"{PRED_PATH}/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}/{input_file}_predictions_self_{finetuned}.sql"
            detailed_output_file = f"{PRED_PATH}/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}/{input_file}_predictions_self_{finetuned}_detailed.jsonl"
        else:
            output_file = f"{PRED_PATH}/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}/{input_file}_predictions_self.sql"
            detailed_output_file = f"{PRED_PATH}/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}/{input_file}_predictions_self_detailed.jsonl"
    
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
    if consistency_mode == 'self':
        avg_consistency = sum(r.get('consistency_score', 0) for r in results) / len(results)
        high_consistency = sum(1 for r in results if r.get('consistency_score', 0) > 0.8)
        consistency_stats = f"\nConsistency Statistics:"
        consistency_stats += f"\nAverage consistency score: {avg_consistency:.3f}"
        consistency_stats += f"\nHigh consistency (>0.8): {high_consistency}/{len(results)}"
    
    print("=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time/len(prompts):.2f}s")
    print(f"Results saved to {output_file}")
    print(f"Detailed results saved to {detailed_output_file}")
    print(consistency_stats)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Batch inference using nl2SQL or nl2NatSQL models')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to fine-tune', choices=[
                            'mlx-community/Llama-3.2-1B-Instruct-4bit',     # 1B
                            'mlx-community/Llama-3.2-3B-Instruct-4bit',     # 3B
                            'mlx-community/Phi-4-mini-reasoning-4bit',      # 3.8B
                            'Qwen/Qwen3-4B-MLX-4bit',                       # 4B
                            'mlx-community/Ministral-8B-Instruct-2410-4bit',# 8B
                            'Qwen/Qwen3-8B-MLX-4bit',                       # 8B
                            'mlx-community/phi-4-4bit',                     # 14B
                            'mlx-community/Qwen3-14B-4bit',                 # 14B
                            'mlx-community/Phi-4-reasoning-plus-4bit'       # 14B
                            'mlx-community/Phi-4-reasoning-4bit'            # 14B
                        ])
    parser.add_argument('--strategy', type=str, required=True, choices=['nl2SQL', 'nl2NatSQL'],
                       help='Strategy used to fine-tune')
    parser.add_argument('--template', type=str, default='template_12',
                       help='Template name to use for training data (default: template_12)')
    parser.add_argument('--use-adapter', action='store_true', default=False,
                        help='Use adapter for inference (default: False)')
    parser.add_argument('--input-file', type=str, required=True,
                       help='Input file for prompts. MUST be a jsonl file')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for processing prompts. If not specified, processes all prompts at once. Use smaller values to avoid memory issues.')
    
    # Self-consistency arguments
    parser.add_argument('--consistency-mode', type=str, default='self',
                       choices=['none', 'self'],
                       help='Consistency strategy (default: self)')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples per prompt for consistency (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for sampling in consistency mode (default: 0.7)')
    
    args = parser.parse_args()
    main(args.model, args.strategy, args.template, args.use_adapter, args.input_file, args.batch_size, 
         args.consistency_mode, args.num_samples, args.temperature)
