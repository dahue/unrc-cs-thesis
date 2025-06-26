import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from mlx_lm import load, generate
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
    max_tokens: int = 512
) -> List[Dict[str, Any]]:
    """Process a batch of prompts and return results"""
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        
        start_time = time.time()
        
        try:
            # Generate response
            messages = [{"role": "user", "content": prompt}]
            message = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            
            response = normalize_response(
                generate(model, tokenizer, prompt=message, max_tokens=max_tokens, verbose=False)
            )

            end_time = time.time()
            
            result = {
                "prompt_index": i,
                "prompt": prompt,
                "response": response,
                "generation_time": end_time - start_time,
                "status": "success"
            }
            
        except Exception as e:
            result = {
                "prompt_index": i,
                "prompt": prompt,
                "response": None,
                "generation_time": 0,
                "status": "error",
                "error": str(e)
            }
            print(f"Error processing prompt {i+1}: {e}")
        
        results.append(result)
        print(f"Completed in {result['generation_time']:.2f}s")
        print("-" * 50)
    
    return results

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

def save_results(results: List[Dict[str, Any]], output_file: str = "pred.sql"):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for query in results:
            f.write(query['response'] + "\n")
    print(f"Results saved to {output_file}")

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

def main(model_name, strategy, template, use_adapter, input_file):
    """
    Fine-tune either nl2SQL or nl2NatSQL models on a specified dataset.
    
    Args:
        model_name (str): Model to fine-tune
    """
    MAX_TOKENS = 512

    template_folder = template.removesuffix('.j2')
    input_file = input_file.removesuffix('.jsonl')
    data = f"{ROOT_PATH}/data/training/{strategy}/{template_folder}/{input_file+'.jsonl'}"

    prompts = load_prompts_from_file(data)
    
    print(f"Starting batch inference with {len(prompts)} prompts")
    print("=" * 60)

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
        max_tokens=MAX_TOKENS
    )
    
    output_file = f"{PRED_PATH}/{strategy}/{template_folder}/{model_name.removeprefix('mlx-community/')}/{input_file+'_predictions_'+finetuned+'.sql'}"
    save_results(results, output_file)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    total_time = sum(r['generation_time'] for r in results)
    
    print("=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time/len(prompts):.2f}s")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Batch inference using nl2SQL or nl2NatSQL models')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to fine-tune', choices=[
                            'mlx-community/Llama-3.2-1B-Instruct-4bit',     # 1B
                            'mlx-community/Llama-3.2-3B-Instruct-4bit',     # 3B
                            'mlx-community/Phi-4-mini-reasoning-4bit',      # 3.8B
                            'mlx-community/Ministral-8B-Instruct-2410-4bit',# 8B
                            'mlx-community/phi-4-4bit',                     # 14B
                            'mlx-community/Qwen3-14B-4bit',                 # 14B
                            'mlx-community/Phi-4-reasoning-plus-4bit'       # 14B
                        ])
    parser.add_argument('--strategy', type=str, required=True, choices=['nl2SQL', 'nl2NatSQL'],
                       help='Strategy used to fine-tune')
    parser.add_argument('--template', type=str, default='template_12',
                       help='Template name to use for training data (default: template_12)')
    parser.add_argument('--use-adapter', action='store_true', default=False,
                        help='Use adapter for inference (default: False)')
    parser.add_argument('--input-file', type=str, required=True,
                       help='Input file for prompts. MUST be a jsonl file')
    args = parser.parse_args()
    main(args.model, args.strategy, args.template, args.use_adapter, args.input_file)
