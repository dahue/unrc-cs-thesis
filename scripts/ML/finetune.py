import os
import subprocess
from dotenv import load_dotenv

def main(model, template, iters):
    """
    Fine-tune nl2SQL models on a specified dataset.

    Args:
        model (str): Model to fine-tune
        template (str): Template name to use for training data (default: template_11)
        iters (int): Number of training iterations
    """
    # Load environment variables
    load_dotenv()
    ROOT_PATH = os.environ.get("ROOT_PATH")
    if not ROOT_PATH:
        raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

    # Strip .j2 extension if present to match folder structure
    template_folder = template.removesuffix('.j2')

    cmd = [
        "mlx_lm.lora",
        "--model", model,
        "--train",
        "--data", f"{ROOT_PATH}/data/training/nl2SQL/{template_folder}",
        "--adapter-path", f"{ROOT_PATH}/data/adapters/nl2SQL/{template_folder}/{model.split('/')[-1]}",
        "--iters", str(iters),
        "--max-seq-length", "2048", # default 2048
        "--batch-size", "2",
        "--num-layers", "8"
    ]

    try:
        # Run the command
        subprocess.run(cmd, check=True)
        print("Fine-tuning completed successfully!")
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune nl2SQL or nl2NatSQL models')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to fine-tune', choices=[
                            'mlx-community/Qwen3-14B-4bit',                   # 14B
                            'mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit',# 14B
                            'mlx-community/phi-4-4bit',                       # 14B
                            'mlx-community/Ministral-8B-Instruct-2410-4bit',  # 8B
                            'mlx-community/Meta-Llama-3.1-8B-Instruct-4bit',  # 8B
                            'mlx-community/Llama-3.2-3B-Instruct-4bit',       # 3B
                            'mlx-community/Llama-3.2-1B-Instruct-4bit',       # 1B

                            'mlx-community/Phi-4-mini-reasoning-4bit',        # 3.8B
                            'Qwen/Qwen3-4B-MLX-4bit',                         # 4B
                            'mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit', # 7B
                            'Qwen/Qwen3-8B-MLX-4bit',                         # 8B
                            'mlx-community/Phi-4-reasoning-plus-4bit',        # 14B
                            'mlx-community/Phi-4-reasoning-4bit'              # 14B
    ])
    parser.add_argument('--template', type=str, default='template_12',
                       help='Template name to use for training data (default: template_12)')
    parser.add_argument('--iters', type=int, default=100,
                       help='Number of training iterations (default: 100)')
    args = parser.parse_args()
    main(args.model, args.template, args.iters)