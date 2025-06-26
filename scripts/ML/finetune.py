import os
import subprocess
from dotenv import load_dotenv

def main(model, strategy, template):
    """
    Fine-tune either nl2SQL or nl2NatSQL models on a specified dataset.
    
    Args:
        model (str): Model to fine-tune
        strategy (str): Either 'nl2SQL' or 'nl2NatSQL'
        template (str): Template name to use for training data (default: template_11)
    """
    if strategy not in ['nl2SQL', 'nl2NatSQL']:
        raise ValueError("strategy must be either 'nl2SQL' or 'nl2NatSQL'")

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
        "--data", f"{ROOT_PATH}/data/training/{strategy}/{template_folder}",
        "--adapter-path", f"{ROOT_PATH}/data/adapters/{strategy}/{template_folder}/{model.removeprefix('mlx-community/')}",
        "--iters", "100",
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
                       help='Model to fine-tune', choices=['mlx-community/Llama-3.2-3B-Instruct-4bit', 'mlx-community/Llama-3.2-1B-Instruct-4bit'])
    parser.add_argument('--strategy', type=str, required=True, choices=['nl2SQL', 'nl2NatSQL'],
                       help='Strategy used to fine-tune')
    parser.add_argument('--template', type=str, default='template_12',
                       help='Template name to use for training data (default: template_12)')
    args = parser.parse_args()
    main(args.model, args.strategy, args.template)