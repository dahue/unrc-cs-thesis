import os
import subprocess
from dotenv import load_dotenv

def main(model, model_type):
    """
    Fine-tune either nl2SQL or nl2NatSQL models on a specified dataset.
    
    Args:
        model (str): Model to fine-tune
        model_type (str): Either 'nl2SQL' or 'nl2NatSQL'
    """
    if model_type not in ['nl2SQL', 'nl2NatSQL']:
        raise ValueError("model_type must be either 'nl2SQL' or 'nl2NatSQL'")

    # Load environment variables
    load_dotenv()
    ROOT_PATH = os.environ.get("ROOT_PATH")
    if not ROOT_PATH:
        raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

    cmd = [
        "mlx_lm.lora",
        "--model", model,
        "--train",
        "--data", f"{ROOT_PATH}/data/training/{model_type}/template_11",
        "--adapter-path", f"{ROOT_PATH}/data/adapters/{model_type}/{model.strip('mlx-community/')}",
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
    parser.add_argument('--model-type', type=str, required=True, choices=['nl2SQL', 'nl2NatSQL'],
                       help='Type of model to fine-tune')
    args = parser.parse_args()
    main(args.model, args.model_type)