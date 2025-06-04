import os
import subprocess
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    ROOT_PATH = os.environ.get("ROOT_PATH")
    if not ROOT_PATH:
        raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

    # Set environment variable
    os.environ["MODEL"] = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    # Define the command
    cmd = [
        "mlx_lm.lora",
        "--model", os.environ["MODEL"],
        "--train",
        "--data", f"{ROOT_PATH}/data/training/nl2SQL",
        "--adapter-path", f"{ROOT_PATH}/experiments/models/Llama-3.2-3B-Instruct-4bit/nl2SQL/adapters",
        "--iters", "50",
        "--max-seq-length", "8192", # default 2048
        "--batch-size", "2",
        "--num-layers", "16"
    ]

    try:
        # Run the command
        subprocess.run(cmd, check=True)
        print("Fine-tuning completed successfully!")
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()