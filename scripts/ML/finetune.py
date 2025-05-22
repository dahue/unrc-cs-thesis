import os
from dotenv import load_dotenv
from mlx_lm import LoRA
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    root_path = os.environ.get("ROOT_PATH")
    if not root_path:
        raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")

    # Model and training parameters
    model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    data_path = os.path.join(root_path, "data", "training", "nl2SQL")
    adapter_path = os.path.join(root_path, "experiments", "models", "Llama-3.2-3B-Instruct-4bit", "adapters", "nl2SQL")
    
    # Create adapter directory if it doesn't exist
    os.makedirs(adapter_path, exist_ok=True)

    logger.info(f"Starting fine-tuning with model: {model_name}")
    logger.info(f"Using data from: {data_path}")
    logger.info(f"Saving adapters to: {adapter_path}")

    try:
        # Initialize LoRA
        lora = LoRA(
            model_name=model_name,
            adapter_path=adapter_path,
            num_layers=16
        )

        # Train the model
        lora.train(
            data_path=data_path,
            batch_size=2,
            iters=50
        )

        logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()