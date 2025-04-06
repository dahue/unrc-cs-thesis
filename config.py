import os
from dotenv import load_dotenv

# Load secrets from .env file (e.g., OpenAI key)
load_dotenv()

# ===============================
# Path Configuration
# ===============================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SPIDER_DATA_DIR = os.path.join(PROJECT_ROOT, "spider_data")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

PREPROCESSED_JSON = os.path.join(DATA_DIR, "preprocessed.json")
PROMPTS_JSON = os.path.join(DATA_DIR, "prompts.json")
EVAL_REPORT_JSON = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
SQLITE_DB_PATH = os.path.join(SPIDER_DATA_DIR, "databases")

# ===============================
# Model Provider Configuration
# ===============================
# Choose between: "openai", "lmstudio", "ollama"
LLM_PROVIDER = "exo"

MODELS = {
    "openai": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "headers": lambda: {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
    },
    "lmstudio": {
        "api_url": "http://localhost:1234/v1/chat/completions",
        "default_model": "mistral-7b-instruct",
        "headers": lambda: {
            "Content-Type": "application/json"
        }
    },
    "ollama": {
        "api_url": "http://localhost:11434/api/chat",
        "default_model": "phi",
        "headers": lambda: {
            "Content-Type": "application/json"
        }
    },
    "exo": {
        "api_url": "http://127.0.0.1:52415/v1/chat/completions",
        "default_model": "llama-3.2-1b",
        # "api_key": os.getenv("OPENAI_API_KEY"),
        "headers": lambda: {
            # "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
    },
}

# ===============================
# Helper Function to Get Config
# ===============================
def get_model_config():
    return MODELS[LLM_PROVIDER]