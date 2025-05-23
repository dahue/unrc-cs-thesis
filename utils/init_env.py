from dotenv import load_dotenv
import os
import sys

load_dotenv()

# Add project root to sys.path
project_path = os.getenv("PYTHONPATH")
if project_path and project_path not in sys.path:
    sys.path.append(project_path)
