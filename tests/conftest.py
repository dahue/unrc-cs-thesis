import os
import pytest
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture(scope="session")
def root_path():
    path = os.environ.get("ROOT_PATH")
    if not path:
        pytest.skip("ROOT_PATH not set in .env")
    return path

@pytest.fixture(scope="session")
def db(root_path):
    path = f"{root_path}/database/OpenText2SQL.db"
    if not os.path.exists(path):
        pytest.skip(f"Database not found at {path}. Run src/pipeline/ingest.py first.")
    return path

@pytest.fixture(scope="session")
def gold_db(db):
    return db

@pytest.fixture(scope="session")
def index_db(db):
    return db
