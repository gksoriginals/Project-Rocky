import os
from dotenv import load_dotenv
from rocky.utils import load_text_file

load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:e2b")

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

SYSTEM_PROMPT = load_text_file(os.path.join(PROMPTS_DIR, "system_prompt.txt"))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "rocky_memory.sqlite3")
