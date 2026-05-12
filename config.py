import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# Gemini 2.5 Flash pricing (USD per 1M tokens, thinking disabled)
INPUT_COST_PER_MILLION  = 0.075
OUTPUT_COST_PER_MILLION = 0.30

# Set thinking_budget=0 to disable chain-of-thought (saves tokens + cost)
THINKING_BUDGET = 0

# --- Groq (used by Pipeline 2 batch runner) ---
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.3-70b-versatile"

# Groq llama-3.3-70b pricing (USD per 1M tokens)
GROQ_INPUT_COST_PER_MILLION  = 0.59
GROQ_OUTPUT_COST_PER_MILLION = 0.79

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
CORPUS_FILE = BASE_DIR / "data" / "corpus.txt"
EVALUATION_DIR = BASE_DIR / "evaluation"
RESULTS_DIR = BASE_DIR / "results"

# --- Data collection ---
WIKIPEDIA_DELAY = 0.5   # seconds between Wikipedia requests
MAX_RETRIES = 3
RETRY_DELAY = 60        # seconds to wait on rate limit

# --- Gemini generation ---
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.1
