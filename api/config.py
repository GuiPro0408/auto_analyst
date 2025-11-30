"""Central configuration for Auto-Analyst."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Default to a lighter-weight instruct model to reduce startup time.
DEFAULT_LLM_MODEL = os.getenv("AUTO_ANALYST_LLM", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEFAULT_EMBED_MODEL = os.getenv("AUTO_ANALYST_EMBED", "all-MiniLM-L6-v2")
VECTOR_STORE_BACKEND = os.getenv("AUTO_ANALYST_VECTOR_STORE", "chroma")
SEARCH_BACKENDS = os.getenv(
    "AUTO_ANALYST_SEARCH_BACKENDS", "duckduckgo,wikipedia"
).split(",")

LOG_LEVEL = os.getenv("AUTO_ANALYST_LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("AUTO_ANALYST_LOG_FORMAT", "plain")  # plain|json
LOG_FILE_PATH = os.getenv("AUTO_ANALYST_LOG_FILE", "auto_analyst.log")
SEARCH_RATE_LIMIT_SECONDS = float(os.getenv("AUTO_ANALYST_SEARCH_RATE_LIMIT", "1.0"))
SEARCH_RETRIES = int(os.getenv("AUTO_ANALYST_SEARCH_RETRIES", "2"))
FETCH_RETRIES = int(os.getenv("AUTO_ANALYST_FETCH_RETRIES", "2"))
FETCH_BACKOFF_SECONDS = float(os.getenv("AUTO_ANALYST_FETCH_BACKOFF", "1.0"))
ADAPTIVE_MAX_ITERS = int(os.getenv("AUTO_ANALYST_ADAPTIVE_MAX_ITERS", "2"))
QC_MAX_PASSES = int(os.getenv("AUTO_ANALYST_QC_MAX_PASSES", "1"))

CHUNK_SIZE = int(os.getenv("AUTO_ANALYST_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("AUTO_ANALYST_CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("AUTO_ANALYST_TOP_K", "6"))
USER_AGENT = os.getenv(
    "AUTO_ANALYST_USER_AGENT", "auto-analyst/1.0 (+https://github.com)"
)
DATA_DIR = BASE_DIR / "data"

GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.4,
    "do_sample": True,
}
