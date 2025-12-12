"""Central configuration for Auto-Analyst."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# LLMs now run via managed cloud providers only (no local inference path).
# Default to Gemini Flash for low-latency, low-cost remote execution.
DEFAULT_LLM_MODEL = os.getenv("AUTO_ANALYST_LLM", "gemini-2.0-flash")
DEFAULT_EMBED_MODEL = os.getenv("AUTO_ANALYST_EMBED", "all-MiniLM-L6-v2")
LLM_BACKEND = os.getenv("AUTO_ANALYST_LLM_BACKEND", "gemini").lower()
GEMINI_MODEL = os.getenv("AUTO_ANALYST_GEMINI_MODEL", "gemini-2.0-flash")
# Support multiple API keys for rotation on rate limits
_api_keys_str = os.getenv("GOOGLE_API_KEYS", "")
GEMINI_API_KEYS: list[str] = (
    [k.strip() for k in _api_keys_str.split(",") if k.strip()] if _api_keys_str else []
)
# Fallback to single key if GOOGLE_API_KEYS not set
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GEMINI_API_KEYS and GEMINI_API_KEY:
    GEMINI_API_KEYS = [GEMINI_API_KEY]
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
HUGGINGFACE_INFERENCE_MODEL = os.getenv(
    "AUTO_ANALYST_HF_INFERENCE_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1"
)
VECTOR_STORE_BACKEND = os.getenv("AUTO_ANALYST_VECTOR_STORE", "chroma")
SEARCH_BACKENDS = os.getenv("AUTO_ANALYST_SEARCH_BACKENDS", "gemini_grounding").split(
    ","
)
SMART_SEARCH_ENABLED = os.getenv("AUTO_ANALYST_SMART_SEARCH", "true").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
VALIDATE_RESULTS_ENABLED = os.getenv("AUTO_ANALYST_VALIDATE_RESULTS", "true").lower() == "true"

# INT8 quantization is disabled by default because it's slow on Pascal GPUs (GTX 10xx).
# Only enable on Ampere (RTX 30xx) or newer GPUs with native INT8 tensor cores.
# Set AUTO_ANALYST_USE_INT8=true to enable if you have a newer GPU.
USE_INT8_QUANTIZATION = os.getenv("AUTO_ANALYST_USE_INT8", "false").lower() == "true"

# Minimum VRAM (in GB) recommended for the default LLM model
# Qwen2.5-3B-Instruct FP16 needs ~6GB VRAM
MIN_RECOMMENDED_VRAM_GB = 6
MIN_RECOMMENDED_MEMORY_GB = 12

LOG_LEVEL = os.getenv("AUTO_ANALYST_LOG_LEVEL", "DEBUG")
LOG_FORMAT = os.getenv("AUTO_ANALYST_LOG_FORMAT", "plain")  # plain|json
LOG_FILE_PATH = os.getenv("AUTO_ANALYST_LOG_FILE", "auto_analyst.log")
SEARCH_RATE_LIMIT_SECONDS = float(os.getenv("AUTO_ANALYST_SEARCH_RATE_LIMIT", "1.0"))
SEARCH_RETRIES = int(os.getenv("AUTO_ANALYST_SEARCH_RETRIES", "2"))
FETCH_RETRIES = int(os.getenv("AUTO_ANALYST_FETCH_RETRIES", "2"))
FETCH_BACKOFF_SECONDS = float(os.getenv("AUTO_ANALYST_FETCH_BACKOFF", "1.0"))
FETCH_CONCURRENCY = int(os.getenv("AUTO_ANALYST_FETCH_CONCURRENCY", "5"))
ADAPTIVE_MAX_ITERS = int(os.getenv("AUTO_ANALYST_ADAPTIVE_MAX_ITERS", "2"))
QC_MAX_PASSES = int(os.getenv("AUTO_ANALYST_QC_MAX_PASSES", "1"))

CHUNK_SIZE = int(os.getenv("AUTO_ANALYST_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("AUTO_ANALYST_CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("AUTO_ANALYST_TOP_K", "6"))
USER_AGENT = os.getenv(
    "AUTO_ANALYST_USER_AGENT", "auto-analyst/1.0 (+https://github.com)"
)
DATA_DIR = BASE_DIR / "data"

CACHE_DB_PATH = Path(
    os.getenv("AUTO_ANALYST_CACHE_PATH", str(DATA_DIR / "query_cache.sqlite3"))
)
CACHE_TTL_SECONDS = int(os.getenv("AUTO_ANALYST_CACHE_TTL", "7200"))

# Generation parameters - max_new_tokens=1024 allows for detailed, structured responses
GENERATION_KWARGS = {
    "max_new_tokens": 1024,
    "temperature": 0.4,
    "do_sample": True,
}

ENABLE_RERANKER = os.getenv("AUTO_ANALYST_ENABLE_RERANK", "true").lower() == "true"
RERANK_MODEL_NAME = os.getenv(
    "AUTO_ANALYST_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

CONVERSATION_MEMORY_TURNS = int(os.getenv("AUTO_ANALYST_MEMORY_TURNS", "5"))
CONVERSATION_SUMMARY_CHARS = int(os.getenv("AUTO_ANALYST_MEMORY_SUMMARY_CHARS", "1200"))
