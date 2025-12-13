"""Central configuration for Auto-Analyst.

Configuration is organized into logical groups:
- Path configuration
- LLM configuration
- Search configuration
- Logging configuration
- Fetcher configuration
- Pipeline configuration
- Cache configuration
- Generator configuration
- Reranker configuration
- Conversation memory configuration
"""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
# LLMs run via managed cloud providers only (no local inference path).
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

# Generation parameters - max_new_tokens=1024 allows for detailed, structured responses
GENERATION_KWARGS: Dict[str, Any] = {
    "max_new_tokens": 1024,
    "temperature": 0.4,
    "do_sample": True,
}

# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================
VECTOR_STORE_BACKEND = os.getenv("AUTO_ANALYST_VECTOR_STORE", "chroma")
SEARCH_BACKENDS = os.getenv(
    "AUTO_ANALYST_SEARCH_BACKENDS", "tavily,gemini_grounding"
).split(",")
SMART_SEARCH_ENABLED = os.getenv("AUTO_ANALYST_SMART_SEARCH", "true").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
VALIDATE_RESULTS_ENABLED = (
    os.getenv("AUTO_ANALYST_VALIDATE_RESULTS", "true").lower() == "true"
)
# When True, falls back to alternative search backends (e.g., Tavily) when
# Gemini API keys are exhausted due to rate limits
SEARCH_FALLBACK_ON_RATE_LIMIT = (
    os.getenv("AUTO_ANALYST_SEARCH_FALLBACK", "true").lower() == "true"
)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = os.getenv("AUTO_ANALYST_LOG_LEVEL", "DEBUG")
LOG_FORMAT = os.getenv("AUTO_ANALYST_LOG_FORMAT", "plain")  # plain|json
LOG_FILE_PATH = os.getenv("AUTO_ANALYST_LOG_FILE", "auto_analyst.log")
LOG_REDACT_QUERIES = (
    os.getenv("AUTO_ANALYST_LOG_REDACT_QUERIES", "false").lower() == "true"
)

# =============================================================================
# FETCHER CONFIGURATION
# =============================================================================
FETCH_RETRIES = int(os.getenv("AUTO_ANALYST_FETCH_RETRIES", "2"))
FETCH_BACKOFF_SECONDS = float(os.getenv("AUTO_ANALYST_FETCH_BACKOFF", "1.0"))
FETCH_CONCURRENCY = int(os.getenv("AUTO_ANALYST_FETCH_CONCURRENCY", "5"))
USER_AGENT = os.getenv(
    "AUTO_ANALYST_USER_AGENT", "auto-analyst/1.0 (+https://github.com)"
)
# Robots.txt handling: "block" (default) or "allow" on parse errors
ROBOTS_ON_ERROR = os.getenv("AUTO_ANALYST_ROBOTS_ON_ERROR", "block").lower()

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
ADAPTIVE_MAX_ITERS = int(os.getenv("AUTO_ANALYST_ADAPTIVE_MAX_ITERS", "2"))
QC_MAX_PASSES = int(os.getenv("AUTO_ANALYST_QC_MAX_PASSES", "1"))
CHUNK_SIZE = int(os.getenv("AUTO_ANALYST_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("AUTO_ANALYST_CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("AUTO_ANALYST_TOP_K", "6"))

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================
CACHE_DB_PATH = Path(
    os.getenv("AUTO_ANALYST_CACHE_PATH", str(DATA_DIR / "query_cache.sqlite3"))
)
CACHE_TTL_SECONDS = int(os.getenv("AUTO_ANALYST_CACHE_TTL", "7200"))
CACHE_MAX_ENTRIES = int(os.getenv("AUTO_ANALYST_CACHE_MAX_ENTRIES", "1000"))

# =============================================================================
# GENERATOR CONFIGURATION (coherence thresholds)
# =============================================================================
COHERENCE_MIN_WORDS = int(os.getenv("AUTO_ANALYST_COHERENCE_MIN_WORDS", "20"))
COHERENCE_MAX_REPETITION_RATIO = float(
    os.getenv("AUTO_ANALYST_COHERENCE_MAX_REPETITION", "0.4")
)
COHERENCE_MIN_ALNUM_RATIO = float(os.getenv("AUTO_ANALYST_COHERENCE_MIN_ALNUM", "0.65"))
COHERENCE_MAX_WORD_REPEAT = int(
    os.getenv("AUTO_ANALYST_COHERENCE_MAX_WORD_REPEAT", "8")
)

# =============================================================================
# RERANKER CONFIGURATION
# =============================================================================
ENABLE_RERANKER = os.getenv("AUTO_ANALYST_ENABLE_RERANK", "true").lower() == "true"
RERANK_MODEL_NAME = os.getenv(
    "AUTO_ANALYST_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# =============================================================================
# CONVERSATION MEMORY CONFIGURATION
# =============================================================================
CONVERSATION_MEMORY_TURNS = int(os.getenv("AUTO_ANALYST_MEMORY_TURNS", "5"))
CONVERSATION_SUMMARY_CHARS = int(os.getenv("AUTO_ANALYST_MEMORY_SUMMARY_CHARS", "1200"))
CONVERSATION_CONTEXT_MAX_CHARS = int(
    os.getenv("AUTO_ANALYST_CONVERSATION_CONTEXT_MAX_CHARS", "800")
)
ANSWER_PREVIEW_MAX_LEN = int(os.getenv("AUTO_ANALYST_ANSWER_PREVIEW_MAX_LEN", "280"))

# =============================================================================
# ADAPTIVE RESEARCH CONFIGURATION
# =============================================================================
MIN_RELEVANCE_THRESHOLD = float(
    os.getenv("AUTO_ANALYST_MIN_RELEVANCE_THRESHOLD", "0.3")
)

# =============================================================================
# QUALITY CONTROL CONFIGURATION
# =============================================================================
QC_MIN_RELEVANCE_THRESHOLD = float(
    os.getenv("AUTO_ANALYST_QC_MIN_RELEVANCE_THRESHOLD", "0.25")
)

# =============================================================================
# SEARCH CONFIGURATION (snippet lengths)
# =============================================================================
GROUNDING_SNIPPET_PREVIEW_LEN = int(
    os.getenv("AUTO_ANALYST_GROUNDING_SNIPPET_PREVIEW_LEN", "200")
)
SYNTHETIC_SNIPPET_MAX_LEN = int(
    os.getenv("AUTO_ANALYST_SYNTHETIC_SNIPPET_MAX_LEN", "300")
)
TAVILY_SNIPPET_MAX_LEN = int(os.getenv("AUTO_ANALYST_TAVILY_SNIPPET_MAX_LEN", "500"))

# =============================================================================
# LLM/GROUNDING RETRY CONFIGURATION
# =============================================================================
LLM_BACKOFF_SECONDS = float(os.getenv("AUTO_ANALYST_LLM_BACKOFF_SECONDS", "0.5"))
GROUNDING_RETRY_DELAY = float(os.getenv("AUTO_ANALYST_GROUNDING_RETRY_DELAY", "0.5"))

# =============================================================================
# FETCH CONFIGURATION
# =============================================================================
FETCH_TIMEOUT = int(os.getenv("AUTO_ANALYST_FETCH_TIMEOUT", "15"))

# =============================================================================
# ROBOTS.TXT CACHE CONFIGURATION
# =============================================================================
ROBOTS_CACHE_TTL_SECONDS = int(os.getenv("AUTO_ANALYST_ROBOTS_CACHE_TTL", "1800"))
