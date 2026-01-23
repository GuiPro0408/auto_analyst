# Auto-Analyst

An autonomous research assistant powered by a LangGraph RAG pipeline. Plans queries, searches the web, chunks and embeds content, retrieves context, generates cited answers, and verifies claims—all using free/open-source components.

## Features

- **Multi-turn Conversation Memory** — Maintains context across follow-up questions
- **Query Classification** — Automatically routes queries (factual, recommendation, creative) to appropriate prompts
- **Adaptive Research** — Iteratively refines search when initial results are insufficient
- **Quality Control** — Automatic assessment and improvement of generated answers
- **Cross-encoder Reranking** — Optional reranking for improved retrieval quality
- **Gemini Grounding Fast Path** — Direct answers from Gemini's web-grounded responses
- **API Key Rotation** — Automatic rotation through multiple API keys on rate limits
- **Query Result Caching** — SQLite-backed cache with TTL expiration
- **Comprehensive Logging** — Structured logs with run correlation IDs

## Quick Start

```bash
# Clone and setup
git clone https://github.com/<your-username>/auto-analyst.git && cd auto-analyst
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Configure API keys (create .env file)
echo "GOOGLE_API_KEY=your_key_here" >> .env

# Run
streamlit run ui/app.py  # http://localhost:8501
```

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                     │
│                     "What are the effects of X?"                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  QUERY CLASSIFY   │
                          │  factual/recom-   │
                          │  mendation/       │
                          │  creative         │
                          └─────────┬─────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PLAN                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Heuristic planner decomposes query into SearchQuery tasks           │   │
│  │ Detects time-sensitivity, topic, and conversation context           │   │
│  │ Example: ["effects of X on Y", "X statistics 2024"]                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/planner.py → List[SearchQuery]                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SEARCH                                                                     │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐                     │
│  │    Tavily    │  │ Gemini Ground  │  │   Smart     │                     │
│  │   (API)      │  │ (Google search)│  │   Search    │                     │
│  └──────────────┘  └────────────────┘  └─────────────┘                     │
│  tools/search.py + tools/smart_search.py → List[SearchResult]               │
│                                                                             │
│  Features: Domain filtering, deduplication, LLM result validation           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FETCH                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ robots.txt check → Parallel download HTML/PDF → Parse content       │   │
│  │ Configurable concurrency, retries, and backoff                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/fetcher.py + tools/parser.py → List[Document]                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CHUNK                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Token-aware splitting (tiktoken) with configurable overlap          │   │
│  │ Metadata preserved: url, title, media_type, chunk_index             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/chunker.py → List[Chunk]                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  EMBED & STORE                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ sentence-transformers (all-MiniLM-L6-v2)                            │   │
│  │         ↓                                                            │   │
│  │ ┌─────────────┐  or  ┌─────────────┐                                │   │
│  │ │  ChromaDB   │      │    FAISS    │                                │   │
│  │ │ (persistent)│      │ (in-memory) │                                │   │
│  │ └─────────────┘      └─────────────┘                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  vector_store/*.py → VectorStore.upsert(chunks)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RETRIEVE + RERANK                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Cosine similarity search → Top-K chunks (default K=6)               │   │
│  │ Optional cross-encoder reranking (ms-marco-MiniLM-L-6-v2)           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/retriever.py + tools/reranker.py → List[ScoredChunk]                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │    ADAPTIVE       │
                          │  Assess context   │
                          │  relevance, may   │
                          │  trigger re-search│
                          └─────────┬─────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GENERATE                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Gemini 2.0 Flash (default) + Context → Answer with [n] citations    │   │
│  │                                                                      │   │
│  │ Query-type-specific prompts:                                         │   │
│  │ • Factual: Strict RAG with mandatory citations                       │   │
│  │ • Recommendation: LLM knowledge + RAG for suggestions                │   │
│  │ • Creative: Flexible LLM response with optional citations            │   │
│  │                                                                      │   │
│  │ Fast path: Use Gemini grounded answer directly when available        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/generator.py:generate_answer() → (answer, citations)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VERIFY                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LLM reviews draft → Removes unsupported claims → Final answer       │   │
│  │ Preserves structure, formatting, and level of detail                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/generator.py:verify_answer() → verified_answer                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  QUALITY CONTROL  │
                          │  Assess answer    │
                          │  quality, may     │
                          │  trigger re-gen   │
                          └─────────┬─────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESEARCH STATE                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                    │   │
│  │   query, query_type, plan, search_results, documents, chunks,       │   │
│  │   retrieved, retrieval_scores, draft_answer, verified_answer,       │   │
│  │   citations, errors, warnings, adaptive_iterations, qc_passes,      │   │
│  │   conversation_history, grounded_answer, grounded_sources           │   │
│  │ }                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  api/state.py:ResearchState                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STREAMLIT UI                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Answer with inline [1][2] citations + expandable source list        │   │
│  │ Conversation memory, API key status, debug panel                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ui/app.py                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage Summary

| Stage          | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| **Classify**   | Routes query to factual/recommendation/creative mode           |
| **Plan**       | Decomposes question into targeted search tasks                 |
| **Search**     | Queries Tavily and/or Gemini Grounding with smart filtering    |
| **Fetch**      | Downloads pages/PDFs in parallel, respecting robots.txt        |
| **Chunk**      | Token-aware splitting with metadata preservation               |
| **Retrieve**   | Vector similarity search with optional cross-encoder reranking |
| **Adaptive**   | Assesses context quality, triggers re-search if needed         |
| **Generate**   | LLM produces answer with `[n]` citations (query-type-aware)    |
| **Verify**     | Prunes unsupported claims while preserving structure           |
| **QC**         | Quality assessment and iterative improvement                   |

## Configuration

### Core Settings

| Variable                         | Default                            | Purpose                              |
| -------------------------------- | ---------------------------------- | ------------------------------------ |
| `AUTO_ANALYST_LLM`               | `gemini-2.0-flash`                 | LLM model identifier                 |
| `AUTO_ANALYST_LLM_BACKEND`       | `gemini`                           | LLM backend (`gemini`/`huggingface`) |
| `AUTO_ANALYST_EMBED`             | `all-MiniLM-L6-v2`                 | Embedding model                      |
| `AUTO_ANALYST_VECTOR_STORE`      | `chroma`                           | Vector store (`chroma`/`faiss`)      |
| `AUTO_ANALYST_TOP_K`             | `6`                                | Retrieved chunks per query           |

### Search Settings

| Variable                         | Default                            | Purpose                              |
| -------------------------------- | ---------------------------------- | ------------------------------------ |
| `AUTO_ANALYST_SEARCH_BACKENDS`   | `tavily,gemini_grounding`          | Comma-separated search backends      |
| `AUTO_ANALYST_SMART_SEARCH`      | `true`                             | LLM-assisted query analysis          |
| `AUTO_ANALYST_VALIDATE_RESULTS`  | `true`                             | LLM filtering of irrelevant hits     |
| `AUTO_ANALYST_SEARCH_FALLBACK`   | `true`                             | Fallback on rate limits              |

### Pipeline Settings

| Variable                         | Default    | Purpose                              |
| -------------------------------- | ---------- | ------------------------------------ |
| `AUTO_ANALYST_ADAPTIVE_MAX_ITERS`| `2`        | Max adaptive search cycles           |
| `AUTO_ANALYST_QC_MAX_PASSES`     | `1`        | Max quality control passes           |
| `AUTO_ANALYST_CHUNK_SIZE`        | `1000`     | Chunk size in tokens                 |
| `AUTO_ANALYST_CHUNK_OVERLAP`     | `200`      | Chunk overlap in tokens              |
| `AUTO_ANALYST_ENABLE_RERANK`     | `true`     | Enable cross-encoder reranking       |
| `AUTO_ANALYST_RERANK_MODEL`      | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model        |

### Fetcher Settings

| Variable                         | Default    | Purpose                              |
| -------------------------------- | ---------- | ------------------------------------ |
| `AUTO_ANALYST_FETCH_RETRIES`     | `2`        | Retry attempts per URL               |
| `AUTO_ANALYST_FETCH_BACKOFF`     | `1.0`      | Backoff factor (seconds)             |
| `AUTO_ANALYST_FETCH_CONCURRENCY` | `5`        | Parallel fetch workers               |
| `AUTO_ANALYST_FETCH_TIMEOUT`     | `15`       | Fetch timeout (seconds)              |
| `AUTO_ANALYST_MIN_CONTENT_LENGTH`| `200`      | Min chars for valid document         |

### Cache Settings

| Variable                         | Default               | Purpose                    |
| -------------------------------- | --------------------- | -------------------------- |
| `AUTO_ANALYST_CACHE_PATH`        | `data/query_cache.sqlite3` | Cache database path   |
| `AUTO_ANALYST_CACHE_TTL`         | `7200`                | Cache TTL (seconds)        |
| `AUTO_ANALYST_CACHE_MAX_ENTRIES` | `1000`                | Max cached entries         |

### Conversation Memory

| Variable                              | Default | Purpose                         |
| ------------------------------------- | ------- | ------------------------------- |
| `AUTO_ANALYST_MEMORY_TURNS`           | `5`     | Conversation turns to remember  |
| `AUTO_ANALYST_MEMORY_SUMMARY_CHARS`   | `1200`  | Max chars in history summary    |
| `AUTO_ANALYST_ANSWER_PREVIEW_MAX_LEN` | `280`   | Answer preview length in memory |

### Logging Settings

| Variable                         | Default            | Purpose                           |
| -------------------------------- | ------------------ | --------------------------------- |
| `AUTO_ANALYST_LOG_LEVEL`         | `DEBUG`            | Log level                         |
| `AUTO_ANALYST_LOG_FORMAT`        | `plain`            | Log format (`plain`/`json`)       |
| `AUTO_ANALYST_LOG_FILE`          | `auto_analyst.log` | Log file path                     |
| `AUTO_ANALYST_LOG_REDACT_QUERIES`| `false`            | Redact queries in logs            |

### API Keys & Secrets

API credentials **must be supplied through environment variables** (never hard-code them):

| Variable               | Required For                              |
| ---------------------- | ----------------------------------------- |
| `GOOGLE_API_KEY`       | Gemini LLM and grounding (single key)     |
| `GOOGLE_API_KEYS`      | Multiple Gemini keys for rotation         |
| `HUGGINGFACE_API_TOKEN`| HuggingFace Inference backend             |
| `TAVILY_API_KEY`       | Tavily search backend                     |

Create a `.env` file in the project root:

```bash
# Required: At least one Gemini API key
GOOGLE_API_KEY=your_gemini_key

# Optional: Multiple keys for rate limit rotation
GOOGLE_API_KEYS=key1,key2,key3

# Optional: Alternative backends
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx

# Recommended settings
AUTO_ANALYST_LLM_BACKEND=gemini
AUTO_ANALYST_SMART_SEARCH=true
AUTO_ANALYST_ENABLE_RERANK=true
```

**Security Note:** Keep `.env` out of version control and rotate any exposed keys.

## Commands

```bash
# Setup
source .venv/bin/activate          # Activate virtualenv

# Run
streamlit run ui/app.py            # Run UI (http://localhost:8501)

# Testing
pytest                             # Run all tests
pytest -v                          # Verbose output
pytest -k "planner"                # Filter by name
pytest --cov=api --cov=tools       # With coverage

# Evaluation
python evaluation/run_evaluation.py --dataset data/sample_eval.json --model all-MiniLM-L6-v2
```

## Project Structure

```
api/                → Orchestration, state management, caching
  graph.py          # LangGraph pipeline nodes and edges
  state.py          # Dataclasses and TypedDict definitions
  state_builder.py  # State construction helpers
  config.py         # Central configuration
  logging_setup.py  # Structured logging with run correlation
  cache_manager.py  # Query result caching
  cache.py          # Cache encoding/decoding
  key_rotator.py    # API key rotation for rate limits
  memory.py         # Conversation history management

tools/              → Functional pipeline components
  planner.py        # Query decomposition into search tasks
  search.py         # Multi-backend web search
  search_backends.py# Backend implementations (Gemini, Tavily)
  search_filters.py # Result filtering and deduplication
  smart_search.py   # LLM-powered search pipeline
  fetcher.py        # URL fetching with robots.txt compliance
  parser.py         # HTML/PDF content extraction
  chunker.py        # Token-aware text splitting
  generator.py      # LLM answer generation with citations
  models.py         # LLM and embedding model loading
  reranker.py       # Cross-encoder reranking
  retriever.py      # Vector similarity search
  gemini_grounding.py # Gemini web-grounded responses
  query_classifier.py # Query type classification
  quality_control.py  # Answer quality assessment
  adaptive_research.py# Context assessment and plan refinement
  text_utils.py     # Shared text utilities
  topic_utils.py    # Topic detection

vector_store/       → Storage abstractions
  base.py           # VectorStore abstract interface
  chroma_store.py   # ChromaDB implementation (persistent)
  faiss_store.py    # FAISS implementation (in-memory)

ui/                 → Streamlit interface
  app.py            # Main application

evaluation/         → RAG evaluation metrics
  metrics.py        # Context relevance, answer correctness, hallucination
  run_evaluation.py # Evaluation runner

tests/              → pytest test suite
```

## Evaluation Metrics

The evaluation module (`evaluation/metrics.py`) provides embedding-based RAG metrics:

| Metric                   | Range | Interpretation                                      |
| ------------------------ | ----- | --------------------------------------------------- |
| **Context Relevance**    | 0-1   | Avg similarity between query and retrieved contexts |
| **Context Sufficiency**  | 0-1   | Fraction of contexts above relevance threshold      |
| **Answer Relevance**     | 0-1   | Similarity between generated and reference answers  |
| **Answer Correctness**   | 0-1   | Direct similarity to ground truth                   |
| **Answer Hallucination** | 0-1   | Fraction of unsupported sentences (lower is better) |

## Architecture Highlights

### Query Classification

Queries are automatically classified to optimize answer generation:

- **Factual** — Strict RAG with mandatory citations (news, research, technical)
- **Recommendation** — LLM knowledge enhanced by RAG context (suggestions, opinions)
- **Creative** — Primarily LLM knowledge with optional citations (brainstorming)

### Adaptive Research

When initial retrieval produces insufficient or low-relevance results:

1. `assess_context()` evaluates chunk count and relevance scores
2. `refine_plan()` generates additional search tasks
3. Pipeline re-executes search → fetch → retrieve cycle
4. Maximum iterations configurable via `AUTO_ANALYST_ADAPTIVE_MAX_ITERS`

### Gemini Grounding Fast Path

When Gemini's Google Search grounding returns a direct answer:

1. The grounded answer is extracted from search results
2. Generation uses the grounded answer directly (skips full RAG)
3. Citations are built from grounding sources
4. Significantly faster for queries with strong web results

### API Key Rotation

For high-volume usage with rate limits:

1. Configure multiple keys: `GOOGLE_API_KEYS=key1,key2,key3`
2. `APIKeyRotator` automatically rotates on 429 errors
3. Keys are reset after successful requests
4. UI shows real-time key availability status

## Prerequisites

- Python 3.11+
- ~8GB disk for models (Phi-3/Mistral + embeddings)

## Notes

- **No paid APIs**—all search, models, and embeddings are free
- **robots.txt compliance**—fetcher respects site restrictions
- See `ressources/*.md` for technical design and evaluation methodology
- Adaptive research: automatically broadens search when context is thin (capped by `AUTO_ANALYST_ADAPTIVE_MAX_ITERS`).
- Quality control: optional refinement loop to improve answers up to `AUTO_ANALYST_QC_MAX_PASSES`.
