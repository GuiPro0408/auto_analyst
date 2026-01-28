<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/LangGraph-RAG%20Pipeline-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Gemini-2.0%20Flash-orange.svg" alt="Gemini">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

# 🔬 Auto-Analyst

> **An autonomous research assistant powered by a LangGraph RAG pipeline.**
>
> Plans queries, searches the web, chunks and embeds content, retrieves context, generates cited answers, and verifies claims—all using free/open-source components.

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Intelligence
- **Multi-turn Conversation Memory** — Maintains context across follow-up questions
- **Query Classification** — Routes queries (factual/recommendation/creative) to appropriate prompts
- **Adaptive Research** — Iteratively refines search when results are insufficient
- **Quality Control** — Automatic assessment and improvement of answers

</td>
<td width="50%">

### 🔍 Retrieval
- **Hybrid Search** — BM25 lexical + semantic embeddings with Reciprocal Rank Fusion
- **Contextual Chunking** — LLM-generated context per chunk (Anthropic's approach)
- **Cross-encoder Reranking** — Optional reranking for improved quality
- **Gemini Grounding Fast Path** — Direct answers from web-grounded responses

</td>
</tr>
<tr>
<td width="50%">

### 🖥️ User Experience
- **Streaming Responses** — Real-time answer generation with Chainlit UI
- **Chat Persistence** — SQLite-backed conversation history
- **Multiple LLM Backends** — Gemini, Groq, HuggingFace, OpenAI-compatible

</td>
<td width="50%">

### ⚙️ Operations
- **API Key Rotation** — Automatic rotation on rate limits
- **Query Result Caching** — SQLite cache with TTL expiration
- **Comprehensive Logging** — Structured logs with run correlation IDs

</td>
</tr>
</table>

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/<your-username>/auto-analyst.git && cd auto-analyst
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Configure API keys (create .env file)
echo "GOOGLE_API_KEY=your_key_here" >> .env

# Run (choose one)
streamlit run ui/app.py              # Streamlit UI (http://localhost:8501)
chainlit run ui/chainlit_app.py -w   # Chainlit UI with streaming (http://localhost:8000)
```

---

## 🔄 Pipeline

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
│  │ Optional contextual chunking: LLM adds document context per chunk   │   │
│  │ Metadata preserved: url, title, media_type, chunk_index             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/chunker.py + tools/contextual_chunker.py → List[Chunk]               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  EMBED & STORE                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ sentence-transformers (BAAI/bge-small-en-v1.5)                      │   │
│  │         ↓                                                            │   │
│  │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │ │  ChromaDB   │  │    FAISS    │  │   Hybrid    │                   │   │
│  │ │ (persistent)│  │ (in-memory) │  │ (BM25+Emb)  │                   │   │
│  │ └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  vector_store/*.py → VectorStore.upsert(chunks)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RETRIEVE + RERANK                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Hybrid search: BM25 + semantic with Reciprocal Rank Fusion          │   │
│  │ Cosine similarity search → Top-K chunks (default K=12)              │   │
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
│  ui/app.py (Streamlit) or ui/chainlit_app.py (Chainlit with streaming)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

<details>
<summary><strong>📋 Stage Summary</strong> (click to expand)</summary>

| Stage | Description |
|:------|:------------|
| 🏷️ **Classify** | Routes query to factual/recommendation/creative mode |
| 📝 **Plan** | Decomposes question into targeted search tasks |
| 🔍 **Search** | Queries Tavily and/or Gemini Grounding with smart filtering |
| 📥 **Fetch** | Downloads pages/PDFs in parallel, respecting robots.txt |
| ✂️ **Chunk** | Token-aware splitting with optional contextual enrichment |
| 🎯 **Retrieve** | Hybrid search (BM25+semantic) with optional cross-encoder rerank |
| 🔄 **Adaptive** | Assesses context quality, triggers re-search if needed |
| ✍️ **Generate** | LLM produces answer with `[n]` citations (query-type-aware) |
| ✅ **Verify** | Prunes unsupported claims while preserving structure |
| 🏆 **QC** | Quality assessment and iterative improvement |

</details>

---

## ⚙️ Configuration

### 🔑 Core Settings

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `AUTO_ANALYST_LLM` | `gemini-2.0-flash` | LLM model identifier |
| `AUTO_ANALYST_LLM_BACKEND` | `gemini` | LLM backend (`gemini`/`groq`/`huggingface`) |
| `AUTO_ANALYST_EMBED` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `AUTO_ANALYST_VECTOR_STORE` | `chroma` | Vector store (`chroma`/`faiss`) |
| `AUTO_ANALYST_TOP_K` | `12` | Retrieved chunks per query |

> **📝 Note:** The default embedding model was changed from `all-MiniLM-L6-v2` to `BAAI/bge-small-en-v1.5` for improved retrieval quality. ChromaDB automatically detects and rebuilds incompatible vector stores.

<details>
<summary><strong>🔍 Search Settings</strong></summary>

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `AUTO_ANALYST_SEARCH_BACKENDS` | `tavily,gemini_grounding` | Comma-separated search backends |
| `AUTO_ANALYST_SMART_SEARCH` | `true` | LLM-assisted query analysis |
| `AUTO_ANALYST_VALIDATE_RESULTS` | `true` | LLM filtering of irrelevant hits |
| `AUTO_ANALYST_SEARCH_FALLBACK` | `true` | Fallback on rate limits |

</details>

<details>
<summary><strong>🔄 Pipeline Settings</strong></summary>

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `AUTO_ANALYST_ADAPTIVE_MAX_ITERS` | `2` | Max adaptive search cycles |
| `AUTO_ANALYST_QC_MAX_PASSES` | `1` | Max quality control passes |
| `AUTO_ANALYST_CHUNK_SIZE` | `1000` | Chunk size in tokens |
| `AUTO_ANALYST_CHUNK_OVERLAP` | `200` | Chunk overlap in tokens |
| `AUTO_ANALYST_ENABLE_RERANK` | `true` | Enable cross-encoder reranking |
| `AUTO_ANALYST_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |

</details>

<details>
<summary><strong>🔀 Hybrid Search Settings</strong></summary>

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `AUTO_ANALYST_HYBRID_SEARCH` | `true` | Enable hybrid BM25 + semantic search |
| `AUTO_ANALYST_BM25_WEIGHT` | `0.3` | BM25 weight in rank fusion (0.0-1.0) |

</details>

<details>
<summary><strong>📄 Contextual Chunking Settings</strong></summary>

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `AUTO_ANALYST_CONTEXTUAL_CHUNKS` | `true` | Enable LLM-generated chunk context |
| `AUTO_ANALYST_CONTEXTUAL_MAX_CHUNKS_PER_DOC` | `4` | Max chunks to contextualize per document |
| `AUTO_ANALYST_CONTEXTUAL_DOCUMENT_CHAR_LIMIT` | `8000` | Max document chars for context prompt |
| `AUTO_ANALYST_CONTEXTUAL_CHUNK_CHAR_LIMIT` | `1200` | Max chunk chars for context generation |

</details>

<details>
<summary><strong>📥 Fetcher Settings</strong></summary>

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `AUTO_ANALYST_FETCH_RETRIES` | `2` | Retry attempts per URL |
| `AUTO_ANALYST_FETCH_BACKOFF` | `1.0` | Backoff factor (seconds) |
| `AUTO_ANALYST_FETCH_CONCURRENCY` | `5` | Parallel fetch workers |
| `AUTO_ANALYST_FETCH_TIMEOUT` | `15` | Fetch timeout (seconds) |
| `AUTO_ANALYST_MIN_CONTENT_LENGTH` | `200` | Min chars for valid document |

</details>

<details>
<summary><strong>💾 Cache Settings</strong></summary>

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `AUTO_ANALYST_CACHE_PATH` | `data/query_cache.sqlite3` | Cache database path |
| `AUTO_ANALYST_CACHE_TTL` | `7200` | Cache TTL (seconds) |
| `AUTO_ANALYST_CACHE_MAX_ENTRIES` | `1000` | Max cached entries |

</details>

<details>
<summary><strong>💬 Conversation Memory</strong></summary>

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `AUTO_ANALYST_MEMORY_TURNS` | `5` | Conversation turns to remember |
| `AUTO_ANALYST_MEMORY_SUMMARY_CHARS` | `1200` | Max chars in history summary |
| `AUTO_ANALYST_ANSWER_PREVIEW_MAX_LEN` | `280` | Answer preview length in memory |

</details>

<details>
<summary><strong>📊 Logging Settings</strong></summary>

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `AUTO_ANALYST_LOG_LEVEL` | `DEBUG` | Log level |
| `AUTO_ANALYST_LOG_FORMAT` | `plain` | Log format (`plain`/`json`) |
| `AUTO_ANALYST_LOG_FILE` | `auto_analyst.log` | Log file path |
| `AUTO_ANALYST_LOG_REDACT_QUERIES` | `false` | Redact queries in logs |

</details>

### 🔐 API Keys & Secrets

> ⚠️ **API credentials must be supplied through environment variables** (never hard-code them)

| Variable | Required For |
|:---------|:-------------|
| `GOOGLE_API_KEY` | Gemini LLM and grounding (single key) |
| `GOOGLE_API_KEYS` | Multiple Gemini keys for rotation |
| `GROQ_API_KEY` | Groq LLM backend |
| `GROQ_MODEL` | Groq model (default: `llama-3.3-70b-versatile`) |
| `HUGGINGFACE_API_TOKEN` | HuggingFace Inference backend |
| `TAVILY_API_KEY` | Tavily search backend |

<details>
<summary><strong>📋 Example .env file</strong></summary>

```bash
# Required: At least one Gemini API key
GOOGLE_API_KEY=your_gemini_key

# Optional: Multiple keys for rate limit rotation
GOOGLE_API_KEYS=key1,key2,key3

# Optional: Groq backend (fast inference)
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
AUTO_ANALYST_LLM_BACKEND=groq

# Optional: Alternative backends
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx

# Recommended settings
AUTO_ANALYST_LLM_BACKEND=gemini
AUTO_ANALYST_SMART_SEARCH=true
AUTO_ANALYST_ENABLE_RERANK=true
AUTO_ANALYST_HYBRID_SEARCH=true
AUTO_ANALYST_CONTEXTUAL_CHUNKS=true
```

</details>

> 🔒 **Security Note:** Keep `.env` out of version control and rotate any exposed keys.

---

## 🛠️ Commands

```bash
# Setup
source .venv/bin/activate          # Activate virtualenv

# Run
streamlit run ui/app.py            # Streamlit UI (http://localhost:8501)
chainlit run ui/chainlit_app.py -w # Chainlit UI with streaming (http://localhost:8000)

# Testing
pytest                             # Run all tests
pytest -v                          # Verbose output
pytest -k "planner"                # Filter by name
pytest --cov=api --cov=tools       # With coverage

# Evaluation
python evaluation/run_evaluation.py --dataset data/sample_eval.json --model BAAI/bge-small-en-v1.5
```

---

## 📁 Project Structure

```
📦 auto-analyst
├── 🔧 api/                 → Orchestration, state management, caching
│   ├── graph.py            # LangGraph pipeline nodes and edges (incl. streaming)
│   ├── state.py            # Dataclasses and TypedDict definitions
│   ├── state_builder.py    # State construction helpers
│   ├── config.py           # Central configuration
│   ├── logging_setup.py    # Structured logging with run correlation
│   ├── cache_manager.py    # Query result caching
│   ├── cache.py            # Cache encoding/decoding
│   ├── key_rotator.py      # API key rotation for rate limits
│   └── memory.py           # Conversation history management
│
├── 🛠️ tools/               → Functional pipeline components
│   ├── planner.py          # Query decomposition into search tasks
│   ├── search.py           # Multi-backend web search
│   ├── search_backends.py  # Backend implementations (Gemini, Tavily)
│   ├── search_filters.py   # Result filtering and deduplication
│   ├── smart_search.py     # LLM-powered search pipeline
│   ├── fetcher.py          # URL fetching with robots.txt compliance
│   ├── parser.py           # HTML/PDF content extraction
│   ├── chunker.py          # Token-aware text splitting
│   ├── contextual_chunker.py  # LLM-generated chunk context
│   ├── generator.py        # LLM answer generation with citations
│   ├── models.py           # LLM and embedding model loading
│   ├── openai_compatible_llm.py  # OpenAI-compatible API wrapper (Groq, etc.)
│   ├── reranker.py         # Cross-encoder reranking
│   ├── retriever.py        # Vector similarity search
│   ├── gemini_grounding.py # Gemini web-grounded responses
│   ├── query_classifier.py # Query type classification
│   ├── quality_control.py  # Answer quality assessment
│   ├── adaptive_research.py # Context assessment and plan refinement
│   ├── text_utils.py       # Shared text utilities
│   └── topic_utils.py      # Topic detection
│
├── 🗄️ vector_store/        → Storage abstractions
│   ├── base.py             # VectorStore abstract interface
│   ├── chroma_store.py     # ChromaDB implementation (persistent)
│   ├── faiss_store.py      # FAISS implementation (in-memory)
│   ├── bm25_store.py       # BM25 lexical search store
│   └── hybrid_store.py     # Hybrid BM25+semantic with RRF fusion
│
├── 🖥️ ui/                  → User interfaces
│   ├── app.py              # Streamlit application
│   ├── chainlit_app.py     # Chainlit app with streaming support
│   └── data_layer.py       # SQLite-backed chat persistence
│
├── 📊 evaluation/          → RAG evaluation metrics
│   ├── metrics.py          # Context relevance, answer correctness, hallucination
│   └── run_evaluation.py   # Evaluation runner
│
└── 🧪 tests/               → pytest test suite
```

---

## 📈 Evaluation Metrics

The evaluation module (`evaluation/metrics.py`) provides embedding-based RAG metrics:

| Metric | Range | Interpretation |
|:-------|:-----:|:---------------|
| 📊 **Context Relevance** | 0-1 | Avg similarity between query and retrieved contexts |
| 📚 **Context Sufficiency** | 0-1 | Fraction of contexts above relevance threshold |
| 🎯 **Answer Relevance** | 0-1 | Similarity between generated and reference answers |
| ✅ **Answer Correctness** | 0-1 | Direct similarity to ground truth |
| ⚠️ **Answer Hallucination** | 0-1 | Fraction of unsupported sentences (lower is better) |

---

## 🏗️ Architecture Highlights

### 🏷️ Query Classification

Queries are automatically classified to optimize answer generation:

| Type | Behavior | Example |
|:-----|:---------|:--------|
| **Factual** | Strict RAG with mandatory citations | News, research, technical |
| **Recommendation** | LLM knowledge enhanced by RAG context | Suggestions, opinions |
| **Creative** | Primarily LLM knowledge with optional citations | Brainstorming |

### 🔄 Adaptive Research

When initial retrieval produces insufficient or low-relevance results:

```
assess_context() → refine_plan() → re-search → fetch → retrieve
        ↑                                            │
        └────────────────────────────────────────────┘
                    (max iterations configurable)
```

1. `assess_context()` evaluates chunk count and relevance scores
2. `refine_plan()` generates additional search tasks
3. Pipeline re-executes search → fetch → retrieve cycle
4. Maximum iterations configurable via `AUTO_ANALYST_ADAPTIVE_MAX_ITERS`

### ⚡ Gemini Grounding Fast Path

When Gemini's Google Search grounding returns a direct answer:

```
Query → Gemini Grounding → Direct Answer → Citations from sources
              ↓
       (skips full RAG pipeline = faster)
```

### 🔀 Hybrid Search

Combines BM25 lexical search with semantic embeddings using **Reciprocal Rank Fusion (RRF)**:

| Method | Excels At |
|:-------|:----------|
| **BM25** | Exact keyword matches (error codes, technical terms, names) |
| **Semantic** | Meaning and synonyms |

Configurable weighting via `AUTO_ANALYST_BM25_WEIGHT` (default: 0.3)

### 📄 Contextual Chunking

Based on [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval):

1. 🤖 LLM generates 2-3 sentence context for each chunk
2. 📝 Context describes document topic, time period, and chunk's role
3. ➕ Context is prepended to chunk before embedding
4. 📈 Improves retrieval by preserving document-level information
5. 🛡️ Circuit breaker prevents excessive LLM calls on rate limits

### 🔑 API Key Rotation

For high-volume usage with rate limits:

```
Request → Key1 (429) → Key2 (429) → Key3 → Success → Reset all keys
```

1. Configure multiple keys: `GOOGLE_API_KEYS=key1,key2,key3`
2. `APIKeyRotator` automatically rotates on 429 errors
3. Keys are reset after successful requests
4. UI shows real-time key availability status

---

## 📋 Prerequisites

| Requirement | Details |
|:------------|:--------|
| **Python** | 3.11+ |
| **Disk Space** | ~4GB for models (embeddings + optional reranker) |

---

## 🖥️ UI Options

<table>
<tr>
<td width="50%">

### Streamlit (Basic)
```bash
streamlit run ui/app.py
```

- ✅ Simple chat interface
- 🔧 Debug panel with pipeline details
- 🔑 API key status display

</td>
<td width="50%">

### Chainlit (Production)
```bash
chainlit run ui/chainlit_app.py -w
```

- ⚡ **Streaming responses** — Real-time answer generation
- 💾 **Chat persistence** — SQLite-backed conversation history
- 📊 **Step visualization** — Shows pipeline progress
- 💬 **Multi-turn support** — Automatic conversation context

</td>
</tr>
</table>

---

## 📝 Notes

| | |
|:---|:---|
| 🆓 | **No paid APIs required** — uses Gemini free tier, open-source models |
| 🤖 | **robots.txt compliance** — fetcher respects site restrictions |
| 🔄 | **Automatic model migration** — ChromaDB detects and rebuilds incompatible embeddings |
| 📚 | See `ressources/*.md` for technical design and evaluation methodology |
| 🔍 | Adaptive research: automatically broadens search when context is thin |
| ✨ | Quality control: optional refinement loop to improve answers |

---

<p align="center">
  <sub>Built with ❤️ using LangGraph, Gemini, and open-source tools</sub>
</p>
