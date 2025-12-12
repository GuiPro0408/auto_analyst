# Auto-Analyst

An autonomous research assistant powered by a LangGraph RAG pipeline. Plans queries, searches free sources, chunks and embeds content, retrieves context, generates cited answers, and verifies claims—all using free/open-source components.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/<your-username>/auto-analyst.git && cd auto-analyst
python -m venv venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

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
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PLAN                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LLM decomposes query into SearchQuery tasks                         │   │
│  │ Example: ["effects of X on Y", "X statistics 2024"]                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/planner.py → List[SearchQuery]                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SEARCH                                                                     │
│  ┌──────────────┐  ┌────────────────┐                                      │
│  │    Tavily    │  │ Gemini Ground  │                                      │
│  │   (API)      │  │ (Google search)│                                      │
│  └──────────────┘  └────────────────┘                                      │
│  tools/search.py → List[SearchResult] (url, title, snippet)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FETCH                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ robots.txt check → Download HTML/PDF → Parse content                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/fetcher.py + tools/parser.py → List[Document]                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CHUNK                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Token-aware splitting (tiktoken) with overlap                       │   │
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
│  RETRIEVE                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Cosine similarity search → Top-K chunks (default K=6)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/retriever.py → List[ScoredChunk]                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GENERATE                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LLM (Phi-3/Mistral/Llama) + Context → Answer with [n] citations     │   │
│  │                                                                      │   │
│  │ Prompt: "Using only the context, answer with inline citations..."   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/generator.py:generate_answer() → (answer, citations)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VERIFY                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LLM reviews draft → Removes unsupported claims → Final answer       │   │
│  │                                                                      │   │
│  │ Prompt: "Remove statements not supported by context..."             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  tools/generator.py:verify_answer() → verified_answer                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESEARCH STATE                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                    │   │
│  │   query, plan, search_results, documents, chunks,                   │   │
│  │   retrieved, draft_answer, verified_answer, citations, errors       │   │
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
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ui/app.py                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage Summary

| Stage        | Description                                     |
| ------------ | ----------------------------------------------- |
| **Plan**     | Decomposes question into targeted search tasks  |
| **Search**   | Queries Tavily and Gemini Grounding             |
| **Fetch**    | Downloads pages/PDFs respecting robots.txt      |
| **Chunk**    | Token-aware splitting with metadata             |
| **Retrieve** | Vector similarity search (ChromaDB/FAISS)       |
| **Generate** | LLM produces answer with `[n]` citations        |
| **Verify**   | Prunes unsupported claims                       |

## Configuration

| Variable                         | Default                            | Purpose                       |
| -------------------------------- | ---------------------------------- | ----------------------------- |
| `AUTO_ANALYST_LLM`               | `gemini-2.0-flash`                 | LLM model identifier          |
| `AUTO_ANALYST_EMBED`             | `all-MiniLM-L6-v2`                 | Embedding model               |
| `AUTO_ANALYST_VECTOR_STORE`      | `chroma`                           | `chroma` or `faiss`           |
| `AUTO_ANALYST_TOP_K`             | `6`                                | Retrieved chunks              |
| `AUTO_ANALYST_LOG_LEVEL`         | `INFO`                             | `DEBUG`/`INFO`/`WARNING`      |
| `AUTO_ANALYST_LOG_FORMAT`        | `plain`                            | `plain` or `json`             |
| `AUTO_ANALYST_SMART_SEARCH`      | `true`                             | Enable LLM-assisted query analysis and validation |
| `AUTO_ANALYST_VALIDATE_RESULTS`  | `true`                             | Use LLM to filter irrelevant search hits |
| `AUTO_ANALYST_FETCH_RETRIES`     | `2`                                | Retry attempts for fetch      |
| `AUTO_ANALYST_FETCH_BACKOFF`     | `1.0` seconds                      | Backoff factor for fetch      |
| `TAVILY_API_KEY`                 | ``                                 | API key for Tavily search backend |
| `AUTO_ANALYST_LOG_FILE`          | `auto_analyst.log`                 | Log file path                 |
| `AUTO_ANALYST_ADAPTIVE_MAX_ITERS`| `2`                                | Adaptive search cycles        |
| `AUTO_ANALYST_QC_MAX_PASSES`     | `1`                                | Quality control passes        |

### API keys & secrets

Some backends require API credentials that **must be supplied through environment variables** (never hard-code them):

- `GOOGLE_API_KEY` — required for the Gemini backend (`AUTO_ANALYST_LLM_BACKEND=gemini`).
- `HUGGINGFACE_API_TOKEN` — required when using the Hugging Face Inference backend.
- `TAVILY_API_KEY` — recommended for the Tavily search backend (more reliable web results).

Create or edit your local `.env` file and add placeholders:

```
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
HUGGINGFACE_API_TOKEN=YOUR_HF_API_TOKEN
AUTO_ANALYST_LLM_BACKEND=gemini
AUTO_ANALYST_SMART_SEARCH=true
AUTO_ANALYST_VALIDATE_RESULTS=true
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
```

Keep the `.env` file out of source control when filling in real credentials, and rotate any keys that may have been exposed previously.

## Commands

```bash
source .venv/bin/activate          # Activate virtualenv
streamlit run ui/app.py                    # Run UI
pytest                                      # Run tests
python evaluation/run_evaluation.py --dataset data/sample_eval.json  # Evaluate
docker build -t auto-analyst . && docker run -p 8501:8501 auto-analyst  # Docker
```

## Project Structure

```
api/          → LangGraph orchestration, state definitions, config
tools/        → Planner, search, fetch, parse, chunk, generate, verify
vector_store/ → ChromaDB and FAISS backends
ui/           → Streamlit interface
evaluation/   → RAG metrics (relevance, correctness, hallucination)
tests/        → Unit and integration tests
```

## Prerequisites

- Python 3.11+
- ~8GB disk for models (Phi-3/Mistral + embeddings)

## Notes

- **No paid APIs**—all search, models, and embeddings are free
- **robots.txt compliance**—fetcher respects site restrictions
- See `ressources/*.md` for technical design and evaluation methodology
- Adaptive research: automatically broadens search when context is thin (capped by `AUTO_ANALYST_ADAPTIVE_MAX_ITERS`).
- Quality control: optional refinement loop to improve answers up to `AUTO_ANALYST_QC_MAX_PASSES`.
