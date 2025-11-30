"""Streamlit UI for Auto-Analyst."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # Ensure local imports work when run via streamlit
    sys.path.insert(0, str(ROOT))

from api.config import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL, TOP_K_RESULTS
from api.graph import run_research
from tools.models import load_llm
from tools.retriever import build_vector_store


st.set_page_config(page_title="Auto-Analyst", layout="wide")
st.title("Auto-Analyst (Free/Open-Source RAG)")
st.caption("Plan → Search → Fetch → Chunk → Embed → Retrieve → Generate → Verify")


if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "progress" not in st.session_state:
    st.session_state.progress = "Idle"


with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider(
        "Number of sources", min_value=2, max_value=10, value=TOP_K_RESULTS, step=1
    )
    llm_model = st.text_input(
        "LLM model",
        value=DEFAULT_LLM_MODEL,
        help="HuggingFace model id for an instruct-tuned LLM.",
    )
    embed_model = st.text_input(
        "Embedding model",
        value=DEFAULT_EMBED_MODEL,
        help="Sentence-transformers model id for embeddings.",
    )
    searx_host = st.text_input(
        "Optional SearxNG host (e.g., https://searx.example.com)"
    )


query = st.text_area("What do you want to research?", height=120)
run = st.button("Run Research", type="primary")


def render_sources(citations):
    for cite in citations:
        url = cite.get("url", "")
        title = cite.get("title", "Source")
        marker = cite.get("marker", "")
        st.markdown(f"{marker} [{title}]({url})" if url else f"{marker} {title}")


progress_placeholder = st.empty()

if run and query.strip():
    with st.spinner("Running autonomous research..."):
        st.session_state.progress = "Loading models"
        progress_placeholder.info(st.session_state.progress)
        llm = load_llm(model_name=llm_model)
        store = build_vector_store(model_name=embed_model)

        st.session_state.progress = "Executing pipeline"
        progress_placeholder.info(st.session_state.progress)
        result = run_research(
            query.strip(),
            llm=llm,
            vector_store=store,
            embed_model=embed_model,
            searx_host=searx_host or None,
            top_k=top_k,
        )
        st.session_state.last_result = result
        st.session_state.history.insert(
            0,
            {
                "query": query,
                "answer": result.verified_answer or result.draft_answer,
                "citations": result.citations,
                "plan": result.plan,
                "errors": result.errors,
                "warnings": getattr(result, "warnings", []),
            },
        )
    st.session_state.progress = "Idle"
    progress_placeholder.success("Done")

if st.session_state.last_result:
    result = st.session_state.last_result
    st.subheader("Plan")
    if result.plan:
        for idx, task in enumerate(result.plan, start=1):
            st.write(f"{idx}. {task.text} — {task.rationale}")
    else:
        st.write("No plan generated.")

    st.subheader("Answer")
    st.write(result.verified_answer or result.draft_answer)

    st.subheader("Sources")
    if result.citations:
        render_sources(result.citations)
    else:
        st.write("No sources available.")

    with st.expander("Retrieved Chunks"):
        for idx, chunk in enumerate(result.retrieved, start=1):
            meta = chunk.metadata or {}
            st.markdown(
                f"**[{idx}] {meta.get('title', 'Source')}** ({meta.get('url', '')})"
            )
            st.write(chunk.text[:1000])

    with st.expander("Debug"):
        st.write(f"Run ID: {result.run_id}")
        st.write(f"Adaptive iterations: {getattr(result, 'adaptive_iterations', 0)}")
        st.write(f"QC passes: {getattr(result, 'qc_passes', 0)}")
        if result.errors:
            st.error("Errors:\n" + "\n".join(result.errors))
        if getattr(result, "warnings", []):
            st.warning("Warnings:\n" + "\n".join(result.warnings))
        if getattr(result, "qc_notes", []):
            st.info("QC notes:\n" + "\n".join(result.qc_notes))
        st.write("Plan tasks:")
        for t in result.plan:
            st.write(f"- {t.text} ({t.rationale})")
        st.write("Search results:")
        for r in result.search_results[:10]:
            st.write(f"- {r.title} ({r.url})")

st.divider()
st.subheader("Session History")
if st.session_state.history:
    for entry in st.session_state.history:
        with st.expander(entry["query"]):
            st.write(entry["answer"])
            render_sources(entry["citations"])
else:
    st.write("No past runs yet.")
