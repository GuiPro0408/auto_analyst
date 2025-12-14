"""Streamlit UI for Auto-Analyst.

Note: This module should be run from the project root directory using:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uuid import uuid4

import streamlit as st

from api.config import (
    CONVERSATION_MEMORY_TURNS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    TOP_K_RESULTS,
)
from api.graph import run_research
from api.key_rotator import get_default_rotator, reset_default_rotator
from api.memory import trim_history
from api.state import ConversationTurn
from tools.models import load_llm
from tools.query_classifier import get_query_type_description
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
if "conversation_snapshot" not in st.session_state:
    st.session_state.conversation_snapshot = []


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

    st.markdown("---")
    st.subheader("API Key Status")
    rotator = get_default_rotator()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Keys", rotator.available_keys)
    with col2:
        st.metric("Total Keys", rotator.total_keys)

    if rotator.is_exhausted:
        st.warning("⚠️ All Gemini API keys are rate-limited. Using fallback search.")
        if st.button("Reset Rate Limits"):
            rotator.reset()
            st.success("Rate limit tracking reset!")
            st.rerun()
    elif rotator.available_keys < rotator.total_keys:
        st.info(f"ℹ️ {rotator.total_keys - rotator.available_keys} key(s) rate-limited")
        if st.button("Reset Rate Limits"):
            rotator.reset()
            st.success("Rate limit tracking reset!")
            st.rerun()

    st.markdown("---")
    st.caption(
        f"Conversation memory keeps the last {CONVERSATION_MEMORY_TURNS} answers so follow-up questions stay grounded."
    )
    if st.button("Reset conversation memory"):
        st.session_state.conversation_snapshot = []
        st.success("Conversation memory cleared")

    with st.expander("Current memory", expanded=False):
        if st.session_state.conversation_snapshot:
            for turn in st.session_state.conversation_snapshot:
                st.markdown(f"**You:** {turn.get('query', '')}")
                answer_preview = turn.get("answer", "")[:200]
                if len(turn.get("answer", "")) > 200:
                    answer_preview += "..."
                st.caption(f"Assistant: {answer_preview}")
        else:
            st.caption("No prior turns stored yet.")


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
        run_id = str(uuid4())
        store = build_vector_store(model_name=embed_model, run_id=run_id)

        history_turns = [
            ConversationTurn.from_dict(turn)
            for turn in st.session_state.conversation_snapshot
        ]
        history_turns = trim_history(history_turns, CONVERSATION_MEMORY_TURNS)

        st.session_state.progress = "Executing pipeline"
        progress_placeholder.info(st.session_state.progress)
        result = run_research(
            query.strip(),
            llm=llm,
            vector_store=store,
            embed_model=embed_model,
            top_k=top_k,
            conversation_history=history_turns,
        )
        st.session_state.last_result = result
        st.session_state.conversation_snapshot = [
            turn.to_dict() for turn in result.conversation_history
        ]
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

    # Show query mode
    query_type = getattr(result, "query_type", "factual")
    mode_desc = get_query_type_description(query_type)
    if query_type == "recommendation":
        st.info(f"🎯 **{mode_desc}**")
    elif query_type == "creative":
        st.info(f"✨ **{mode_desc}**")
    else:
        st.info(f"📚 **{mode_desc}**")

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
        st.write(f"Query type: {getattr(result, 'query_type', 'factual')}")
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
