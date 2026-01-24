"""Streamlit UI for Auto-Analyst.

Note: This module should be run from the project root directory using:
    streamlit run ui/app.py
"""

import html
import sys
import time
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

# =============================================================================
# PAGE CONFIG & CUSTOM STYLING
# =============================================================================
st.set_page_config(
    page_title="Auto-Analyst", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS for better section differentiation
st.markdown(
    """
<style>
    /* Main section cards */
    .section-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .section-card-blue {
        border-left-color: #2196F3;
    }
    
    .section-card-purple {
        border-left-color: #9C27B0;
    }
    
    .section-card-orange {
        border-left-color: #FF9800;
    }
    
    .section-card-red {
        border-left-color: #f44336;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Plan task styling */
    .plan-task {
        background: rgba(76, 175, 80, 0.1);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    /* Source card */
    .source-card {
        background: rgba(33, 150, 243, 0.1);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(33, 150, 243, 0.3);
    }
    
    /* Answer container */
    .answer-container {
        background: rgba(156, 39, 176, 0.05);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid rgba(156, 39, 176, 0.2);
        line-height: 1.7;
    }
    
    /* Query input area enhancement */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #4CAF50 !important;
    }
    
    /* Button styling */
    .stButton>button[kind="primary"] {
        background: linear-gradient(90deg, #4CAF50, #45a049) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .badge-success { background: rgba(76, 175, 80, 0.2); color: #4CAF50; }
    .badge-info { background: rgba(33, 150, 243, 0.2); color: #2196F3; }
    .badge-warning { background: rgba(255, 152, 0, 0.2); color: #FF9800; }
    
    /* Quick stats bar */
    .stats-bar {
        display: flex;
        gap: 1.5rem;
        padding: 0.75rem 1rem;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .stat-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
    }
    .stat-value { font-weight: 600; }
    
    /* Progress steps */
    .progress-steps {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        padding: 1rem;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .step { 
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        background: rgba(255,255,255,0.1);
    }
    .step-done { background: rgba(76, 175, 80, 0.3); color: #4CAF50; }
    .step-active { background: rgba(33, 150, 243, 0.3); color: #2196F3; animation: pulse 1s infinite; }
    .step-pending { opacity: 0.5; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    
    /* Small query mode pill */
    .query-mode-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("# 🔬 Auto-Analyst")
st.markdown("##### Autonomous Research Assistant powered by RAG")


if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "progress" not in st.session_state:
    st.session_state.progress = "Idle"
if "progress_step" not in st.session_state:
    st.session_state.progress_step = 0
if "execution_time" not in st.session_state:
    st.session_state.execution_time = 0
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


query = st.text_area(
    "What do you want to research?",
    height=100,
    placeholder="Enter your research question here...",
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    run = st.button("🔍 Run Research", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.last_result = None
        st.rerun()


def render_sources(citations):
    """Render citations as styled cards."""
    for cite in citations:
        url = cite.get("url", "")
        title = cite.get("title", "Source")
        marker = cite.get("marker", "")
        if url:
            st.markdown(
                f"""<div class="source-card">
                {marker} <a href="{url}" target="_blank">{title}</a>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="source-card">{marker} {title}</div>""",
                unsafe_allow_html=True,
            )


progress_placeholder = st.empty()


def render_progress_steps(current_step: int):
    """Render step-by-step progress indicator."""
    steps = ["📥 Load", "📋 Plan", "🔍 Search", "📄 Fetch", "🧠 Generate", "✅ Verify"]
    step_html = '<div class="progress-steps">'
    for i, step in enumerate(steps):
        if i < current_step:
            step_html += f'<span class="step step-done">✓ {step}</span>'
        elif i == current_step:
            step_html += f'<span class="step step-active">⏳ {step}</span>'
        else:
            step_html += f'<span class="step step-pending">{step}</span>'
        if i < len(steps) - 1:
            step_html += '<span style="color: #666;">→</span>'
    step_html += '</div>'
    return step_html


if run and query.strip():
    start_time = time.time()
    
    # Step 0: Loading models
    st.session_state.progress_step = 0
    progress_placeholder.markdown(render_progress_steps(0), unsafe_allow_html=True)
    llm = load_llm(model_name=llm_model)
    run_id = str(uuid4())
    store = build_vector_store(model_name=embed_model, run_id=run_id)

    history_turns = [
        ConversationTurn.from_dict(turn)
        for turn in st.session_state.conversation_snapshot
    ]
    history_turns = trim_history(history_turns, CONVERSATION_MEMORY_TURNS)

    # Step 1-5: Pipeline execution (shown as single step since run_research is atomic)
    st.session_state.progress_step = 1
    progress_placeholder.markdown(render_progress_steps(1), unsafe_allow_html=True)
    
    result = run_research(
        query.strip(),
        llm=llm,
        vector_store=store,
        embed_model=embed_model,
        top_k=top_k,
        conversation_history=history_turns,
    )
    
    # Record execution time
    st.session_state.execution_time = time.time() - start_time
    
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
            "execution_time": st.session_state.execution_time,
        },
    )
    
    # Show completion
    st.session_state.progress_step = 5
    progress_placeholder.markdown(render_progress_steps(6), unsafe_allow_html=True)

if st.session_state.last_result:
    result = st.session_state.last_result
    
    st.markdown("---")

    # ==========================================================================
    # QUICK STATS BAR
    # ==========================================================================
    exec_time = st.session_state.get("execution_time", 0)
    num_sources = len(result.citations)
    num_chunks = len(result.retrieved)
    iterations = getattr(result, "adaptive_iterations", 0)
    qc_passes = getattr(result, "qc_passes", 0)
    is_verified = bool(result.verified_answer)
    
    # Query mode as small pill
    query_type = getattr(result, "query_type", "factual")
    mode_icons = {"recommendation": "🎯", "creative": "✨", "factual": "📚"}
    mode_colors = {"recommendation": "#FF9800", "creative": "#9C27B0", "factual": "#2196F3"}
    icon = mode_icons.get(query_type, "📚")
    color = mode_colors.get(query_type, "#2196F3")
    
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-item"><span class="query-mode-pill" style="background: {color}33; color: {color};">{icon} {query_type.title()}</span></div>
        <div class="stat-item">⏱️ <span class="stat-value">{exec_time:.1f}s</span></div>
        <div class="stat-item">📚 <span class="stat-value">{num_sources}</span> sources</div>
        <div class="stat-item">📄 <span class="stat-value">{num_chunks}</span> chunks</div>
        <div class="stat-item">🔄 <span class="stat-value">{iterations}</span> iterations</div>
        <div class="stat-item">{'✅' if is_verified else '⚠️'} <span class="stat-value">{'Verified' if is_verified else 'Draft'}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # ANSWER SECTION (Primary - show first)
    # ==========================================================================
    answer_text = result.verified_answer or result.draft_answer
    
    # Answer header with copy button
    st.markdown("### 💡 Answer")
    
    # Render the answer
    st.markdown(answer_text)
    
    # Copy button using st.code's built-in copy functionality
    with st.expander("📋 Copy Answer", expanded=False):
        st.code(answer_text, language="markdown")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================================
    # SOURCES SECTION
    # ==========================================================================
    st.markdown("### 📚 Sources")
    if result.citations:
        cols = st.columns(2)
        for i, cite in enumerate(result.citations):
            with cols[i % 2]:
                url = cite.get("url", "")
                title = cite.get("title", "Source")
                marker = cite.get("marker", "")
                if url:
                    st.markdown(f"""<div class="source-card">
                        <strong>{marker}</strong> <a href="{url}" target="_blank">{title}</a>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="source-card"><strong>{marker}</strong> {title}</div>""", unsafe_allow_html=True)
    else:
        st.info("No sources available.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================================
    # PLAN SECTION (Collapsed by default)
    # ==========================================================================
    with st.expander(f"📋 Research Plan ({len(result.plan)} tasks)", expanded=False):
        if result.plan:
            for idx, task in enumerate(result.plan, start=1):
                st.markdown(f"""<div class="plan-task">
                    <strong>{idx}.</strong> {task.text}
                    <br><small style="color: #888;">— {task.rationale}</small>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No plan generated.")

    # ==========================================================================
    # DEBUG SECTION (Contains chunks, errors, search results)
    # ==========================================================================
    with st.expander("🔧 Debug Information", expanded=False):
        # Run info
        st.markdown(f"**Run ID:** `{result.run_id}`")
        
        # Errors & Warnings first (if any)
        if result.errors:
            st.error("**Errors:**\n" + "\n".join(f"• {e}" for e in result.errors))
        if getattr(result, "warnings", []):
            st.warning("**Warnings:**\n" + "\n".join(f"• {w}" for w in result.warnings))
        if getattr(result, "qc_notes", []):
            st.info("**QC Notes:**\n" + "\n".join(f"• {n}" for n in result.qc_notes))
        
        # Search results
        st.markdown("**Search Results:**")
        for r in result.search_results[:10]:
            st.markdown(f"- [{r.title}]({r.url})")
        
        # Retrieved chunks (merged here)
        st.markdown(f"**Retrieved Chunks ({len(result.retrieved)}):**")
        for idx, chunk in enumerate(result.retrieved, start=1):
            meta = chunk.metadata or {}
            title = html.escape(meta.get('title', 'Source'))
            url = meta.get('url', '')
            # Escape chunk text to prevent HTML injection
            chunk_preview = html.escape(chunk.text[:400])
            if len(chunk.text) > 400:
                chunk_preview += "..."
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); padding: 0.75rem; 
                            border-radius: 8px; margin-bottom: 0.5rem; font-size: 0.85rem;
                            border: 1px solid rgba(255,255,255,0.1);">
                    <strong>[{idx}] {title}</strong>
                    {" — <a href='" + url + "' target='_blank'>link</a>" if url else ""}
                    <br><span style="color: #888;">{chunk_preview}</span>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# SESSION HISTORY
# =============================================================================
if st.session_state.history:
    st.markdown(f"### 🕐 Session History ({len(st.session_state.history)} queries)")
    for entry in st.session_state.history:
        exec_t = entry.get('execution_time', 0)
        time_str = f" • {exec_t:.1f}s" if exec_t else ""
        with st.expander(f"📝 {entry['query'][:70]}{'...' if len(entry['query']) > 70 else ''}{time_str}"):
            st.markdown(f"""<div class="answer-container">{entry["answer"]}</div>""", unsafe_allow_html=True)
            if entry["citations"]:
                st.markdown("<br>**Sources:**", unsafe_allow_html=True)
                for cite in entry["citations"]:
                    url = cite.get("url", "")
                    title = cite.get("title", "Source")
                    marker = cite.get("marker", "")
                    if url:
                        st.markdown(f"{marker} [{title}]({url})")
                    else:
                        st.markdown(f"{marker} {title}")
