"""LangGraph orchestration for the Auto-Analyst pipeline."""

from time import perf_counter
from typing import Any, List, Optional, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from api.cache_manager import CacheManager
from api.config import (
    ADAPTIVE_MAX_ITERS,
    CACHE_DB_PATH,
    CACHE_TTL_SECONDS,
    CONVERSATION_MEMORY_TURNS,
    CONVERSATION_SUMMARY_CHARS,
    DEFAULT_EMBED_MODEL,
    ENABLE_RERANKER,
    FETCH_CONCURRENCY,
    QC_MAX_PASSES,
    SMART_SEARCH_ENABLED,
    TOP_K_RESULTS,
)
from api.logging_setup import get_logger
from api.memory import (
    append_turn,
    resolve_followup_query,
    summarize_history,
    trim_history,
)
from api.state import Chunk, ConversationTurn, Document, GraphState, ResearchState
from api.state_builder import (
    build_research_state,
    create_initial_state,
    extract_grounded_answer,
    normalize_conversation_history,
)
from tools.adaptive_research import assess_context, refine_plan
from tools.fetcher import fetch_documents_parallel, fetch_url
from tools.generator import build_citations, generate_answer, verify_answer
from tools.planner import plan_query
from tools.quality_control import assess_answer, improve_answer
from tools.query_classifier import classify_query, get_query_type_description
from tools.retriever import build_vector_store, chunk_documents
from tools.reranker import rerank_chunks
from tools.search import run_search_tasks
from tools.smart_search import smart_search
from tools.chunker import TextChunker
from tools.models import load_llm
from vector_store.base import VectorStore


def preserve_grounded_state(state: GraphState) -> dict[str, Any]:
    """Extract grounded answer and sources from state for passthrough.

    Use this helper in pipeline nodes to preserve grounded state across
    node boundaries without repeating the boilerplate.

    Args:
        state: Current pipeline state.

    Returns:
        Dict with grounded_answer and grounded_sources keys.
    """
    return {
        "grounded_answer": state.get("grounded_answer", ""),
        "grounded_sources": state.get("grounded_sources", []),
    }


def resolve_effective_query(state: GraphState) -> str:
    """Resolve follow-up queries using conversation history."""
    query_text = state.get("query", "")
    history = state.get("conversation_history", []) or []
    return resolve_followup_query(query_text, history)


def build_workflow(
    llm=None,
    vector_store: Optional[VectorStore] = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = TOP_K_RESULTS,
    run_id: str | None = None,
):
    """Create a LangGraph app that runs the research pipeline."""
    chunker = TextChunker()
    llm = llm or load_llm()
    store = vector_store or build_vector_store(model_name=embed_model, run_id=run_id)

    def plan_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.plan", run_id=state.get("run_id"))
        start = perf_counter()
        query_text = state.get("query", "")
        history = state.get("conversation_history", []) or []
        resolved_query = resolve_followup_query(query_text, history)
        context_summary = summarize_history(
            history,
            max_chars=CONVERSATION_SUMMARY_CHARS,
        )
        log.info(
            "plan_start",
            extra={
                "query": query_text,
                "query_length": len(query_text),
                "history_turns": len(history),
                "resolved_query_differs": resolved_query != query_text,
            },
        )
        try:
            plan, time_sensitive = plan_query(
                resolved_query,
                conversation_context=context_summary,
            )
            log.info(
                "plan_complete",
                extra={
                    "tasks": len(plan),
                    "duration_ms": (perf_counter() - start) * 1000,
                    "query": query_text,
                    "time_sensitive": time_sensitive,
                    "history_context": bool(context_summary),
                },
            )
            return {"plan": plan, "time_sensitive": time_sensitive}
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("plan_failed", extra={"error": str(exc)})
            errors = state.get("errors", [])
            errors.append(f"plan_failed: {exc}")
            return {"plan": [], "time_sensitive": False, "errors": errors}

    def search_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.search", run_id=state.get("run_id"))
        start = perf_counter()
        plan = state.get("plan", [])
        query = state.get("query", "")
        effective_query = resolve_effective_query(state)
        log.info(
            "search_start",
            extra={
                "tasks": len(plan),
                "query": query,
                "resolved_query_differs": effective_query != query,
            },
        )
        try:
            if SMART_SEARCH_ENABLED and effective_query:
                log.info("search_using_smart_search", extra={"query": effective_query})
                results, warnings = smart_search(
                    effective_query,
                    max_results=5,
                    run_id=state.get("run_id"),
                )
            else:
                results, warnings = run_search_tasks(
                    plan,
                    max_results=5,
                    run_id=state.get("run_id"),
                )
            log.info(
                "search_complete",
                extra={
                    "results": len(results),
                    "duration_ms": (perf_counter() - start) * 1000,
                    "sources": list(set(r.source for r in results)),
                    "urls": [r.url for r in results[:5]],
                },
            )
            # Use immutable concatenation instead of extend
            updated_warnings = state.get("warnings", []) + warnings

            # Extract grounded answer using centralized utility
            grounded_answer, grounded_sources = extract_grounded_answer(
                results, run_id=state.get("run_id")
            )

            return {
                "search_results": results,
                "warnings": updated_warnings,
                "grounded_answer": grounded_answer,
                "grounded_sources": grounded_sources,
            }
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("search_failed", extra={"error": str(exc)})
            # Use immutable concatenation instead of append
            updated_errors = state.get("errors", []) + [f"search_failed: {exc}"]
            updated_warnings = state.get("warnings", []) + [
                "Search failed; proceeding with empty results."
            ]
            return {
                "search_results": [],
                "warnings": updated_warnings,
                "errors": updated_errors,
                "grounded_answer": "",
                "grounded_sources": [],
            }

    def fetch_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.fetch", run_id=state.get("run_id"))
        start = perf_counter()
        search_results = state.get("search_results", [])
        log.info(
            "fetch_start",
            extra={
                "search_results": len(search_results),
                "max_workers": FETCH_CONCURRENCY,
            },
        )
        try:
            base_warnings = state.get("warnings", [])
            new_warnings: List[str] = []
            documents: List[Document] = []
            fetch_failed = 0
            if search_results:
                documents, fetch_warnings = fetch_documents_parallel(
                    search_results,
                    max_workers=FETCH_CONCURRENCY,
                    run_id=state.get("run_id"),
                )
                new_warnings.extend(fetch_warnings)
                fetch_failed = len(search_results) - len(documents)
            if not documents:
                new_warnings.append("No documents fetched from search results.")

            chunk_start = perf_counter()
            chunks: List[Chunk] = []
            if documents:
                chunks = chunk_documents(documents, chunker, run_id=state.get("run_id"))
                store.upsert(chunks)

            log.info(
                "fetch_chunk_complete",
                extra={
                    "documents": len(documents),
                    "from_results": len(search_results),
                    "success": len(documents),
                    "failed": fetch_failed,
                    "chunks": len(chunks),
                    "chunk_duration_ms": (perf_counter() - chunk_start) * 1000,
                    "duration_ms": (perf_counter() - start) * 1000,
                },
            )
            if not chunks:
                new_warnings.append(
                    "No chunks available; downstream answers may be empty."
                )

            return cast(
                GraphState,
                {
                    "documents": documents,
                    "warnings": base_warnings + new_warnings,
                    "chunks": chunks,
                    **preserve_grounded_state(state),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("fetch_failed", extra={"error": str(exc)})
            # Use immutable concatenation
            updated_errors = state.get("errors", []) + [f"fetch_failed: {exc}"]
            updated_warnings = state.get("warnings", []) + [
                "Fetch failed; proceeding with empty documents."
            ]
            return cast(
                GraphState,
                {
                    "documents": [],
                    "warnings": updated_warnings,
                    "errors": updated_errors,
                    "chunks": [],
                    **preserve_grounded_state(state),
                },
            )

    def retrieve_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.retrieve", run_id=state.get("run_id"))
        start = perf_counter()
        chunks = state.get("chunks")
        if not chunks:
            log.warning("retrieve_no_chunks")
            # Use immutable concatenation
            updated_warnings = state.get("warnings", []) + [
                "No chunks to retrieve from; skipping retrieval."
            ]
            return cast(
                GraphState,
                {
                    "retrieved": [],
                    "retrieval_scores": [],
                    "warnings": updated_warnings,
                    **preserve_grounded_state(state),
                },
            )
        try:
            query_text = state.get("query", "")
            effective_query = resolve_effective_query(state)
            log.info(
                "retrieve_start",
                extra={
                    "query": query_text,
                    "resolved_query_differs": effective_query != query_text,
                    "available_chunks": len(chunks),
                    "top_k": top_k,
                },
            )
            scored = store.query(effective_query, top_k=top_k)
            retrieved = [sc.chunk for sc in scored]
            scores = [sc.score for sc in scored]
            if ENABLE_RERANKER and retrieved:
                reranked, rerank_scores = rerank_chunks(
                    effective_query,
                    retrieved,
                    top_k=top_k,
                    run_id=state.get("run_id"),
                )
                if reranked:
                    retrieved = reranked
                    # Only update scores if reranker actually produced scores
                    # (None means reranking was skipped/failed - preserve original vector scores)
                    if rerank_scores is not None:
                        scores = rerank_scores
                    log.debug(
                        "retrieve_reranked",
                        extra={
                            "original": len(scored),
                            "reranked": len(reranked),
                            "top_score": scores[0] if scores else None,
                            "used_rerank_scores": rerank_scores is not None,
                        },
                    )
            log.info(
                "retrieve_complete",
                extra={
                    "retrieved": len(retrieved),
                    "top_k": top_k,
                    "scores": scores,
                    "avg_score": sum(scores) / len(scores) if scores else 0,
                    "duration_ms": (perf_counter() - start) * 1000,
                },
            )
            # Use immutable concatenation
            updated_warnings = state.get("warnings", [])
            if not retrieved:
                updated_warnings = updated_warnings + [
                    "No retrieved context; answer may be unsupported."
                ]
            return cast(
                GraphState,
                {
                    "retrieved": retrieved,
                    "retrieval_scores": scores,
                    "warnings": updated_warnings,
                    **preserve_grounded_state(state),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("retrieve_failed", extra={"error": str(exc)})
            # Use immutable concatenation
            updated_errors = state.get("errors", []) + [f"retrieve_failed: {exc}"]
            updated_warnings = state.get("warnings", []) + [
                "Retrieval failed; proceeding with empty context."
            ]
            return cast(
                GraphState,
                {
                    "retrieved": [],
                    "retrieval_scores": [],
                    "warnings": updated_warnings,
                    "errors": updated_errors,
                    **preserve_grounded_state(state),
                },
            )

    def adaptive_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.adaptive", run_id=state.get("run_id"))
        start = perf_counter()
        base_warnings = state.get("warnings", [])
        adaptive_iterations = state.get("adaptive_iterations", 0)
        retrieved = state.get("retrieved", [])
        retrieval_scores = state.get("retrieval_scores", [])
        query_text = state.get("query", "")
        effective_query = resolve_effective_query(state)

        # If we already have a grounded answer, skip adaptive expansion
        if state.get("grounded_answer") and state.get("grounded_sources"):
            log.info(
                "adaptive_skip_grounded",
                extra={
                    "adaptive_iterations": adaptive_iterations,
                    "retrieved": len(retrieved),
                },
            )
            return cast(
                GraphState,
                {
                    "warnings": base_warnings,
                    "adaptive_iterations": adaptive_iterations,
                    **preserve_grounded_state(state),
                },
            )

        log.info(
            "adaptive_start",
            extra={
                "adaptive_iterations": adaptive_iterations,
                "retrieved_count": len(retrieved),
                "retrieval_scores": retrieval_scores,
                "max_iters": ADAPTIVE_MAX_ITERS,
            },
        )
        # Pass query and scores to assess_context for relevance-based assessment
        needs_more, assess_warnings = assess_context(
            retrieved,
            min_chunks=1,
            run_id=state.get("run_id"),
            query=effective_query,
            scores=retrieval_scores,
        )
        # Use immutable concatenation
        updated_warnings = base_warnings + assess_warnings
        if not needs_more or adaptive_iterations >= ADAPTIVE_MAX_ITERS:
            log.info(
                "adaptive_skip",
                extra={
                    "reason": (
                        "sufficient_context"
                        if not needs_more
                        else "max_iterations_reached"
                    ),
                    "adaptive_iterations": adaptive_iterations,
                },
            )
            return cast(
                GraphState,
                {
                    "warnings": updated_warnings,
                    "adaptive_iterations": adaptive_iterations,
                    **preserve_grounded_state(state),
                },
            )
        log.info(
            "adaptive_triggered",
            extra={
                "reason": "insufficient_or_irrelevant_context",
                "retrieved": len(retrieved),
                "avg_score": (
                    sum(retrieval_scores) / len(retrieval_scores)
                    if retrieval_scores
                    else 0
                ),
            },
        )
        new_tasks = refine_plan(effective_query, state.get("plan", []))
        plan = state.get("plan", []) + new_tasks
        log.debug(
            "adaptive_refined_plan",
            extra={"new_tasks": len(new_tasks), "total_tasks": len(plan)},
        )
        results, search_warns = run_search_tasks(
            plan,
            max_results=5,
            run_id=state.get("run_id"),
        )
        # Use immutable concatenation - collect all new warnings
        cycle_warnings = list(search_warns)
        documents = []
        for res in results:
            doc, warn = fetch_url(res, run_id=state.get("run_id"))
            if doc:
                documents.append(doc)
            if warn:
                cycle_warnings.append(f"{res.url}: {warn}")
        store.clear()
        chunks = chunk_documents(documents, chunker, run_id=state.get("run_id"))
        store.upsert(chunks)
        scored = store.query(effective_query, top_k=top_k)
        retrieved = [sc.chunk for sc in scored]
        retrieval_scores = [sc.score for sc in scored]
        adaptive_iterations += 1
        log.info(
            "adaptive_cycle_complete",
            extra={
                "adaptive_iterations": adaptive_iterations,
                "tasks": len(plan),
                "retrieved": len(retrieved),
                "retrieval_scores": retrieval_scores,
                "documents": len(documents),
                "chunks": len(chunks),
                "duration_ms": (perf_counter() - start) * 1000,
            },
        )
        return cast(
            GraphState,
            {
                "plan": plan,
                "search_results": results,
                "documents": documents,
                "chunks": chunks,
                "retrieved": retrieved,
                "retrieval_scores": retrieval_scores,
                "warnings": updated_warnings + cycle_warnings,
                "adaptive_iterations": adaptive_iterations,
                **preserve_grounded_state(state),
            },
        )

    def generate_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.generate", run_id=state.get("run_id"))
        start = perf_counter()
        retrieved = state.get("retrieved", [])
        query = state.get("query", "")
        effective_query = resolve_effective_query(state)
        history_summary = summarize_history(
            state.get("conversation_history", []) or [],
            max_chars=CONVERSATION_SUMMARY_CHARS,
        )

        # Check if we have a grounded answer from Gemini (fast path)
        grounded_answer = state.get("grounded_answer", "")
        grounded_sources = state.get("grounded_sources", [])

        if grounded_answer and grounded_sources:
            log.info(
                "generate_using_grounded_answer",
                extra={
                    "answer_length": len(grounded_answer),
                    "sources_count": len(grounded_sources),
                    "duration_ms": (perf_counter() - start) * 1000,
                },
            )
            remapped, citations = build_citations(grounded_sources, grounded_answer)
            return {
                "draft_answer": remapped,
                "citations": citations,
                "retrieved": grounded_sources,  # Use grounded sources for verification
            }

        log.info(
            "generate_start",
            extra={
                "query": query,
                "resolved_query_differs": effective_query != query,
                "retrieved_chunks": len(retrieved),
                "history_context": bool(history_summary),
            },
        )
        if not retrieved:
            log.warning("generate_no_context")
            warnings = state.get("warnings", [])
            warnings.append("No retrieved context; generating fallback answer.")
            return {
                "draft_answer": "No sufficient context retrieved to answer.",
                "citations": [],
                "warnings": warnings,
            }
        try:
            query_type = state.get("query_type", "factual")
            answer, citations = generate_answer(
                llm,
                effective_query,
                retrieved,
                conversation_context=history_summary,
                query_type=query_type,
            )
            log.info(
                "generate_complete",
                extra={
                    "citations": len(citations),
                    "answer_length": len(answer),
                    "duration_ms": (perf_counter() - start) * 1000,
                    "query_type": query_type,
                },
            )
            return {"draft_answer": answer, "citations": citations}
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("generate_failed", extra={"error": str(exc)})
            errors = state.get("errors", [])
            errors.append(f"generate_failed: {exc}")
            return {
                "draft_answer": "Generation failed; unable to produce answer.",
                "citations": [],
                "errors": errors,
            }

    def verify_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.verify", run_id=state.get("run_id"))
        start = perf_counter()
        draft = state.get("draft_answer", "")
        query = state.get("query", "")
        effective_query = resolve_effective_query(state)
        retrieved = state.get("retrieved", [])
        history_summary = summarize_history(
            state.get("conversation_history", []) or [],
            max_chars=CONVERSATION_SUMMARY_CHARS,
        )
        log.info(
            "verify_start",
            extra={
                "draft_length": len(draft),
                "retrieved_chunks": len(retrieved),
                "history_context": bool(history_summary),
                "resolved_query_differs": effective_query != query,
            },
        )
        try:
            verified = verify_answer(
                llm,
                draft,
                effective_query,
                retrieved,
                conversation_context=history_summary,
            )
            log.info(
                "verify_complete",
                extra={
                    "verified_length": len(verified),
                    "changed": verified != draft,
                    "duration_ms": (perf_counter() - start) * 1000,
                },
            )
            return {"verified_answer": verified}
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("verify_failed", extra={"error": str(exc)})
            # Use immutable concatenation
            updated_errors = state.get("errors", []) + [f"verify_failed: {exc}"]
            # Fall back to draft answer if verification fails
            return {"verified_answer": draft, "errors": updated_errors}

    def qc_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.qc", run_id=state.get("run_id"))
        start = perf_counter()
        qc_passes = state.get("qc_passes", 0)
        answer = state.get("verified_answer") or state.get("draft_answer", "")
        query = state.get("query", "")
        retrieved = state.get("retrieved", [])
        retrieval_scores = state.get("retrieval_scores", [])
        log.info(
            "qc_start",
            extra={
                "qc_passes": qc_passes,
                "answer_length": len(answer),
                "retrieved_chunks": len(retrieved),
                "retrieval_scores": retrieval_scores,
            },
        )
        try:
            assessment = assess_answer(
                query,
                answer,
                retrieved,
                run_id=state.get("run_id"),
                retrieval_scores=retrieval_scores,
            )
            # Use immutable concatenation
            updated_notes = state.get("qc_notes", []) + [
                f"pass {qc_passes}: issues={assessment['issues']}"
            ]
            if assessment["is_good_enough"] or qc_passes >= QC_MAX_PASSES:
                status = (
                    "accepted"
                    if assessment["is_good_enough"]
                    else "max_iterations_reached"
                )
                if qc_passes >= QC_MAX_PASSES and not assessment["is_good_enough"]:
                    log.warning(
                        "qc_max_iterations_reached",
                        extra={
                            "qc_passes": qc_passes,
                            "max_passes": QC_MAX_PASSES,
                            "issues": assessment["issues"],
                        },
                    )
                log.info(
                    "qc_complete",
                    extra={
                        "qc_passes": qc_passes,
                        "status": status,
                        "issues": assessment["issues"],
                        "duration_ms": (perf_counter() - start) * 1000,
                    },
                )
                return {"qc_passes": qc_passes, "qc_notes": updated_notes}
            log.info(
                "qc_needs_improvement",
                extra={"issues": assessment["issues"], "attempting_improvement": True},
            )
            improved = improve_answer(llm, query, answer, retrieved)
            qc_passes += 1
            log.info(
                "qc_refine",
                extra={
                    "qc_passes": qc_passes,
                    "issues": assessment["issues"],
                    "improved_length": len(improved),
                    "duration_ms": (perf_counter() - start) * 1000,
                },
            )
            return {
                "verified_answer": improved,
                "qc_passes": qc_passes,
                "qc_notes": updated_notes,
            }
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("qc_failed", extra={"error": str(exc)})
            # Use immutable concatenation
            updated_errors = state.get("errors", []) + [f"qc_failed: {exc}"]
            updated_notes = state.get("qc_notes", []) + [f"pass {qc_passes}: qc_failed"]
            return {
                "qc_passes": qc_passes,
                "qc_notes": updated_notes,
                "errors": updated_errors,
            }

    graph = StateGraph(GraphState)
    graph.add_node("plan", lambda state: plan_node(cast(GraphState, state)))
    graph.add_node("search", lambda state: search_node(cast(GraphState, state)))
    graph.add_node("fetch", lambda state: fetch_node(cast(GraphState, state)))
    graph.add_node("retrieve", lambda state: retrieve_node(cast(GraphState, state)))
    graph.add_node("generate", lambda state: generate_node(cast(GraphState, state)))
    graph.add_node("verify", lambda state: verify_node(cast(GraphState, state)))
    graph.add_node("adaptive", lambda state: adaptive_node(cast(GraphState, state)))
    graph.add_node("qc", lambda state: qc_node(cast(GraphState, state)))

    def should_continue_after_plan(state: GraphState) -> str:
        """Check if plan succeeded and decide whether to continue or short-circuit."""
        plan = state.get("plan", [])
        errors = state.get("errors", [])
        # Short-circuit if plan is empty AND we have critical errors
        if not plan and any("plan_failed" in e for e in errors):
            return "generate"  # Skip to generate with fallback
        return "search"

    def should_continue_after_search(state: GraphState) -> str:
        """Check if search produced results; short-circuit if all backends failed."""
        results = state.get("search_results", [])
        grounded_answer = state.get("grounded_answer", "")
        # If we have a grounded answer, continue normally
        if grounded_answer:
            return "fetch"
        # If no results and no grounded answer, we can still try fetch for any cached content
        return "fetch"

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        should_continue_after_plan,
        {"search": "search", "generate": "generate"},
    )
    graph.add_conditional_edges(
        "search",
        should_continue_after_search,
        {"fetch": "fetch"},
    )
    graph.add_edge("fetch", "retrieve")
    graph.add_edge("retrieve", "adaptive")
    graph.add_edge("adaptive", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "qc")
    graph.add_edge("qc", END)

    return graph.compile()


def run_research(
    query: str,
    llm=None,
    vector_store: Optional[VectorStore] = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = TOP_K_RESULTS,
    conversation_history: Optional[List[ConversationTurn]] = None,
) -> ResearchState:
    """Run the full pipeline and return a ResearchState object."""
    run_id = str(uuid4())
    logger = get_logger(__name__, run_id=run_id)
    overall_start = perf_counter()

    # Normalize and trim conversation history
    normalized_history = normalize_conversation_history(conversation_history)
    history_window = trim_history(
        normalized_history, max_turns=CONVERSATION_MEMORY_TURNS
    )

    # Check cache (include conversation history in cache key)
    cache_manager = CacheManager(CACHE_DB_PATH, CACHE_TTL_SECONDS, run_id=run_id)
    cached_result = cache_manager.get_cached_result(
        query, conversation_history=history_window
    )
    if cached_result:
        return cached_result

    # Classify the query to determine answer generation strategy
    query_type = classify_query(query, run_id=run_id)
    logger.info(
        "query_classified",
        extra={
            "query_type": query_type,
            "description": get_query_type_description(query_type),
        },
    )

    logger.info(
        "run_research_start",
        extra={
            "query": query,
            "embed_model": embed_model,
            "top_k": top_k,
            "has_llm": llm is not None,
            "has_custom_store": vector_store is not None,
            "history_turns": len(history_window),
            "query_type": query_type,
        },
    )

    # Build and invoke workflow
    workflow = build_workflow(
        llm=llm,
        vector_store=vector_store,
        embed_model=embed_model,
        top_k=top_k,
        run_id=run_id,
    )
    initial_state = create_initial_state(query, run_id, history_window, query_type)
    result = workflow.invoke(cast(GraphState, initial_state))

    # Build research state (handles grounded answer extraction internally)
    research_state = build_research_state(
        query=query,
        run_id=run_id,
        result=result,
        conversation_history=history_window,
    )

    # Update conversation history with the new turn
    answer_text = research_state.verified_answer or research_state.draft_answer
    updated_history = append_turn(
        history_window,
        query=query,
        answer=answer_text,
        citations=research_state.citations,
        max_turns=CONVERSATION_MEMORY_TURNS,
    )
    research_state.conversation_history = updated_history

    # Log completion
    duration_ms = (perf_counter() - overall_start) * 1000
    logger.info(
        "run_complete",
        extra={
            "duration_ms": duration_ms,
            "chunks": len(research_state.chunks),
            "retrieved": len(research_state.retrieved),
            "documents": len(research_state.documents),
            "search_results": len(research_state.search_results),
            "plan_tasks": len(research_state.plan),
            "citations": len(research_state.citations),
            "errors": len(research_state.errors),
            "warnings": len(research_state.warnings),
            "adaptive_iterations": research_state.adaptive_iterations,
            "qc_passes": research_state.qc_passes,
            "answer_length": len(answer_text),
            "history_turns": len(updated_history),
        },
    )

    # Cache result if appropriate (include conversation history in cache key)
    cache_manager.save_result(
        query, research_state, conversation_history=history_window
    )
    return research_state


def run_research_streaming(
    query: str,
    llm=None,
    vector_store: Optional[VectorStore] = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = TOP_K_RESULTS,
    conversation_history: Optional[List[ConversationTurn]] = None,
):
    """Run the research pipeline with streaming support.

    Yields events during pipeline execution:
        {"type": "step", "node": str, "status": "start"|"complete", "data": dict}
        {"type": "token", "text": str, "phase": "generate"|"verify"}
        {"type": "complete", "result": ResearchState}

    Args:
        query: The research question.
        llm: Optional LLM instance (created if not provided).
        vector_store: Optional vector store (created if not provided).
        embed_model: Embedding model name.
        top_k: Number of chunks to retrieve.
        conversation_history: Previous conversation turns.

    Yields:
        Dict with event type and associated data.
    """
    from tools.generator import (
        build_citations,
        generate_answer_stream,
        verify_answer_stream,
    )

    run_id = str(uuid4())
    logger = get_logger(__name__, run_id=run_id)
    overall_start = perf_counter()

    # Normalize and trim conversation history
    normalized_history = normalize_conversation_history(conversation_history)
    history_window = trim_history(
        normalized_history, max_turns=CONVERSATION_MEMORY_TURNS
    )

    # Check cache first
    cache_manager = CacheManager(CACHE_DB_PATH, CACHE_TTL_SECONDS, run_id=run_id)
    cached_result = cache_manager.get_cached_result(
        query, conversation_history=history_window
    )
    if cached_result:
        yield {"type": "complete", "result": cached_result, "from_cache": True}
        return

    # Classify the query
    query_type = classify_query(query, run_id=run_id)
    logger.info(
        "streaming_query_classified",
        extra={"query_type": query_type},
    )

    # Initialize LLM and store
    llm = llm or load_llm()
    store = vector_store or build_vector_store(model_name=embed_model, run_id=run_id)

    # Build the workflow
    workflow = build_workflow(
        llm=llm,
        vector_store=store,
        embed_model=embed_model,
        top_k=top_k,
        run_id=run_id,
    )
    initial_state = create_initial_state(query, run_id, history_window, query_type)

    # Node-to-step mapping for progress reporting
    node_info = {
        "plan": {"step": 1, "label": "Planning searches"},
        "search": {"step": 2, "label": "Searching sources"},
        "fetch": {"step": 3, "label": "Fetching documents"},
        "retrieve": {"step": 4, "label": "Retrieving context"},
        "adaptive": {"step": 5, "label": "Assessing context"},
        "generate": {"step": 6, "label": "Generating answer"},
        "verify": {"step": 7, "label": "Verifying answer"},
        "qc": {"step": 8, "label": "Quality check"},
    }

    # Track state as we stream through the pipeline
    accumulated_state = dict(initial_state)
    current_node = None

    # Stream through LangGraph nodes using sync iteration
    # Note: We use the sync stream() API and emit events
    for chunk in workflow.stream(
        cast(GraphState, initial_state), stream_mode="updates"
    ):
        # chunk is {node_name: {state_updates}}
        for node_name, state_update in chunk.items():
            if node_name == "__start__" or node_name == "__end__":
                continue

            info = node_info.get(node_name, {"step": 0, "label": node_name})

            # Emit step start
            yield {
                "type": "step",
                "node": node_name,
                "status": "start",
                "step": info["step"],
                "label": info["label"],
            }

            # Accumulate state
            accumulated_state.update(state_update)

            # Emit progress info based on node
            data = {}
            if node_name == "plan":
                plan = state_update.get("plan", [])
                data["task_count"] = len(plan)
                data["tasks"] = [t.text for t in plan[:3]] if plan else []
            elif node_name == "search":
                results = state_update.get("search_results", [])
                data["result_count"] = len(results)
                data["sources"] = [r.title for r in results[:5]] if results else []
            elif node_name == "fetch":
                docs = state_update.get("documents", [])
                chunks = state_update.get("chunks", [])
                data["document_count"] = len(docs)
                data["chunk_count"] = len(chunks)
            elif node_name == "retrieve":
                retrieved = state_update.get("retrieved", [])
                data["retrieved_count"] = len(retrieved)

            # Emit step complete
            yield {
                "type": "step",
                "node": node_name,
                "status": "complete",
                "step": info["step"],
                "label": info["label"],
                "data": data,
            }

    # Now do streaming generation and verification
    retrieved = accumulated_state.get("retrieved", [])
    grounded_answer = accumulated_state.get("grounded_answer", "")
    grounded_sources = accumulated_state.get("grounded_sources", [])

    # If we have a grounded answer, use it directly
    if grounded_answer and grounded_sources:
        logger.info("streaming_using_grounded_answer")
        remapped, citations = build_citations(grounded_sources, grounded_answer)
        yield {
            "type": "step",
            "node": "generate",
            "status": "start",
            "step": 6,
            "label": "Using grounded answer",
        }
        yield {"type": "token", "text": remapped, "phase": "generate"}
        yield {
            "type": "step",
            "node": "generate",
            "status": "complete",
            "step": 6,
            "label": "Generating answer",
            "data": {"answer_length": len(remapped)},
        }
        draft_answer = remapped
        final_citations = citations
        # Skip verify streaming for grounded answers
        verified_answer = remapped
    else:
        # Stream generate
        yield {
            "type": "step",
            "node": "generate_stream",
            "status": "start",
            "step": 6,
            "label": "Generating answer",
        }

        history_summary = summarize_history(
            history_window, max_chars=CONVERSATION_SUMMARY_CHARS
        )
        effective_query = resolve_followup_query(query, history_window)

        draft_answer = ""
        final_citations = []
        for partial, is_complete, citations in generate_answer_stream(
            llm, effective_query, retrieved, history_summary, query_type
        ):
            if is_complete:
                draft_answer = partial
                final_citations = citations
                # Also emit final token so UI receives the complete answer
                yield {"type": "token", "text": partial, "phase": "generate"}
            else:
                yield {"type": "token", "text": partial, "phase": "generate"}

        yield {
            "type": "step",
            "node": "generate_stream",
            "status": "complete",
            "step": 6,
            "label": "Generating answer",
            "data": {"answer_length": len(draft_answer)},
        }

        # Stream verify
        yield {
            "type": "step",
            "node": "verify_stream",
            "status": "start",
            "step": 7,
            "label": "Verifying answer",
        }

        verified_answer = ""
        for partial, is_complete in verify_answer_stream(
            llm, draft_answer, effective_query, retrieved, history_summary
        ):
            if is_complete:
                verified_answer = partial
                # Also emit final token so UI receives the complete verified answer
                yield {"type": "token", "text": partial, "phase": "verify"}
            else:
                yield {"type": "token", "text": partial, "phase": "verify"}

        yield {
            "type": "step",
            "node": "verify_stream",
            "status": "complete",
            "step": 7,
            "label": "Verifying answer",
            "data": {"verified_length": len(verified_answer)},
        }

    # Build final research state
    research_state = ResearchState(
        query=query,
        run_id=run_id,
        plan=accumulated_state.get("plan", []),
        search_results=accumulated_state.get("search_results", []),
        documents=accumulated_state.get("documents", []),
        chunks=accumulated_state.get("chunks", []),
        retrieved=grounded_sources if grounded_sources else retrieved,
        retrieval_scores=accumulated_state.get("retrieval_scores", []),
        draft_answer=draft_answer,
        verified_answer=verified_answer,
        citations=final_citations,
        errors=accumulated_state.get("errors", []),
        warnings=accumulated_state.get("warnings", []),
        adaptive_iterations=accumulated_state.get("adaptive_iterations", 0),
        qc_passes=accumulated_state.get("qc_passes", 0),
        qc_notes=accumulated_state.get("qc_notes", []),
        time_sensitive=accumulated_state.get("time_sensitive", False),
        grounded_answer=grounded_answer,
        grounded_sources=grounded_sources,
        query_type=query_type,
    )

    # Update conversation history
    answer_text = verified_answer or draft_answer
    updated_history = append_turn(
        history_window,
        query=query,
        answer=answer_text,
        citations=final_citations,
        max_turns=CONVERSATION_MEMORY_TURNS,
    )
    research_state.conversation_history = updated_history

    # Log completion
    duration_ms = (perf_counter() - overall_start) * 1000
    logger.info(
        "streaming_run_complete",
        extra={
            "duration_ms": duration_ms,
            "citations": len(final_citations),
            "answer_length": len(answer_text),
        },
    )

    # Cache result
    cache_manager.save_result(
        query, research_state, conversation_history=history_window
    )

    yield {"type": "complete", "result": research_state}
