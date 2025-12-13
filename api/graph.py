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
from tools.retriever import build_vector_store, chunk_documents
from tools.reranker import rerank_chunks
from tools.search import SOURCE_GEMINI_GROUNDING, run_search_tasks
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
                    "task_queries": [t.text for t in plan],
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
        log.info(
            "search_start",
            extra={
                "tasks": len(plan),
                "task_queries": [t.text for t in plan],
                "query": query,
            },
        )
        try:
            if SMART_SEARCH_ENABLED and query:
                log.info("search_using_smart_search", extra={"query": query})
                results, warnings = smart_search(
                    query,
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

            return cast(GraphState, {
                "documents": documents,
                "warnings": base_warnings + new_warnings,
                "chunks": chunks,
                **preserve_grounded_state(state),
            })
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("fetch_failed", extra={"error": str(exc)})
            # Use immutable concatenation
            updated_errors = state.get("errors", []) + [f"fetch_failed: {exc}"]
            updated_warnings = state.get("warnings", []) + [
                "Fetch failed; proceeding with empty documents."
            ]
            return cast(GraphState, {
                "documents": [],
                "warnings": updated_warnings,
                "errors": updated_errors,
                "chunks": [],
                **preserve_grounded_state(state),
            })

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
            return cast(GraphState, {
                "retrieved": [],
                "retrieval_scores": [],
                "warnings": updated_warnings,
                **preserve_grounded_state(state),
            })
        try:
            query_text = state.get("query", "")
            log.info(
                "retrieve_start",
                extra={
                    "query": query_text,
                    "available_chunks": len(chunks),
                    "top_k": top_k,
                },
            )
            scored = store.query(query_text, top_k=top_k)
            retrieved = [sc.chunk for sc in scored]
            scores = [sc.score for sc in scored]
            if ENABLE_RERANKER and retrieved:
                reranked, rerank_scores = rerank_chunks(
                    query_text,
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
            return cast(GraphState, {
                "retrieved": retrieved,
                "retrieval_scores": scores,
                "warnings": updated_warnings,
                **preserve_grounded_state(state),
            })
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("retrieve_failed", extra={"error": str(exc)})
            # Use immutable concatenation
            updated_errors = state.get("errors", []) + [f"retrieve_failed: {exc}"]
            updated_warnings = state.get("warnings", []) + [
                "Retrieval failed; proceeding with empty context."
            ]
            return cast(GraphState, {
                "retrieved": [],
                "retrieval_scores": [],
                "warnings": updated_warnings,
                "errors": updated_errors,
                **preserve_grounded_state(state),
            })

    def adaptive_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.adaptive", run_id=state.get("run_id"))
        start = perf_counter()
        base_warnings = state.get("warnings", [])
        adaptive_iterations = state.get("adaptive_iterations", 0)
        retrieved = state.get("retrieved", [])
        retrieval_scores = state.get("retrieval_scores", [])
        query_text = state.get("query", "")

        # If we already have a grounded answer, skip adaptive expansion
        if state.get("grounded_answer") and state.get("grounded_sources"):
            log.info(
                "adaptive_skip_grounded",
                extra={
                    "adaptive_iterations": adaptive_iterations,
                    "retrieved": len(retrieved),
                },
            )
            return cast(GraphState, {
                "warnings": base_warnings,
                "adaptive_iterations": adaptive_iterations,
                **preserve_grounded_state(state),
            })

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
            query=query_text,
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
            return cast(GraphState, {
                "warnings": updated_warnings,
                "adaptive_iterations": adaptive_iterations,
                **preserve_grounded_state(state),
            })
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
        new_tasks = refine_plan(state.get("query", ""), state.get("plan", []))
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
        scored = store.query(query_text, top_k=top_k)
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
        return cast(GraphState, {
            "plan": plan,
            "search_results": results,
            "documents": documents,
            "chunks": chunks,
            "retrieved": retrieved,
            "retrieval_scores": retrieval_scores,
            "warnings": updated_warnings + cycle_warnings,
            "adaptive_iterations": adaptive_iterations,
            **preserve_grounded_state(state),
        })

    def generate_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.generate", run_id=state.get("run_id"))
        start = perf_counter()
        retrieved = state.get("retrieved", [])
        query = state.get("query", "")
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
            answer, citations = generate_answer(
                llm,
                query,
                retrieved,
                conversation_context=history_summary,
            )
            log.info(
                "generate_complete",
                extra={
                    "citations": len(citations),
                    "answer_length": len(answer),
                    "duration_ms": (perf_counter() - start) * 1000,
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
            },
        )
        try:
            verified = verify_answer(
                llm,
                draft,
                query,
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
                log.info(
                    "qc_complete",
                    extra={
                        "qc_passes": qc_passes,
                        "status": "accepted",
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

    logger.info(
        "run_research_start",
        extra={
            "query": query,
            "embed_model": embed_model,
            "top_k": top_k,
            "has_llm": llm is not None,
            "has_custom_store": vector_store is not None,
            "history_turns": len(history_window),
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
    initial_state = create_initial_state(query, run_id, history_window)
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
