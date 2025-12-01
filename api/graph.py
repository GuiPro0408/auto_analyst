"""LangGraph orchestration for the Auto-Analyst pipeline."""

from time import perf_counter
from typing import Any, Optional, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from api.config import (
    ADAPTIVE_MAX_ITERS,
    DEFAULT_EMBED_MODEL,
    QC_MAX_PASSES,
    TOP_K_RESULTS,
)
from api.logging_setup import get_logger
from api.state import GraphState, ResearchState
from tools.adaptive_research import assess_context, refine_plan
from tools.fetcher import fetch_url
from tools.generator import generate_answer, verify_answer
from tools.planner import plan_query
from tools.quality_control import assess_answer, improve_answer
from tools.retriever import build_vector_store, chunk_documents
from tools.search import run_search_tasks
from tools.chunker import TextChunker
from tools.models import load_llm
from vector_store.base import VectorStore


def build_workflow(
    llm=None,
    vector_store: Optional[VectorStore] = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    searx_host: Optional[str] = None,
    top_k: int = TOP_K_RESULTS,
    run_id: str | None = None,
):
    """Create a LangGraph app that runs the research pipeline."""
    chunker = TextChunker()
    llm = llm or load_llm()
    store = vector_store or build_vector_store(model_name=embed_model)

    def plan_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.plan", run_id=state.get("run_id"))
        start = perf_counter()
        query_text = state.get("query", "")
        log.info(
            "plan_start",
            extra={"query": query_text, "query_length": len(query_text)},
        )
        try:
            # Default to heuristic planner for stability
            # plan_query now returns (tasks, time_sensitive) tuple
            plan, time_sensitive = plan_query(query_text, llm=None)
            log.info(
                "plan_complete",
                extra={
                    "tasks": len(plan),
                    "duration_ms": (perf_counter() - start) * 1000,
                    "query": query_text,
                    "task_queries": [t.text for t in plan],
                    "time_sensitive": time_sensitive,
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
        log.info(
            "search_start",
            extra={"tasks": len(plan), "task_queries": [t.text for t in plan]},
        )
        results, warnings = run_search_tasks(
            plan,
            max_results=5,
            searx_host=searx_host,
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
        state_warnings = state.get("warnings", [])
        state_warnings.extend(warnings)
        return {"search_results": results, "warnings": state_warnings}

    def fetch_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.fetch", run_id=state.get("run_id"))
        start = perf_counter()
        search_results = state.get("search_results", [])
        log.info(
            "fetch_start",
            extra={"search_results": len(search_results)},
        )
        documents = []
        warnings = state.get("warnings", [])
        fetch_success = 0
        fetch_failed = 0
        for idx, res in enumerate(search_results):
            log.debug(
                "fetch_url_attempt",
                extra={"index": idx, "url": res.url, "source": res.source},
            )
            doc, warn = fetch_url(res, run_id=state.get("run_id"))
            if doc:
                documents.append(doc)
                fetch_success += 1
            else:
                fetch_failed += 1
            if warn:
                warnings.append(f"{res.url}: {warn}")
        if not documents:
            warnings.append("No documents fetched from search results.")
        log.info(
            "fetch_complete",
            extra={
                "documents": len(documents),
                "from_results": len(search_results),
                "success": fetch_success,
                "failed": fetch_failed,
                "duration_ms": (perf_counter() - start) * 1000,
            },
        )
        return {"documents": documents, "warnings": warnings}

    def chunk_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.chunk", run_id=state.get("run_id"))
        start = perf_counter()
        documents = state.get("documents", [])
        log.info(
            "chunk_start",
            extra={
                "documents": len(documents),
                "total_content_chars": sum(len(d.content) for d in documents),
            },
        )
        store.clear()
        chunks = chunk_documents(documents, chunker, run_id=state.get("run_id"))
        store.upsert(chunks)
        log.info(
            "chunk_complete",
            extra={
                "chunks": len(chunks),
                "duration_ms": (perf_counter() - start) * 1000,
                "avg_chunk_length": (
                    sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
                ),
            },
        )
        warnings = state.get("warnings", [])
        if not chunks:
            warnings.append("No chunks available; downstream answers may be empty.")
        return {"chunks": chunks, "warnings": warnings}

    def retrieve_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.retrieve", run_id=state.get("run_id"))
        start = perf_counter()
        chunks = state.get("chunks")
        if not chunks:
            log.warning("retrieve_no_chunks")
            warnings = state.get("warnings", [])
            warnings.append("No chunks to retrieve from; skipping retrieval.")
            return {"retrieved": [], "retrieval_scores": [], "warnings": warnings}
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
        warnings = state.get("warnings", [])
        if not retrieved:
            warnings.append("No retrieved context; answer may be unsupported.")
        return {"retrieved": retrieved, "retrieval_scores": scores, "warnings": warnings}

    def adaptive_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.adaptive", run_id=state.get("run_id"))
        start = perf_counter()
        warnings = state.get("warnings", [])
        adaptive_iterations = state.get("adaptive_iterations", 0)
        retrieved = state.get("retrieved", [])
        retrieval_scores = state.get("retrieval_scores", [])
        query_text = state.get("query", "")
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
        warnings.extend(assess_warnings)
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
            return {
                "warnings": warnings,
                "adaptive_iterations": adaptive_iterations,
            }
        log.info(
            "adaptive_triggered",
            extra={
                "reason": "insufficient_or_irrelevant_context",
                "retrieved": len(retrieved),
                "avg_score": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0,
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
            searx_host=searx_host,
            run_id=state.get("run_id"),
        )
        warnings.extend(search_warns)
        documents = []
        for res in results:
            doc, warn = fetch_url(res, run_id=state.get("run_id"))
            if doc:
                documents.append(doc)
            if warn:
                warnings.append(f"{res.url}: {warn}")
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
        return {
            "plan": plan,
            "search_results": results,
            "documents": documents,
            "chunks": chunks,
            "retrieved": retrieved,
            "retrieval_scores": retrieval_scores,
            "warnings": warnings,
            "adaptive_iterations": adaptive_iterations,
        }

    def generate_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.generate", run_id=state.get("run_id"))
        start = perf_counter()
        retrieved = state.get("retrieved", [])
        query = state.get("query", "")
        log.info(
            "generate_start",
            extra={"query": query, "retrieved_chunks": len(retrieved)},
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
        answer, citations = generate_answer(llm, query, retrieved)
        log.info(
            "generate_complete",
            extra={
                "citations": len(citations),
                "answer_length": len(answer),
                "duration_ms": (perf_counter() - start) * 1000,
            },
        )
        return {"draft_answer": answer, "citations": citations}

    def verify_node(state: GraphState) -> GraphState:
        log = get_logger("api.graph.verify", run_id=state.get("run_id"))
        start = perf_counter()
        draft = state.get("draft_answer", "")
        query = state.get("query", "")
        retrieved = state.get("retrieved", [])
        log.info(
            "verify_start",
            extra={"draft_length": len(draft), "retrieved_chunks": len(retrieved)},
        )
        verified = verify_answer(
            llm,
            draft,
            query,
            retrieved,
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
        assessment = assess_answer(
            query,
            answer,
            retrieved,
            run_id=state.get("run_id"),
            retrieval_scores=retrieval_scores,
        )
        notes = state.get("qc_notes", [])
        notes.append(f"pass {qc_passes}: issues={assessment['issues']}")
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
            return {"qc_passes": qc_passes, "qc_notes": notes}
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
            "qc_notes": notes,
        }

    graph = StateGraph(GraphState)
    graph.add_node("plan", lambda state: plan_node(cast(GraphState, state)))
    graph.add_node("search", lambda state: search_node(cast(GraphState, state)))
    graph.add_node("fetch", lambda state: fetch_node(cast(GraphState, state)))
    graph.add_node("chunk", lambda state: chunk_node(cast(GraphState, state)))
    graph.add_node("retrieve", lambda state: retrieve_node(cast(GraphState, state)))
    graph.add_node("generate", lambda state: generate_node(cast(GraphState, state)))
    graph.add_node("verify", lambda state: verify_node(cast(GraphState, state)))
    graph.add_node("adaptive", lambda state: adaptive_node(cast(GraphState, state)))
    graph.add_node("qc", lambda state: qc_node(cast(GraphState, state)))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "fetch")
    graph.add_edge("fetch", "chunk")
    graph.add_edge("chunk", "retrieve")
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
    searx_host: Optional[str] = None,
    top_k: int = TOP_K_RESULTS,
) -> ResearchState:
    """Run the full pipeline and return a ResearchState object."""
    run_id = str(uuid4())
    logger = get_logger(__name__, run_id=run_id)
    overall_start = perf_counter()
    logger.info(
        "run_research_start",
        extra={
            "query": query,
            "embed_model": embed_model,
            "top_k": top_k,
            "has_llm": llm is not None,
            "has_custom_store": vector_store is not None,
            "searx_host": searx_host,
        },
    )
    workflow = build_workflow(
        llm=llm,
        vector_store=vector_store,
        embed_model=embed_model,
        searx_host=searx_host,
        top_k=top_k,
        run_id=run_id,
    )
    logger.debug("run_research_workflow_built")
    initial_state = {
        "query": query,
        "run_id": run_id,
        "plan": [],
        "search_results": [],
        "documents": [],
        "chunks": [],
        "retrieved": [],
        "retrieval_scores": [],
        "draft_answer": "",
        "verified_answer": "",
        "citations": [],
        "errors": [],
        "warnings": [],
        "adaptive_iterations": 0,
        "qc_passes": 0,
        "qc_notes": [],
        "time_sensitive": False,
    }
    logger.debug("run_research_invoking_workflow")
    result = workflow.invoke(cast(GraphState, initial_state))
    duration_ms = (perf_counter() - overall_start) * 1000
    logger.info(
        "run_complete",
        extra={
            "duration_ms": duration_ms,
            "chunks": len(result.get("chunks", [])),
            "retrieved": len(result.get("retrieved", [])),
            "documents": len(result.get("documents", [])),
            "search_results": len(result.get("search_results", [])),
            "plan_tasks": len(result.get("plan", [])),
            "citations": len(result.get("citations", [])),
            "errors": len(result.get("errors", [])),
            "warnings": len(result.get("warnings", [])),
            "adaptive_iterations": result.get("adaptive_iterations", 0),
            "qc_passes": result.get("qc_passes", 0),
            "answer_length": len(
                result.get("verified_answer", "") or result.get("draft_answer", "")
            ),
        },
    )
    return ResearchState(
        query=query,
        run_id=run_id,
        plan=result.get("plan", []),
        search_results=result.get("search_results", []),
        documents=result.get("documents", []),
        chunks=result.get("chunks", []),
        retrieved=result.get("retrieved", []),
        draft_answer=result.get("draft_answer", ""),
        verified_answer=result.get("verified_answer", ""),
        citations=result.get("citations", []),
        errors=result.get("errors", []),
        warnings=result.get("warnings", []),
        adaptive_iterations=result.get("adaptive_iterations", 0),
        qc_passes=result.get("qc_passes", 0),
        qc_notes=result.get("qc_notes", []),
    )
