"""LangGraph orchestration for the Auto-Analyst pipeline."""

from time import perf_counter
from typing import Dict, Optional
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

    def plan_node(state: Dict):
        log = get_logger("api.graph.plan", run_id=state.get("run_id"))
        start = perf_counter()
        try:
            plan = plan_query(state["query"], llm=llm)
            log.info(
                "plan_complete",
                extra={
                    "tasks": len(plan),
                    "duration_ms": (perf_counter() - start) * 1000,
                    "query": state["query"],
                },
            )
            return {"plan": plan}
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("plan_failed")
            errors = state.get("errors", [])
            errors.append(f"plan_failed: {exc}")
            return {"plan": [], "errors": errors}

    def search_node(state: Dict):
        log = get_logger("api.graph.search", run_id=state.get("run_id"))
        start = perf_counter()
        results, warnings = run_search_tasks(
            state["plan"],
            max_results=5,
            searx_host=searx_host,
            run_id=state.get("run_id"),
        )
        log.info(
            "search_complete",
            extra={
                "results": len(results),
                "duration_ms": (perf_counter() - start) * 1000,
            },
        )
        state_warnings = state.get("warnings", [])
        state_warnings.extend(warnings)
        return {"search_results": results, "warnings": state_warnings}

    def fetch_node(state: Dict):
        log = get_logger("api.graph.fetch", run_id=state.get("run_id"))
        documents = []
        warnings = state.get("warnings", [])
        for res in state.get("search_results", []):
            doc, warn = fetch_url(res, run_id=state.get("run_id"))
            if doc:
                documents.append(doc)
            if warn:
                warnings.append(f"{res.url}: {warn}")
        if not documents:
            warnings.append("No documents fetched from search results.")
        log.info(
            "fetch_complete",
            extra={
                "documents": len(documents),
                "from_results": len(state.get("search_results", [])),
            },
        )
        return {"documents": documents, "warnings": warnings}

    def chunk_node(state: Dict):
        log = get_logger("api.graph.chunk", run_id=state.get("run_id"))
        store.clear()
        chunks = chunk_documents(
            state.get("documents", []), chunker, run_id=state.get("run_id")
        )
        store.upsert(chunks)
        log.info("chunk_complete", extra={"chunks": len(chunks)})
        warnings = state.get("warnings", [])
        if not chunks:
            warnings.append("No chunks available; downstream answers may be empty.")
        return {"chunks": chunks, "warnings": warnings}

    def retrieve_node(state: Dict):
        log = get_logger("api.graph.retrieve", run_id=state.get("run_id"))
        scored = store.query(state["query"], top_k=top_k)
        retrieved = [sc.chunk for sc in scored]
        log.info(
            "retrieve_complete", extra={"retrieved": len(retrieved), "top_k": top_k}
        )
        warnings = state.get("warnings", [])
        if not retrieved:
            warnings.append("No retrieved context; answer may be unsupported.")
        return {"retrieved": retrieved, "warnings": warnings}

    def adaptive_node(state: Dict):
        log = get_logger("api.graph.adaptive", run_id=state.get("run_id"))
        warnings = state.get("warnings", [])
        adaptive_iterations = state.get("adaptive_iterations", 0)
        needs_more, assess_warnings = assess_context(
            state.get("retrieved", []), min_chunks=1
        )
        warnings.extend(assess_warnings)
        if not needs_more or adaptive_iterations >= ADAPTIVE_MAX_ITERS:
            return {
                "warnings": warnings,
                "adaptive_iterations": adaptive_iterations,
            }
        new_tasks = refine_plan(state["query"], state.get("plan", []))
        plan = state.get("plan", []) + new_tasks
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
        scored = store.query(state["query"], top_k=top_k)
        retrieved = [sc.chunk for sc in scored]
        adaptive_iterations += 1
        log.info(
            "adaptive_cycle_complete",
            extra={
                "adaptive_iterations": adaptive_iterations,
                "tasks": len(plan),
                "retrieved": len(retrieved),
            },
        )
        return {
            "plan": plan,
            "search_results": results,
            "documents": documents,
            "chunks": chunks,
            "retrieved": retrieved,
            "warnings": warnings,
            "adaptive_iterations": adaptive_iterations,
        }

    def generate_node(state: Dict):
        log = get_logger("api.graph.generate", run_id=state.get("run_id"))
        answer, citations = generate_answer(
            llm, state["query"], state.get("retrieved", [])
        )
        log.info("generate_complete", extra={"citations": len(citations)})
        return {"draft_answer": answer, "citations": citations}

    def verify_node(state: Dict):
        log = get_logger("api.graph.verify", run_id=state.get("run_id"))
        verified = verify_answer(
            llm,
            state.get("draft_answer", ""),
            state["query"],
            state.get("retrieved", []),
        )
        log.info("verify_complete")
        return {"verified_answer": verified}

    def qc_node(state: Dict):
        log = get_logger("api.graph.qc", run_id=state.get("run_id"))
        qc_passes = state.get("qc_passes", 0)
        answer = state.get("verified_answer") or state.get("draft_answer", "")
        assessment = assess_answer(state["query"], answer, state.get("retrieved", []))
        notes = state.get("qc_notes", [])
        notes.append(f"pass {qc_passes}: issues={assessment['issues']}")
        if assessment["is_good_enough"] or qc_passes >= QC_MAX_PASSES:
            log.info(
                "qc_complete", extra={"qc_passes": qc_passes, "status": "accepted"}
            )
            return {"qc_passes": qc_passes, "qc_notes": notes}
        improved = improve_answer(
            llm, state["query"], answer, state.get("retrieved", [])
        )
        qc_passes += 1
        log.info(
            "qc_refine",
            extra={"qc_passes": qc_passes, "issues": assessment["issues"]},
        )
        return {
            "verified_answer": improved,
            "qc_passes": qc_passes,
            "qc_notes": notes,
        }

    graph = StateGraph(GraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("adaptive", adaptive_node)
    graph.add_node("qc", qc_node)

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
    workflow = build_workflow(
        llm=llm,
        vector_store=vector_store,
        embed_model=embed_model,
        searx_host=searx_host,
        top_k=top_k,
        run_id=run_id,
    )
    initial_state = {
        "query": query,
        "run_id": run_id,
        "plan": [],
        "search_results": [],
        "documents": [],
        "chunks": [],
        "retrieved": [],
        "draft_answer": "",
        "verified_answer": "",
        "citations": [],
        "errors": [],
        "warnings": [],
        "adaptive_iterations": 0,
        "qc_passes": 0,
        "qc_notes": [],
    }
    result = workflow.invoke(initial_state)
    logger.info(
        "run_complete",
        extra={
            "duration_ms": (perf_counter() - overall_start) * 1000,
            "chunks": len(result.get("chunks", [])),
            "retrieved": len(result.get("retrieved", [])),
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
