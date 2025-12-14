"""State building utilities extracted from graph.py."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.logging_setup import get_logger
from api.state import (
    Chunk,
    ConversationTurn,
    ResearchState,
    SearchResult,
)
from tools.generator import build_citations


def normalize_conversation_history(
    history: Optional[List[Any]],
    run_id: Optional[str] = None,
) -> List[ConversationTurn]:
    """Normalize conversation history to list of ConversationTurn.

    Args:
        history: List of ConversationTurn or dict objects.
        run_id: Optional run ID for logging.

    Returns:
        Normalized list of ConversationTurn objects.
    """
    logger = get_logger(__name__, run_id=run_id)

    if not history:
        return []

    normalized: List[ConversationTurn] = []
    for idx, turn in enumerate(history):
        if isinstance(turn, ConversationTurn):
            normalized.append(turn)
        elif isinstance(turn, dict):
            try:
                normalized.append(ConversationTurn.from_dict(turn))
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "conversation_turn_invalid",
                    extra={"index": idx, "error": str(exc)},
                )
        else:
            logger.warning(
                "conversation_turn_unknown_type",
                extra={"index": idx, "type": type(turn).__name__},
            )

    return normalized


def extract_grounded_answer(
    search_results: List[SearchResult],
    run_id: Optional[str] = None,
) -> tuple[str, List[Chunk]]:
    """Extract grounded answer and sources from search results.

    Args:
        search_results: List of search results from the pipeline.
        run_id: Optional run ID for logging.

    Returns:
        Tuple of (grounded_answer, grounded_sources).
    """
    logger = get_logger(__name__, run_id=run_id)
    grounded_answer = ""
    grounded_sources: List[Chunk] = []

    for idx, sr in enumerate(search_results):
        if sr.source == "gemini_grounding" and sr.content:
            grounded_answer = sr.content
            grounded_sources = [
                Chunk(
                    id=f"grounding_{j}",
                    text=source.snippet or source.title,
                    metadata={
                        "url": source.url,
                        "title": source.title,
                        "source": "gemini_grounding",
                        "media_type": "text",
                    },
                )
                for j, source in enumerate(search_results)
                if source.source == "gemini_grounding" and source.url
            ]
            logger.debug(
                "grounded_answer_extracted",
                extra={
                    "answer_length": len(grounded_answer),
                    "sources_count": len(grounded_sources),
                },
            )
            break

    return grounded_answer, grounded_sources


def build_research_state(
    query: str,
    run_id: str,
    result: Dict[str, Any],
    conversation_history: List[ConversationTurn],
) -> ResearchState:
    """Build a ResearchState from pipeline result.

    Args:
        query: Original query string.
        run_id: Run correlation ID.
        result: Raw result dict from workflow.
        conversation_history: Updated conversation history.

    Returns:
        Populated ResearchState object.
    """
    logger = get_logger(__name__, run_id=run_id)

    # Extract grounded data from result
    grounded_answer = result.get("grounded_answer") or ""
    grounded_sources = result.get("grounded_sources", []) or []
    search_results = result.get("search_results", []) or []

    # Try to extract grounded answer from search results if not in state
    if not grounded_answer:
        grounded_answer, grounded_sources = extract_grounded_answer(
            search_results, run_id=run_id
        )

    # Determine final answer values
    draft_answer = result.get("draft_answer", "")
    citations = result.get("citations", [])
    retrieved = result.get("retrieved", [])

    if grounded_answer:
        remapped, citations = build_citations(grounded_sources, grounded_answer)
        draft_answer = remapped
        if grounded_sources:
            retrieved = grounded_sources

    verified_answer = result.get("verified_answer", "")
    if grounded_answer and not verified_answer.startswith("Verified answer"):
        verified_answer = f"Verified answer: {draft_answer}"

    logger.debug(
        "research_state_built",
        extra={
            "has_grounded_answer": bool(grounded_answer),
            "draft_length": len(draft_answer),
            "verified_length": len(verified_answer),
            "citations_count": len(citations),
            "history_turns": len(conversation_history),
        },
    )

    return ResearchState(
        query=query,
        run_id=run_id,
        plan=result.get("plan", []),
        search_results=search_results,
        documents=result.get("documents", []),
        chunks=result.get("chunks", []),
        retrieved=retrieved,
        retrieval_scores=result.get("retrieval_scores", []),
        draft_answer=draft_answer,
        verified_answer=verified_answer,
        citations=citations,
        errors=result.get("errors", []),
        warnings=result.get("warnings", []),
        adaptive_iterations=result.get("adaptive_iterations", 0),
        qc_passes=result.get("qc_passes", 0),
        qc_notes=result.get("qc_notes", []),
        time_sensitive=result.get("time_sensitive", False),
        conversation_history=conversation_history,
        grounded_answer=grounded_answer,
        grounded_sources=grounded_sources,
        query_type=result.get("query_type", "factual"),
    )


def create_initial_state(
    query: str,
    run_id: str,
    conversation_history: List[ConversationTurn],
    query_type: str = "factual",
) -> Dict[str, Any]:
    """Create the initial state dict for the workflow.

    Args:
        query: The search query.
        run_id: Run correlation ID.
        conversation_history: Prior conversation turns.
        query_type: Query classification (factual, recommendation, creative).

    Returns:
        Initial state dictionary.
    """
    return {
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
        "conversation_history": conversation_history,
        "grounded_answer": "",
        "grounded_sources": [],
        "query_type": query_type,
    }
