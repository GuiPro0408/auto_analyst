"""Shared data structures for the research pipeline.

Design Note:
    This module uses two parallel type systems:

    1. **Dataclasses** (SearchQuery, SearchResult, Document, Chunk, ConversationTurn,
       ResearchState): Used for domain entities with methods and defaults. These are
       the types used throughout the application logic and as return values.

    2. **TypedDict** (GraphState): Required by LangGraph for state management.
       LangGraph nodes receive and return GraphState dicts, which get merged
       automatically by the framework.

    The ResearchState dataclass mirrors GraphState fields but provides a cleaner
    API for consumers. Use state_builder.py functions to convert between them.
"""

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class SearchQuery:
    """A search query with metadata for routing."""

    text: str
    rationale: str = ""
    topic: str = ""


@dataclass
class SearchResult:
    """A search result from any backend."""

    url: str
    title: str
    snippet: str = ""
    source: str = "web"
    content: str = ""  # Pre-fetched content (e.g., grounded search summaries)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A fetched document with parsed content."""

    url: str
    title: str
    content: str
    media_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A text chunk for embedding and retrieval."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""

    query: str
    answer: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConversationTurn":
        return cls(
            query=payload.get("query", ""),
            answer=payload.get("answer", ""),
            citations=payload.get("citations", []),
            timestamp=float(payload.get("timestamp") or time()),
        )


@dataclass
class ResearchState:
    """Final state returned by the research pipeline.

    This dataclass provides a clean API for consumers. It mirrors GraphState
    but with proper defaults and helper methods.
    """

    query: str
    run_id: str = ""
    plan: List[SearchQuery] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    retrieved: List[Chunk] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    draft_answer: str = ""
    verified_answer: str = ""
    citations: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adaptive_iterations: int = 0
    qc_passes: int = 0
    qc_notes: List[str] = field(default_factory=list)
    time_sensitive: bool = False
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    grounded_answer: str = ""
    grounded_sources: List[Chunk] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Append an error message to the errors list."""
        self.errors.append(message)

    def has_grounded_answer(self) -> bool:
        """Check if this state has a grounded answer from Gemini."""
        return bool(self.grounded_answer and self.grounded_sources)


class GraphState(TypedDict, total=False):
    """LangGraph state dictionary for pipeline nodes.

    This TypedDict is required by LangGraph. Nodes receive and return
    partial dicts that get merged into the full state. Use total=False
    to allow partial updates.
    """

    query: str
    run_id: str
    plan: List[SearchQuery]
    search_results: List[SearchResult]
    documents: List[Document]
    chunks: List[Chunk]
    retrieved: List[Chunk]
    retrieval_scores: List[float]  # Similarity scores from vector store query
    draft_answer: str
    verified_answer: str
    citations: List[Dict[str, str]]
    errors: List[str]
    warnings: List[str]
    adaptive_iterations: int
    qc_passes: int
    qc_notes: List[str]
    time_sensitive: bool  # Flag for time-sensitive queries
    conversation_history: List[ConversationTurn]
    grounded_answer: str  # Direct answer from Gemini grounding
    grounded_sources: List[Chunk]  # Sources from grounding for citations
