"""Shared data structures for the research pipeline."""

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class SearchQuery:
    text: str
    rationale: str = ""


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str = ""
    source: str = "web"
    content: str = ""  # Pre-fetched content (e.g., Wikipedia summaries)


@dataclass
class Document:
    url: str
    title: str
    content: str
    media_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
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
    query: str
    run_id: str = ""
    plan: List[SearchQuery] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    retrieved: List[Chunk] = field(default_factory=list)
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

    def add_error(self, message: str) -> None:
        self.errors.append(message)


class GraphState(TypedDict, total=False):
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
