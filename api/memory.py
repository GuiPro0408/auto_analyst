"""Conversation memory helpers for Auto-Analyst."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from api.config import (
    ANSWER_PREVIEW_MAX_LEN,
    CONVERSATION_MEMORY_TURNS,
    CONVERSATION_SUMMARY_CHARS,
)
from api.state import ConversationTurn

# Pronouns that typically indicate follow-up references
_FOLLOWUP_PRONOUNS = {
    "it",
    "its",
    "they",
    "them",
    "their",
    "theirs",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "him",
    "her",
}


def trim_history(
    history: Sequence[ConversationTurn],
    max_turns: int = CONVERSATION_MEMORY_TURNS,
) -> List[ConversationTurn]:
    """Return the most recent ``max_turns`` conversation turns."""
    if max_turns <= 0 or not history:
        return []
    return list(history[-max_turns:])


def summarize_history(
    history: Sequence[ConversationTurn],
    max_turns: int | None = None,
    max_chars: int = CONVERSATION_SUMMARY_CHARS,
) -> str:
    """Create a concise textual summary of recent conversation turns."""
    if not history:
        return ""

    turns = history[-(max_turns or min(3, len(history))) :]
    lines: List[str] = []
    for idx, turn in enumerate(turns, start=1):
        answer_preview = " ".join(turn.answer.strip().split())
        if len(answer_preview) > ANSWER_PREVIEW_MAX_LEN:
            answer_preview = answer_preview[: ANSWER_PREVIEW_MAX_LEN - 3] + "..."
        lines.append(f"Turn {idx}: Q: {turn.query.strip()}\nA: {answer_preview}")

    summary = "\n".join(lines).strip()
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return summary


def append_turn(
    history: Sequence[ConversationTurn],
    query: str,
    answer: str,
    citations: List[Dict[str, str]],
    max_turns: int = CONVERSATION_MEMORY_TURNS,
) -> List[ConversationTurn]:
    """Return a new history list with the latest turn appended and trimmed."""
    if not query:
        return trim_history(history, max_turns)

    new_history = list(history)
    new_history.append(
        ConversationTurn(query=query, answer=answer, citations=citations)
    )
    return trim_history(new_history, max_turns)


def _is_follow_up(query: str) -> bool:
    tokens = set(re.findall(r"\b[a-z]+\b", query.lower()))
    return any(token in _FOLLOWUP_PRONOUNS for token in tokens)


def resolve_followup_query(
    query: str,
    history: Sequence[ConversationTurn],
    context_chars: int = 200,
) -> str:
    """Inject lightweight context for follow-up questions that rely on pronouns."""
    if not history or not query:
        return query
    if not _is_follow_up(query):
        return query

    anchor = history[-1].query or history[-1].answer
    anchor = " ".join(anchor.split())
    if not anchor:
        return query

    if len(anchor) > context_chars:
        anchor = anchor[: context_chars - 3] + "..."
    return f"{query.strip()} (context: {anchor})"


__all__ = [
    "append_turn",
    "resolve_followup_query",
    "summarize_history",
    "trim_history",
]
