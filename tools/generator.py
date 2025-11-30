"""Answer generation and verification helpers."""

from typing import Dict, List, Tuple

from api.state import Chunk


def _format_context(chunks: List[Chunk]) -> str:
    lines = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        title = meta.get("title") or "Source"
        url = meta.get("url") or ""
        lines.append(f"[{idx + 1}] {title} ({url}) :: {chunk.text}")
    return "\n".join(lines)


def build_citations(chunks: List[Chunk]) -> List[Dict[str, str]]:
    citations = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        citations.append(
            {
                "marker": f"[{idx + 1}]",
                "url": meta.get("url", ""),
                "title": meta.get("title", "Source"),
                "media_type": meta.get("media_type", "text"),
            }
        )
    return citations


def generate_answer(
    llm, query: str, retrieved: List[Chunk]
) -> Tuple[str, List[Dict[str, str]]]:
    if not retrieved:
        return "No context retrieved to answer the question.", []

    context_block = _format_context(retrieved)
    prompt = (
        "You are an evidence-based research assistant. Using only the context provided, "
        "write a concise answer to the user question. Cite supporting evidence inline using [n] "
        "where n matches the numbered context entries. Do not fabricate details.\n\n"
        f"User question: {query}\n\nContext:\n{context_block}\n\nAnswer:"
    )
    output = llm(prompt)[0]["generated_text"]
    # Strip prompt echo if present
    answer = (
        output.split("Answer:", 1)[-1].strip()
        if "Answer:" in output
        else output.strip()
    )
    citations = build_citations(retrieved)
    return answer, citations


def verify_answer(llm, draft: str, query: str, retrieved: List[Chunk]) -> str:
    if not retrieved:
        return draft
    context_block = _format_context(retrieved)
    prompt = (
        "You are a fact-checking verifier. Review the draft answer against the provided context. "
        "Remove or correct any statements that are not directly supported. Preserve inline citations [n] "
        "only when the claim is supported by the corresponding context entry. Keep the answer concise.\n\n"
        f"User question: {query}\n\nContext:\n{context_block}\n\nDraft answer:\n{draft}\n\nVerified answer:"
    )
    output = llm(prompt)[0]["generated_text"]
    return (
        output.split("Verified answer:", 1)[-1].strip()
        if "Verified answer:" in output
        else output.strip()
    )
