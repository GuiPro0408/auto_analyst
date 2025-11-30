"""Quality control agent for iterative answer improvement."""

from typing import Dict, List

from api.state import Chunk
from tools.generator import generate_answer, verify_answer


def assess_answer(
    question: str, answer: str, contexts: List[Chunk]
) -> Dict[str, object]:
    """Lightweight heuristic assessment to decide if further improvement is needed."""
    issues: List[str] = []
    if not answer or not answer.strip():
        issues.append("Answer is empty.")
    if not contexts:
        issues.append("No supporting context retrieved.")
    # Require at least one citation marker
    if "[" not in answer:
        issues.append("Missing citations.")
    is_good_enough = len(issues) == 0
    return {"is_good_enough": is_good_enough, "issues": issues}


def improve_answer(llm, question: str, answer: str, contexts: List[Chunk]) -> str:
    """Use the existing generation/verification flow to improve the answer."""
    # Reuse generator + verifier prompts for refinement
    draft, citations = generate_answer(llm, question, contexts)
    improved = verify_answer(llm, draft, question, contexts)
    return improved
