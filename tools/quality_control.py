"""Quality control agent for iterative answer improvement."""

from typing import Dict, List, Optional

from api.logging_setup import get_logger
from api.state import Chunk
from tools.generator import generate_answer, verify_answer

# Phrases that indicate the LLM failed to generate a proper answer
FALLBACK_PHRASES = [
    "Unable to generate",
    "AI was unable to synthesize",
    "could not find relevant information",
    "No context retrieved",
    "No sufficient context",
]


def _needs_structured_list(question: str) -> bool:
    q = question.lower()
    triggers = {
        "list",
        "releases",
        "releasing",
        "release",
        "lineup",
        "schedule",
        "standings",
        "ranking",
        "table",
        "top",
        "currently",
        "current",
        "today",
        "upcoming",
        "fall",
        "spring",
        "summer",
        "winter",
    }
    return any(term in q for term in triggers)


def assess_answer(
    question: str,
    answer: str,
    contexts: List[Chunk],
    run_id: Optional[str] = None,
    retrieval_scores: Optional[List[float]] = None,
) -> Dict[str, object]:
    """Lightweight heuristic assessment to decide if further improvement is needed.

    Checks:
    1. Answer is not empty
    2. Supporting contexts exist
    3. Answer has citations
    4. Answer does not contain fallback/failure phrases
    5. Retrieval scores are above minimum threshold (if provided)
    """
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "assess_answer_start",
        extra={
            "question": question,
            "answer_length": len(answer),
            "contexts": len(contexts),
            "has_retrieval_scores": retrieval_scores is not None,
        },
    )
    issues: List[str] = []

    # Check 1: Answer is not empty
    if not answer or not answer.strip():
        issues.append("Answer is empty.")
        logger.warning("assess_answer_empty")

    # Check 2: Supporting contexts exist
    if not contexts:
        issues.append("No supporting context retrieved.")
        logger.warning("assess_answer_no_context")

    # Check 3: Require at least one citation marker
    if "[" not in answer:
        issues.append("Missing citations.")
        logger.warning("assess_answer_no_citations")

    # Check 4: Answer should not contain fallback phrases indicating failure
    for phrase in FALLBACK_PHRASES:
        if phrase.lower() in answer.lower():
            issues.append(f"Answer contains fallback phrase: '{phrase}'")
            logger.warning(
                "assess_answer_fallback_detected",
                extra={"phrase": phrase},
            )
            break  # Only report first fallback phrase found

    # Check 5: Retrieval scores should be above threshold (if available)
    if retrieval_scores and len(retrieval_scores) > 0:
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        min_qc_threshold = 0.25
        if avg_score < min_qc_threshold:
            issues.append(
                f"Low retrieval relevance (avg: {avg_score:.3f} < {min_qc_threshold})"
            )
            logger.warning(
                "assess_answer_low_retrieval_relevance",
                extra={
                    "avg_score": round(avg_score, 4),
                    "threshold": min_qc_threshold,
                },
            )

    # Check 6: List-style questions should return structured bullets/numbered items
    if _needs_structured_list(question):
        if "-" not in answer and "1." not in answer:
            issues.append("Missing structured list format for a list-style question.")
            logger.warning("assess_answer_missing_list_structure")

    is_good_enough = len(issues) == 0
    logger.info(
        "assess_answer_complete",
        extra={
            "is_good_enough": is_good_enough,
            "issues": issues,
            "issue_count": len(issues),
        },
    )
    return {"is_good_enough": is_good_enough, "issues": issues}


def improve_answer(
    llm, question: str, answer: str, contexts: List[Chunk], run_id: Optional[str] = None
) -> str:
    """Use the existing generation/verification flow to improve the answer."""
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "improve_answer_start",
        extra={
            "question": question,
            "current_answer_length": len(answer),
            "contexts": len(contexts),
        },
    )
    # Reuse generator + verifier prompts for refinement
    draft, citations = generate_answer(llm, question, contexts)
    logger.debug(
        "improve_answer_draft_generated",
        extra={"draft_length": len(draft), "citations": len(citations)},
    )
    improved = verify_answer(llm, draft, question, contexts)
    logger.info(
        "improve_answer_complete",
        extra={
            "improved_length": len(improved),
            "improvement_delta": len(improved) - len(answer),
        },
    )
    return improved
