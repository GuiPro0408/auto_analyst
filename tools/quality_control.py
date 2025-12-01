"""Quality control agent for iterative answer improvement."""

from typing import Dict, List, Optional

from api.logging_setup import get_logger
from api.state import Chunk
from tools.generator import generate_answer, verify_answer


def assess_answer(
    question: str, answer: str, contexts: List[Chunk], run_id: Optional[str] = None
) -> Dict[str, object]:
    """Lightweight heuristic assessment to decide if further improvement is needed."""
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "assess_answer_start",
        extra={
            "question": question,
            "answer_length": len(answer),
            "contexts": len(contexts),
        },
    )
    issues: List[str] = []
    if not answer or not answer.strip():
        issues.append("Answer is empty.")
        logger.warning("assess_answer_empty")
    if not contexts:
        issues.append("No supporting context retrieved.")
        logger.warning("assess_answer_no_context")
    # Require at least one citation marker
    if "[" not in answer:
        issues.append("Missing citations.")
        logger.warning("assess_answer_no_citations")
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
