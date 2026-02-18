"""Tests for quality control citation behavior."""

import pytest

from api.state import Chunk
from tools.quality_control import assess_answer, improve_answer


@pytest.mark.unit
def test_assess_answer_requires_numeric_citation_markers(monkeypatch):
    monkeypatch.setattr("api.backend_utils.LLM_BACKEND", "gemini")
    contexts = [
        Chunk(
            id="1",
            text="context",
            metadata={"title": "Doc", "url": "http://example.com"},
        )
    ]

    bad = assess_answer("question", "This has a bracket only [", contexts)
    good = assess_answer("question", "This has citation [1]", contexts)

    assert "Missing citations." in bad["issues"]
    assert "Missing citations." not in good["issues"]


@pytest.mark.unit
def test_improve_answer_uses_repair_mode_for_missing_citations(monkeypatch):
    monkeypatch.setattr("api.backend_utils.LLM_BACKEND", "gemini")
    contexts = [
        Chunk(
            id="1",
            text="context",
            metadata={"title": "Doc", "url": "http://example.com"},
        )
    ]
    captured: dict[str, str] = {}

    def fake_generate_answer(*args, **kwargs):
        captured["citation_mode"] = kwargs.get("citation_mode", "default")
        return "Draft with citation [1]", [{"marker": "[1]"}]

    monkeypatch.setattr("tools.quality_control.generate_answer", fake_generate_answer)
    monkeypatch.setattr(
        "tools.quality_control.verify_answer",
        lambda *args, **kwargs: "Improved answer [1]",
    )

    improved = improve_answer(
        llm=None,
        question="question",
        answer="answer without citations",
        contexts=contexts,
        issues=["Missing citations."],
    )

    assert captured["citation_mode"] == "repair"
    assert improved == "Improved answer [1]"
