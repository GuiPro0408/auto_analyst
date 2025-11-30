from api.state import Chunk
from tools.generator import build_citations, generate_answer, verify_answer


class FakeLLM:
    def __call__(self, prompt):
        return [{"generated_text": "Answer: test answer with citation [1]"}]


def test_generate_answer_builds_citations():
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = FakeLLM()
    answer, citations = generate_answer(llm, "question", [chunk])
    assert "citation" in answer.lower()
    assert citations and citations[0]["marker"] == "[1]"


def test_verify_answer_pass_through():
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = FakeLLM()
    verified = verify_answer(llm, "Draft", "question", [chunk])
    assert verified
