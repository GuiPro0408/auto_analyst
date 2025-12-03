from api.state import Chunk
from tools.generator import build_citations, generate_answer, verify_answer


class FakeLLM:
    def __call__(self, prompt):
        return [{"generated_text": "Answer: test answer with citation [1]"}]


class CapturingLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = ""

    def __call__(self, prompt):
        self.last_prompt = prompt
        return [{"generated_text": self.response}]


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


def test_generate_answer_receives_conversation_context():
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = CapturingLLM("Answer: ok [1]")
    context = "Turn 1: Q: Solar roofs\nA: Details"
    generate_answer(llm, "What about it?", [chunk], conversation_context=context)
    assert "Prior conversation summary" in llm.last_prompt


def test_verify_answer_receives_conversation_context():
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = CapturingLLM("Verified answer: ok [1]")
    context = "Turn 1: Q: Solar roofs\nA: Details"
    verify_answer(
        llm,
        "Draft answer",
        "What about it?",
        [chunk],
        conversation_context=context,
    )
    assert "Prior conversation summary" in llm.last_prompt
