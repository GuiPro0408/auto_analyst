"""Lightweight healthcheck to validate imports and workflow wiring.

Avoids loading heavy models; uses lightweight stubs instead.
"""

from uuid import uuid4

from api.graph import build_workflow
from vector_store.base import ScoredChunk, VectorStore
from api.state import Chunk


class _StubLLM:
    def __call__(self, prompt):
        return [{"generated_text": "stub"}]


class _StubStore(VectorStore):
    def __init__(self):
        self.chunks = []

    def upsert(self, chunks):
        self.chunks.extend(chunks)

    def query(self, text, top_k=1):
        return [ScoredChunk(chunk=Chunk(id="1", text="stub", metadata={}), score=1.0)]


def main():
    # Ensure workflow can compile and run one pass with stubs
    app = build_workflow(llm=_StubLLM(), vector_store=_StubStore())
    initial_state = {
        "query": "ping",
        "run_id": str(uuid4()),
        "plan": [],
        "search_results": [],
        "documents": [],
        "chunks": [],
        "retrieved": [],
        "draft_answer": "",
        "verified_answer": "",
        "citations": [],
        "errors": [],
        "warnings": [],
        "adaptive_iterations": 0,
        "qc_passes": 0,
        "qc_notes": [],
    }
    app.invoke(initial_state)
    print("OK")


if __name__ == "__main__":
    main()
