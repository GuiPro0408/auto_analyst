from tools.planner import plan_query


class RecordingLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompt = ""

    def __call__(self, prompt):
        self.prompt = prompt
        return [{"generated_text": self.response}]


def test_plan_query_heuristic():
    query = "Impacts of solar energy adoption on the grid?"
    tasks, is_time_sensitive = plan_query(query, llm=None, max_tasks=3)
    assert tasks
    assert len(tasks) <= 3
    assert all(task.text for task in tasks)
    assert isinstance(is_time_sensitive, bool)


def test_plan_query_uses_conversation_context():
    query = "What about its battery warranty?"
    context = "Turn 1: Q: Tell me about Tesla Model 3\nA: Discussion about Tesla."
    recorder = RecordingLLM("example -- rationale")
    tasks, _ = plan_query(
        query,
        llm=recorder,
        max_tasks=3,
        conversation_context=context,
    )
    assert tasks
    assert "Prior conversation context" in recorder.prompt
