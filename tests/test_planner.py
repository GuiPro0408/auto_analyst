from tools.planner import plan_query


def test_plan_query_heuristic():
    query = "Impacts of solar energy adoption on the grid?"
    tasks, is_time_sensitive = plan_query(query, max_tasks=3)
    assert tasks
    assert len(tasks) <= 3
    assert all(task.text for task in tasks)
    assert isinstance(is_time_sensitive, bool)


def test_plan_query_uses_conversation_context():
    query = "What about its battery warranty?"
    context = "Turn 1: Q: Tell me about Tesla Model 3\nA: Discussion about Tesla."
    tasks_with_context, _ = plan_query(
        query, max_tasks=3, conversation_context=context
    )
    tasks_without_context, _ = plan_query(query, max_tasks=3)

    assert tasks_with_context
    assert [t.text for t in tasks_with_context] != [t.text for t in tasks_without_context]
