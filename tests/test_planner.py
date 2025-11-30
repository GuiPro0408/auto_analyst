from tools.planner import plan_query


def test_plan_query_heuristic():
    query = "Impacts of solar energy adoption on the grid?"
    tasks = plan_query(query, llm=None, max_tasks=3)
    assert tasks
    assert len(tasks) <= 3
    assert all(task.text for task in tasks)
