from api.memory import append_turn, resolve_followup_query, summarize_history, trim_history
from api.state import ConversationTurn


def test_trim_history_limits_turns():
    history = [ConversationTurn(query=f"Q{i}", answer="A") for i in range(5)]
    trimmed = trim_history(history, max_turns=2)
    assert len(trimmed) == 2
    assert trimmed[0].query == "Q3"


def test_append_turn_adds_entry_and_trims():
    history = [ConversationTurn(query="Q1", answer="A1")]
    updated = append_turn(history, "Q2", "A2", [], max_turns=1)
    assert len(updated) == 1
    assert updated[0].query == "Q2"


def test_summarize_history_returns_text():
    history = [
        ConversationTurn(query="Tell me about solar roofs", answer="Lots of info"),
        ConversationTurn(query="What about batteries?", answer="Battery info"),
    ]
    summary = summarize_history(history, max_turns=2)
    assert "solar roofs" in summary.lower()
    assert "batteries" in summary.lower()


def test_resolve_followup_query_injects_context():
    history = [
        ConversationTurn(
            query="Details on Tesla Model 3",
            answer="Electric sedan",
        )
    ]
    resolved = resolve_followup_query("What about its warranty?", history)
    assert "context" in resolved.lower()
    assert "tesla" in resolved.lower()
