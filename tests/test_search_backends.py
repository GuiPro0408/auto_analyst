from tools import search


class DummyResponse:
    def __init__(self, text="", json_data=None):
        self.text = text
        self._json = json_data or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


def test_wikipedia_backend_returns_results(monkeypatch):
    """Test WikipediaBackend parses API response correctly."""
    backend = search.WikipediaBackend()
    json_payload = {
        "query": {
            "search": [
                {
                    "title": "Test Article",
                    "snippet": "This is a <span class=\"searchmatch\">test</span> snippet.",
                }
            ]
        }
    }

    def fake_get(*_, **__):
        return DummyResponse(json_data=json_payload)

    monkeypatch.setattr(search.requests, "get", fake_get)

    results = backend.search("test", max_results=1)
    assert len(results) == 1
    assert results[0].source == "wikipedia"
    assert "test" in results[0].snippet.lower()
    assert "searchmatch" not in results[0].snippet  # HTML cleaned


def test_run_search_tasks_falls_back_to_gemini(monkeypatch):
    class EmptyBackend(search.SearchBackend):
        name = "empty"

        def search(self, *_, **__):
            return []

        def supports_topic(self, *_):
            return True

    class GeminiBackend(search.SearchBackend):
        name = "gemini_grounding"

        def search(self, *_, **__):
            return [
                search.SearchResult(
                    url="https://www.premierleague.com/table",
                    title="Premier League Table",
                    snippet="Arsenal lead the table.",
                    source="gemini_grounding",
                    content="Arsenal lead the table.",
                )
            ]

        def supports_topic(self, *_):
            return True

    # Force search backends to return nothing, ensuring we fall back to Gemini
    monkeypatch.setattr(search, "SEARCH_BACKENDS", ["empty"])
    monkeypatch.setattr(search, "get_backend", lambda name, **__: GeminiBackend() if name == "gemini_grounding" else EmptyBackend())

    tasks = [search.SearchQuery(text="current premier league standings", rationale="")]
    results, warnings = search.run_search_tasks(tasks, max_results=3)

    assert any(r.source == "gemini_grounding" for r in results)
    assert warnings == [] or warnings == ["No search results found; consider refining the query."]


def test_run_search_tasks_fallback_chain(monkeypatch):
    class EmptyBackend(search.SearchBackend):
        name = "empty"

        def search(self, *_, **__):
            return []

        def supports_topic(self, *_):
            return True

    class WikiBackend(search.SearchBackend):
        name = "wikipedia"

        def search(self, *_, **__):
            return [
                search.SearchResult(
                    url="https://en.wikipedia.org/wiki/Premier_League",
                    title="Premier League",
                    snippet="Premier League standings",
                    source="wikipedia",
                    content="Premier League standings",
                )
            ]

        def supports_topic(self, *_):
            return True

    monkeypatch.setattr(search, "SEARCH_BACKENDS", ["empty"])
    monkeypatch.setattr(
        search, "FALLBACK_BACKEND_ORDER", ["wikipedia", "gemini_grounding"]
    )
    monkeypatch.setattr(
        search,
        "get_backend",
        lambda name, **__: WikiBackend()
        if name == "wikipedia"
        else (EmptyBackend() if name == "empty" else None),
    )

    tasks = [search.SearchQuery(text="current premier league standings", rationale="")]
    results, warnings = search.run_search_tasks(tasks, max_results=3)

    assert any(r.source == "wikipedia" for r in results)
    assert warnings == [] or warnings == ["No search results found; consider refining the query."]


def test_run_search_tasks_seed_sports_when_empty(monkeypatch):
    class EmptyBackend(search.SearchBackend):
        name = "empty"

        def search(self, *_, **__):
            return []

        def supports_topic(self, *_):
            return True

    monkeypatch.setattr(search, "SEARCH_BACKENDS", ["empty"])
    monkeypatch.setattr(
        search,
        "get_backend",
        lambda name, **__: EmptyBackend() if name == "empty" else None,
    )
    monkeypatch.setattr(search, "FALLBACK_BACKEND_ORDER", [])

    tasks = [search.SearchQuery(text="premier league standings", rationale="", topic="sports")]
    results, warnings = search.run_search_tasks(tasks, max_results=3)

    assert results == []
    assert warnings == ["No search results found; consider refining the query."]


def test_run_search_tasks_seed_news_when_empty(monkeypatch):
    class EmptyBackend(search.SearchBackend):
        name = "empty"

        def search(self, *_, **__):
            return []

        def supports_topic(self, *_):
            return True

    monkeypatch.setattr(search, "SEARCH_BACKENDS", ["empty"])
    monkeypatch.setattr(
        search,
        "get_backend",
        lambda name, **__: EmptyBackend() if name == "empty" else None,
    )
    monkeypatch.setattr(search, "FALLBACK_BACKEND_ORDER", [])

    tasks = [search.SearchQuery(text="latest openai news", rationale="", topic="news")]
    results, warnings = search.run_search_tasks(tasks, max_results=3)

    assert results == []
    assert warnings == ["No search results found; consider refining the query."]
