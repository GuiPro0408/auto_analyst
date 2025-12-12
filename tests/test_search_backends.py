from tools import search


class DummyResponse:
    def __init__(self, text="", json_data=None):
        self.text = text
        self._json = json_data or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


def test_arxiv_backend_parses_feed(monkeypatch):
    backend = search.ArxivBackend()
    sample_xml = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Sample Paper</title>
        <summary>This is a sample abstract.</summary>
        <id>http://arxiv.org/abs/1234.5678</id>
      </entry>
    </feed>
    """.strip()

    def fake_get(*_, **__):
        return DummyResponse(text=sample_xml)

    monkeypatch.setattr(search.requests, "get", fake_get)

    results = backend.search("quantum", max_results=1)
    assert len(results) == 1
    assert results[0].source == "arxiv"
    assert "sample" in results[0].snippet.lower()


def test_openalex_backend_builds_snippet(monkeypatch):
    backend = search.OpenAlexBackend()
    json_payload = {
        "results": [
            {
                "display_name": "OpenAlex Paper",
                "id": "https://openalex.org/W123",
                "abstract_inverted_index": {
                    "hello": [0],
                    "world": [1],
                },
            }
        ]
    }

    def fake_get(*_, **__):
        return DummyResponse(json_data=json_payload)

    monkeypatch.setattr(search.requests, "get", fake_get)

    results = backend.search("science", max_results=1)
    assert len(results) == 1
    assert results[0].source == "openalex"
    assert results[0].snippet == "hello world"


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
