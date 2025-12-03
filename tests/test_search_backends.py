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