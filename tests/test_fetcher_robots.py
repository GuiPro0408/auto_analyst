import types

from tools import fetcher


class FakeRobotFileParser:
    def __init__(self, allowed=True, raise_on_read=False):
        self.allowed = allowed
        self.raise_on_read = raise_on_read
        self.read_calls = 0

    def set_url(self, url):
        self.url = url

    def read(self):
        self.read_calls += 1
        if self.raise_on_read:
            raise RuntimeError("robots read failed")

    def can_fetch(self, user_agent, url):
        return self.allowed


def test_is_allowed_denies_on_robots_failure(monkeypatch):
    fake_rp = FakeRobotFileParser(raise_on_read=True)
    monkeypatch.setattr(fetcher, "RobotFileParser", lambda: fake_rp)
    fetcher._robots_cache = {}

    allowed = fetcher.is_allowed("https://example.com/page")

    assert allowed is False
    assert fake_rp.read_calls == 1
    # cached denial should be stored
    assert fetcher._robots_cache.get("example.com")[0] is False


def test_is_allowed_uses_cache(monkeypatch):
    fake_rp = FakeRobotFileParser(allowed=True)
    monkeypatch.setattr(fetcher, "RobotFileParser", lambda: fake_rp)
    fetcher._robots_cache = {}

    first = fetcher.is_allowed("https://example.com/page")
    second = fetcher.is_allowed("https://example.com/other")

    assert first is True
    assert second is True
    # read called only once due to cache hit
    assert fake_rp.read_calls == 1
