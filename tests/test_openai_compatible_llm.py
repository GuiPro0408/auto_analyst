"""Tests for OpenAICompatibleLLM auth fail-fast and fallback behavior."""

import httpx
import pytest

from tools.openai_compatible_llm import OpenAICompatibleLLM


def _make_response(status_code: int, payload: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(
        status_code,
        request=request,
        json=payload
        or {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                    }
                }
            ]
        },
    )


class _FakeClient:
    responses: list[httpx.Response] = []
    post_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        _FakeClient.post_calls += 1
        return _FakeClient.responses.pop(0)


def test_auth_failure_403_uses_fallback_and_enters_cooldown(monkeypatch):
    _FakeClient.responses = [
        _make_response(403, {"error": {"message": "forbidden"}}),
    ]
    _FakeClient.post_calls = 0
    monkeypatch.setattr("tools.openai_compatible_llm.httpx.Client", _FakeClient)

    fallback_calls: list[str] = []

    def fallback(prompt: str):
        fallback_calls.append(prompt)
        return [{"generated_text": "fallback"}]

    llm = OpenAICompatibleLLM(
        model_name="llama-3.3-70b-versatile",
        api_key="key",
        generation_kwargs={},
        base_url="https://api.groq.com/openai/v1",
        provider_name="groq",
        fallback_llm=fallback,
        auth_cooldown_seconds=120,
    )

    first = llm("prompt one")
    second = llm("prompt two")

    assert first == [{"generated_text": "fallback"}]
    assert second == [{"generated_text": "fallback"}]
    assert fallback_calls == ["prompt one", "prompt two"]
    # Second call should skip HTTP request while cooldown is active
    assert _FakeClient.post_calls == 1


def test_rate_limit_429_does_not_disable_provider(monkeypatch):
    _FakeClient.responses = [
        _make_response(429, {"error": {"message": "rate limit"}}),
        _make_response(429, {"error": {"message": "rate limit"}}),
    ]
    _FakeClient.post_calls = 0
    monkeypatch.setattr("tools.openai_compatible_llm.httpx.Client", _FakeClient)

    fallback_calls: list[str] = []

    def fallback(prompt: str):
        fallback_calls.append(prompt)
        return [{"generated_text": "fallback"}]

    llm = OpenAICompatibleLLM(
        model_name="llama-3.3-70b-versatile",
        api_key="key",
        generation_kwargs={},
        base_url="https://api.groq.com/openai/v1",
        provider_name="groq",
        fallback_llm=fallback,
        auth_cooldown_seconds=120,
    )

    llm("prompt one")
    llm("prompt two")

    # 429 should continue trying provider on later calls (no cooldown disable)
    assert _FakeClient.post_calls == 2
    assert fallback_calls == ["prompt one", "prompt two"]


def test_auth_failure_without_fallback_raises_clear_error(monkeypatch):
    _FakeClient.responses = [
        _make_response(403, {"error": {"message": "forbidden"}}),
    ]
    _FakeClient.post_calls = 0
    monkeypatch.setattr("tools.openai_compatible_llm.httpx.Client", _FakeClient)

    llm = OpenAICompatibleLLM(
        model_name="llama-3.3-70b-versatile",
        api_key="key",
        generation_kwargs={},
        base_url="https://api.groq.com/openai/v1",
        provider_name="groq",
        auth_cooldown_seconds=120,
    )

    with pytest.raises(RuntimeError, match="authentication/permission failure"):
        llm("prompt")
