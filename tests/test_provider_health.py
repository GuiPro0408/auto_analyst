"""Tests for provider health validation utilities."""

import httpx

from tools import provider_health


def _make_response(method: str, status_code: int, url: str, payload: dict) -> httpx.Response:
    request = httpx.Request(method, url)
    return httpx.Response(status_code, request=request, json=payload)


class _FakeClient:
    get_response: httpx.Response | Exception | None = None
    post_response: httpx.Response | Exception | None = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, *args, **kwargs):
        if isinstance(_FakeClient.get_response, Exception):
            raise _FakeClient.get_response
        return _FakeClient.get_response

    def post(self, *args, **kwargs):
        if isinstance(_FakeClient.post_response, Exception):
            raise _FakeClient.post_response
        return _FakeClient.post_response


def test_validate_groq_success(monkeypatch):
    _FakeClient.get_response = _make_response(
        "GET",
        200,
        "https://api.groq.com/openai/v1/models",
        {"data": [{"id": "llama-3.3-70b-versatile"}]},
    )
    _FakeClient.post_response = _make_response(
        "POST",
        200,
        "https://api.groq.com/openai/v1/chat/completions",
        {"choices": [{"message": {"content": "pong"}}]},
    )
    monkeypatch.setattr("tools.provider_health.httpx.Client", _FakeClient)

    result = provider_health.validate_groq(
        api_key="gsk_test_123456",
        model_name="llama-3.3-70b-versatile",
        timeout_s=2,
    )

    assert result["ok"] is True
    assert result["model_visible"] is True
    assert result["chat_callable"] is True
    assert result["failure_type"] is None


def test_validate_groq_invalid_key(monkeypatch):
    _FakeClient.get_response = _make_response(
        "GET",
        401,
        "https://api.groq.com/openai/v1/models",
        {"error": {"message": "unauthorized"}},
    )
    _FakeClient.post_response = None
    monkeypatch.setattr("tools.provider_health.httpx.Client", _FakeClient)

    result = provider_health.validate_groq(
        api_key="bad-key",
        model_name="llama-3.3-70b-versatile",
        timeout_s=2,
    )

    assert result["ok"] is False
    assert result["failure_type"] == "invalid_credentials"
    assert result["http_status"] == 401


def test_validate_groq_permission_or_model_error(monkeypatch):
    _FakeClient.get_response = _make_response(
        "GET",
        200,
        "https://api.groq.com/openai/v1/models",
        {"data": [{"id": "another-model"}]},
    )
    _FakeClient.post_response = _make_response(
        "POST",
        404,
        "https://api.groq.com/openai/v1/chat/completions",
        {"error": {"message": "model not found"}},
    )
    monkeypatch.setattr("tools.provider_health.httpx.Client", _FakeClient)

    result = provider_health.validate_groq(
        api_key="gsk_test_123456",
        model_name="llama-3.3-70b-versatile",
        timeout_s=2,
    )

    assert result["ok"] is False
    assert result["failure_type"] == "model_not_found"
    assert result["model_visible"] is False
    assert result["chat_callable"] is False


def test_validate_groq_timeout(monkeypatch):
    _FakeClient.get_response = httpx.TimeoutException("timed out")
    _FakeClient.post_response = None
    monkeypatch.setattr("tools.provider_health.httpx.Client", _FakeClient)

    result = provider_health.validate_groq(
        api_key="gsk_test_123456",
        model_name="llama-3.3-70b-versatile",
        timeout_s=1,
    )

    assert result["ok"] is False
    assert result["failure_type"] == "timeout"


def test_run_cli_redacts_full_key(monkeypatch, capsys):
    monkeypatch.setattr(provider_health, "GROQ_API_KEY", "gsk_secret_key_abcdef123456")
    monkeypatch.setattr(provider_health, "GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(
        provider_health,
        "validate_groq",
        lambda api_key, model_name, timeout_s: {
            "ok": False,
            "http_status": 403,
            "failure_type": "permission_denied",
            "message": "forbidden",
            "model_visible": False,
            "chat_callable": False,
        },
    )

    exit_code = provider_health.run_cli(
        ["--provider", "groq", "--timeout", "1", "--model", "llama-3.3-70b-versatile"]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "gsk_secret_key_abcdef123456" not in output
    assert '"key_suffix": "123456"' in output
