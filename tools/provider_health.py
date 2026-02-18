"""Provider health checks for external LLM APIs."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import httpx

from api.config import GROQ_API_KEY, GROQ_MODEL

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _key_suffix(api_key: str) -> str:
    if not api_key:
        return ""
    return api_key[-6:]


def _status_failure_type(status_code: int) -> str:
    if status_code == 401:
        return "invalid_credentials"
    if status_code == 403:
        return "permission_denied"
    if status_code == 404:
        return "model_not_found"
    if status_code == 429:
        return "rate_limited"
    if 500 <= status_code:
        return "provider_error"
    return "http_error"


def _response_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = {}

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()[:300]
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()[:300]
    return response.text.strip()[:300]


def validate_groq(
    api_key: str,
    model_name: str,
    timeout_s: int = 8,
) -> Dict[str, Any]:
    """Validate Groq credentials and model accessibility with direct API calls."""
    result: Dict[str, Any] = {
        "ok": False,
        "http_status": None,
        "failure_type": None,
        "message": "",
        "model_visible": False,
        "chat_callable": False,
    }
    if not api_key:
        result["failure_type"] = "missing_api_key"
        result["message"] = "GROQ_API_KEY is not configured."
        return result
    if not model_name:
        result["failure_type"] = "missing_model_name"
        result["message"] = "AUTO_ANALYST_GROQ_MODEL is not configured."
        return result

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=timeout_s) as client:
            models_resp = client.get(f"{GROQ_BASE_URL}/models", headers=headers)
            result["http_status"] = models_resp.status_code
            if models_resp.status_code >= 400:
                result["failure_type"] = _status_failure_type(models_resp.status_code)
                result["message"] = (
                    f"GET /models failed: {_response_message(models_resp)}"
                )
                return result

            data = models_resp.json()
            models = data.get("data", []) if isinstance(data, dict) else []
            model_ids = {
                item.get("id", "")
                for item in models
                if isinstance(item, dict) and isinstance(item.get("id", ""), str)
            }
            result["model_visible"] = model_name in model_ids

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0,
            }
            chat_resp = client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            result["http_status"] = chat_resp.status_code
            if chat_resp.status_code >= 400:
                result["failure_type"] = _status_failure_type(chat_resp.status_code)
                result["message"] = (
                    f"POST /chat/completions failed: {_response_message(chat_resp)}"
                )
                return result

            result["chat_callable"] = True
            result["ok"] = True
            result["failure_type"] = None
            result["message"] = "Groq key and model validation passed."
            return result
    except httpx.TimeoutException as exc:
        result["failure_type"] = "timeout"
        result["message"] = str(exc)[:300] or "Groq validation timed out."
        return result
    except httpx.RequestError as exc:
        result["failure_type"] = "network_error"
        result["message"] = str(exc)[:300] or "Network error during Groq validation."
        return result
    except Exception as exc:  # pragma: no cover - defensive
        result["failure_type"] = "unexpected_error"
        result["message"] = str(exc)[:300]
        return result


def run_cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate external provider credentials/model permissions."
    )
    parser.add_argument(
        "--provider",
        choices=["groq"],
        default="groq",
        help="Provider to validate.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=8,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--model",
        default=GROQ_MODEL,
        help="Model name to validate.",
    )
    args = parser.parse_args(argv)

    if args.provider != "groq":
        print(json.dumps({"ok": False, "message": "Unsupported provider"}))
        return 2

    result = validate_groq(
        api_key=GROQ_API_KEY,
        model_name=args.model,
        timeout_s=max(1, args.timeout),
    )
    output = {
        "provider": "groq",
        "model": args.model,
        "key_suffix": _key_suffix(GROQ_API_KEY),
        **result,
    }
    print(json.dumps(output))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
