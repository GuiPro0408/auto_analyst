import pytest

from tools import models
from tools.models import GeminiLLM, HuggingFaceInferenceLLM


class DummyGenAI:
    configured_api_key = None
    last_prompt = None
    last_config = None

    @classmethod
    def configure(cls, api_key=None, **_kwargs):
        cls.configured_api_key = api_key

    class GenerativeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt, generation_config=None):
            DummyGenAI.last_prompt = prompt
            DummyGenAI.last_config = generation_config

            class _Response:
                text = "Answer: hello world"
                candidates = []

            return _Response()


def reset_dummy_state():
    DummyGenAI.configured_api_key = None
    DummyGenAI.last_prompt = None
    DummyGenAI.last_config = None


def test_gemini_llm_maps_generation_kwargs():
    reset_dummy_state()
    llm = GeminiLLM(
        model_name="gemini-1.5-flash",
        api_key="test-key",
        generation_kwargs={"max_new_tokens": 42, "temperature": 0.1},
        genai_module=DummyGenAI,
    )

    result = llm("prompt here")

    assert result == [{"generated_text": "Answer: hello world"}]
    assert DummyGenAI.configured_api_key == "test-key"
    assert DummyGenAI.last_prompt == "prompt here"
    assert DummyGenAI.last_config == {
        "max_output_tokens": 42,
        "temperature": 0.1,
    }


def test_gemini_llm_missing_api_key_raises():
    with pytest.raises(ValueError):
        GeminiLLM(
            model_name="gemini-1.5-flash",
            api_key="",
            generation_kwargs={},
            genai_module=DummyGenAI,
        )


def test_load_llm_uses_gemini_backend(monkeypatch):
    models.load_llm.cache_clear()
    monkeypatch.setattr(models, "LLM_BACKEND", "gemini")
    monkeypatch.setattr(models, "GEMINI_API_KEY", "secret")

    # Ensure API key list is recomputed for this test
    monkeypatch.setattr(models, "GEMINI_API_KEYS", [])

    captured = {}

    class FakeGemini:
        def __init__(self, model_name, generation_kwargs, api_keys=None, **_kwargs):
            captured["model_name"] = model_name
            captured["api_keys"] = api_keys
            captured["generation_kwargs"] = generation_kwargs

    monkeypatch.setattr(models, "GeminiLLM", FakeGemini)

    try:
        llm = models.load_llm(model_name="custom-gemini")

        assert captured["model_name"] == "custom-gemini"
        assert captured["api_keys"] == ["secret"]
        assert captured["generation_kwargs"] == models.GENERATION_KWARGS
        assert isinstance(llm, FakeGemini)
    finally:
        models.load_llm.cache_clear()


def test_hf_inference_llm_requires_token():
    with pytest.raises(ValueError):
        HuggingFaceInferenceLLM(
            model_name="test-model",
            api_token="",
            generation_kwargs={},
            client=None,
        )


def test_load_llm_uses_hf_backend(monkeypatch):
    models.load_llm.cache_clear()
    monkeypatch.setattr(models, "LLM_BACKEND", "huggingface")
    monkeypatch.setattr(models, "HUGGINGFACE_API_TOKEN", "token")

    captured = {}

    class FakeHF:
        def __init__(self, model_name, api_token, generation_kwargs, **_kwargs):
            captured["model_name"] = model_name
            captured["api_token"] = api_token
            captured["generation_kwargs"] = generation_kwargs

    monkeypatch.setattr(models, "HuggingFaceInferenceLLM", FakeHF)

    try:
        llm = models.load_llm(model_name="custom-hf")

        assert isinstance(llm, FakeHF)
        assert captured["model_name"] == "custom-hf"
        assert captured["api_token"] == "token"
        assert captured["generation_kwargs"] == models.GENERATION_KWARGS
    finally:
        models.load_llm.cache_clear()


def test_load_llm_falls_back_to_hf_when_gemini_dependency_missing(monkeypatch):
    models.load_llm.cache_clear()
    monkeypatch.setattr(models, "LLM_BACKEND", "gemini")
    monkeypatch.setattr(models, "GEMINI_API_KEY", "secret")
    monkeypatch.setattr(models, "HUGGINGFACE_API_TOKEN", "hf-token")
    monkeypatch.setattr(models, "HUGGINGFACE_INFERENCE_MODEL", "hf/default")

    monkeypatch.setattr(models, "GEMINI_API_KEYS", [])

    class BrokenGemini:
        def __init__(self, *args, **kwargs):  # noqa: D401
            raise ImportError("missing google-generativeai")

    class FakeHF:
        def __init__(self, model_name, api_token, generation_kwargs, **_kwargs):
            self.model_name = model_name
            self.api_token = api_token
            self.generation_kwargs = generation_kwargs

    monkeypatch.setattr(models, "GeminiLLM", BrokenGemini)
    monkeypatch.setattr(models, "HuggingFaceInferenceLLM", FakeHF)

    try:
        llm = models.load_llm(model_name="custom")
        assert isinstance(llm, FakeHF)
        assert llm.model_name == "custom"
        assert llm.api_token == "hf-token"
        assert llm.generation_kwargs == models.GENERATION_KWARGS
    finally:
        models.load_llm.cache_clear()


def test_gemini_llm_rotates_api_keys_on_rate_limit():
    class RateLimitGenAI:
        configured_keys = []
        calls = 0

        @classmethod
        def configure(cls, api_key=None, **_kwargs):
            cls.configured_keys.append(api_key)

        class GenerativeModel:
            def __init__(self, model_name):
                self.model_name = model_name

            def generate_content(self, prompt, generation_config=None):
                RateLimitGenAI.calls += 1
                if RateLimitGenAI.calls == 1:
                    raise Exception("429 resource exhausted")

                class _Response:
                    text = "Recovered after rotation"
                    candidates = []

                return _Response()

    llm = GeminiLLM(
        model_name="gemini-1.5-flash",
        generation_kwargs={},
        api_keys=["key-one", "key-two"],
        genai_module=RateLimitGenAI,
    )

    result = llm("hello")

    assert result == [{"generated_text": "Recovered after rotation"}]
    assert RateLimitGenAI.configured_keys == ["key-one", "key-two"]
    assert RateLimitGenAI.calls == 2


def test_gemini_llm_uses_fallback_when_all_keys_rate_limited():
    class AlwaysRateLimitGenAI:
        configured_keys = []
        calls = 0

        @classmethod
        def configure(cls, api_key=None, **_kwargs):
            cls.configured_keys.append(api_key)

        class GenerativeModel:
            def __init__(self, model_name):
                self.model_name = model_name

            def generate_content(self, prompt, generation_config=None):
                AlwaysRateLimitGenAI.calls += 1
                raise Exception("429 quota exceeded")

    fallback_calls = []

    def fallback_fn(prompt: str):
        fallback_calls.append(prompt)

        class _Response:
            text = "fallback-success"
            candidates = []

        return [{"generated_text": "fallback-success"}]

    llm = GeminiLLM(
        model_name="gemini-1.5-flash",
        generation_kwargs={},
        api_keys=["only-key"],
        genai_module=AlwaysRateLimitGenAI,
        fallback_llm=fallback_fn,
    )

    result = llm("question")

    assert result == [{"generated_text": "fallback-success"}]
    assert fallback_calls == ["question"]
    assert AlwaysRateLimitGenAI.configured_keys == ["only-key"]


def test_hf_inference_llm_retries_on_failure():
    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def chat_completion(self, messages=None, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("500 internal error")

            class _Msg:
                content = "ok"

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    client = FlakyClient()
    llm = HuggingFaceInferenceLLM(
        model_name="test-model",
        api_token="token",
        generation_kwargs={},
        client=client,
        retries=2,
        backoff_seconds=0,
    )

    result = llm("hello")

    assert client.calls == 2
    assert result == [{"generated_text": "ok"}]