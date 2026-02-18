"""Tests for Chainlit configuration defaults."""

from pathlib import Path
import tomllib


def test_chainlit_ui_language_forced_to_en_us():
    config_path = Path(__file__).resolve().parent.parent / ".chainlit" / "config.toml"
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    assert payload["UI"]["language"] == "en-US"
