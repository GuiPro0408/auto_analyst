"""Tests for logging run-id context propagation."""

import logging

from api.logging_setup import _DefaultFormatter, bind_run_id


def _make_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )


def test_context_run_id_applies_when_record_has_no_run_id():
    formatter = _DefaultFormatter(fmt="%(levelname)s [run=%(run_id)s] %(message)s")
    with bind_run_id("ctx-123"):
        formatted = formatter.format(_make_record())
    assert "[run=ctx-123]" in formatted


def test_explicit_run_id_overrides_context_run_id():
    formatter = _DefaultFormatter(fmt="%(levelname)s [run=%(run_id)s] %(message)s")
    record = _make_record()
    record.run_id = "explicit-456"
    with bind_run_id("ctx-123"):
        formatted = formatter.format(record)
    assert "[run=explicit-456]" in formatted


def test_context_reset_restores_default_run_id():
    formatter = _DefaultFormatter(fmt="%(levelname)s [run=%(run_id)s] %(message)s")
    with bind_run_id("ctx-123"):
        formatter.format(_make_record())
    formatted = formatter.format(_make_record())
    assert "[run=-]" in formatted
