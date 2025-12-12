"""Centralised logging setup."""

import json
import logging
import sys
from logging import Logger, LoggerAdapter
from typing import Any, Dict

from api import config


class _DefaultFormatter(logging.Formatter):
    def format(
        self, record: logging.LogRecord
    ) -> str:  # pragma: no cover - thin wrapper
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    def format(
        self, record: logging.LogRecord
    ) -> str:  # pragma: no cover - thin wrapper
        payload: Dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_root() -> None:
    if logging.getLogger().handlers:
        return
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(name)s] [run=%(run_id)s] %(message)s"
    formatter = (
        _JsonFormatter()
        if config.LOG_FORMAT.lower() == "json"
        else _DefaultFormatter(fmt=fmt)
    )

    class _RedactFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
            if config.LOG_REDACT_QUERIES:
                for field in ("query", "task_queries", "urls"):
                    if hasattr(record, field):
                        setattr(record, field, "[REDACTED]")
            return True

    redact_filter = _RedactFilter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(redact_filter)
    handlers = [handler]

    if config.LOG_FILE_PATH:
        file_handler = logging.FileHandler(config.LOG_FILE_PATH)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redact_filter)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)


def get_logger(name: str, run_id: str | None = None) -> Logger | LoggerAdapter:
    """Return a logger configured with optional run correlation id."""
    _configure_root()
    base = logging.getLogger(name)
    if run_id:
        return LoggerAdapter(base, extra={"run_id": run_id})
    return base
