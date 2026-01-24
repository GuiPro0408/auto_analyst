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

    class _NoiseFilter(logging.Filter):
        """Filter out noisy third-party loggers that clutter logs."""

        NOISY_LOGGERS = (
            "watchdog.observers.inotify_buffer",
            "watchdog.observers",
            "urllib3.connectionpool",
            "httpcore",
            "httpx",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            # Suppress DEBUG logs from noisy third-party libraries
            if record.name.startswith(self.NOISY_LOGGERS):
                return record.levelno >= logging.WARNING
            return True

    redact_filter = _RedactFilter()
    noise_filter = _NoiseFilter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(redact_filter)
    handler.addFilter(noise_filter)
    handlers = [handler]

    if config.LOG_FILE_PATH:
        file_handler = logging.FileHandler(config.LOG_FILE_PATH)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redact_filter)
        file_handler.addFilter(noise_filter)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)


def configure_logging(
    level: str | None = None,
    log_format: str | None = None,
    log_file: str | None = None,
    redact_queries: bool | None = None,
) -> None:
    """Explicitly configure logging settings.

    This function allows programmatic control over logging configuration,
    overriding environment variable settings when provided.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to config.LOG_LEVEL.
        log_format: Output format ('plain' or 'json'). Defaults to config.LOG_FORMAT.
        log_file: Path to log file. Defaults to config.LOG_FILE_PATH.
        redact_queries: Whether to redact sensitive fields. Defaults to config.LOG_REDACT_QUERIES.
    """
    # Clear existing handlers to allow reconfiguration
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Apply overrides or use config defaults
    effective_level = getattr(
        logging, (level or config.LOG_LEVEL).upper(), logging.INFO
    )
    effective_format = (log_format or config.LOG_FORMAT).lower()
    effective_file = log_file if log_file is not None else config.LOG_FILE_PATH
    effective_redact = (
        redact_queries if redact_queries is not None else config.LOG_REDACT_QUERIES
    )

    fmt = "%(asctime)s %(levelname)s [%(name)s] [run=%(run_id)s] %(message)s"
    formatter = (
        _JsonFormatter() if effective_format == "json" else _DefaultFormatter(fmt=fmt)
    )

    class _RedactFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
            if effective_redact:
                for field in ("query", "task_queries", "urls"):
                    if hasattr(record, field):
                        setattr(record, field, "[REDACTED]")
            return True

    class _NoiseFilter(logging.Filter):
        """Filter out noisy third-party loggers that clutter logs."""

        NOISY_LOGGERS = (
            "watchdog.observers.inotify_buffer",
            "watchdog.observers",
            "urllib3.connectionpool",
            "httpcore",
            "httpx",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            if record.name.startswith(self.NOISY_LOGGERS):
                return record.levelno >= logging.WARNING
            return True

    redact_filter = _RedactFilter()
    noise_filter = _NoiseFilter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(redact_filter)
    handler.addFilter(noise_filter)
    handlers = [handler]

    if effective_file:
        file_handler = logging.FileHandler(effective_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redact_filter)
        file_handler.addFilter(noise_filter)
        handlers.append(file_handler)

    logging.basicConfig(level=effective_level, handlers=handlers, force=True)


def get_logger(name: str, run_id: str | None = None) -> Logger | LoggerAdapter:
    """Return a logger configured with optional run correlation id."""
    _configure_root()
    base = logging.getLogger(name)
    if run_id:
        return LoggerAdapter(base, extra={"run_id": run_id})
    return base
