"""Structured logging configuration using structlog.

Renders JSON in production (for CloudWatch / Datadog ingestion) and a colourised
console renderer otherwise. Every event is augmented with ``service``,
``timestamp`` and ``level`` fields.
"""
from __future__ import annotations

import logging
import os
import sys

import structlog

SERVICE_NAME = "voiceflow-ml"


def _add_service(_logger, _method, event_dict):
    event_dict.setdefault("service", SERVICE_NAME)
    return event_dict


def configure_logging(env: str | None = None, debug: bool = False) -> None:
    """Configure both stdlib logging and structlog for the application."""
    env = env or os.getenv("ENV", "development").lower()
    is_prod = env in {"production", "prod", "staging"}

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.DEBUG if debug else logging.INFO,
        force=True,
    )

    base_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        _add_service,
    ]

    if is_prod:
        base_processors.append(structlog.processors.JSONRenderer())
    else:
        base_processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=base_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
