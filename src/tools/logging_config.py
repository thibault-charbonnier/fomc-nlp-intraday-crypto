"""Modern console logging configuration using rich.

Provides get_logger(name) to obtain a configured logger. Uses LOG_LEVEL env var
to set the level (default INFO). Integrates RichHandler for pretty console output
and includes a helper function to enable logging for other libraries.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from rich.logging import RichHandler  # type: ignore
    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when rich isn't installed
    import logging as _logging

    class RichHandler(_logging.StreamHandler):
        """Lightweight fallback handler when rich isn't available.

        It behaves like a normal StreamHandler so the rest of the code can
        continue to call RichHandler(...).
        """

        def __init__(self, *args, **kwargs):
            super().__init__()

    _RICH_AVAILABLE = False


def _get_log_level() -> int:
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def configure_root_logger(level: Optional[int] = None) -> None:
    """Configure the root logger with RichHandler.

    This is idempotent — calling multiple times won't duplicate handlers.
    """
    if level is None:
        level = _get_log_level()

    root = logging.getLogger()
    # Avoid adding multiple RichHandlers if already present
    if any(isinstance(h, RichHandler) for h in root.handlers):
        root.setLevel(level)
        return

    fmt = "%(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if _RICH_AVAILABLE:
        handler = RichHandler(rich_tracebacks=True)
    else:
        handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler.setFormatter(formatter)

    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured to use Rich for console output.

    Example:
        logger = get_logger(__name__)
        logger.info("Starting pipeline")
    """
    configure_root_logger()
    return logging.getLogger(name)


def enable_external_library_logging(name: str, level: Optional[int] = None) -> None:
    """Enable logging for an external library (like requests) at the given level."""
    if level is None:
        level = _get_log_level()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # propagate to root which is handled by RichHandler
    logger.propagate = True
