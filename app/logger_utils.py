"""Utility module for logging, validation, and safe execution.."""

import logging
from contextlib import contextmanager
from typing import Generator, Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistent logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


@contextmanager
def safe_execution(logger: logging.Logger, context: str = "") -> Generator[None, None, None]:
    """Context manager for standardized exception handling."""
    try:
        yield
    except Exception as e:
        msg = f"[{context}] {e}" if context else str(e)
        logger.error(msg)
        raise


def validate_non_empty(value: Optional[str], field: str) -> str:
    """Ensure a string field is non-empty."""
    if not value or not value.strip():
        raise ValueError(f"{field} cannot be empty.")
    return value.strip()
