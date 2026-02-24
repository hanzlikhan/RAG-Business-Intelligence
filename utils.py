"""
utils.py
────────
Shared utilities for AI-BOS:
  - Structured logging at INFO level
  - Tenacity retry decorator (3 attempts, exponential backoff)
"""

import logging
import functools
from typing import Callable, TypeVar, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import time

def timer(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.getLogger(__name__).info("Function %s took %.4f seconds", func.__name__, elapsed)
        return res
    return wrapper

def async_timer(func: Callable) -> Callable:
    """Decorator to measure and log async function execution time."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        res = await func(*args, **kwargs)
        elapsed = time.time() - start
        logging.getLogger(__name__).info("Async Function %s took %.4f seconds", func.__name__, elapsed)
        return res
    return wrapper

# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Create and return a module-level logger with INFO level and a
    human-readable format.

    Args:
        name: Usually ``__name__`` from the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Retry decorator ────────────────────────────────────────────────────────────

_logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def with_retries(func: F) -> F:
    """
    Decorator: retry the wrapped function up to 3 times with exponential
    back-off (2 s → 4 s → 8 s) on any ``Exception``.

    Usage::

        @with_retries
        def call_api() -> str:
            ...

    Args:
        func: The callable to wrap.

    Returns:
        The wrapped callable with retry logic applied.
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def validate_query(query: str) -> str:
    """
    Validate and sanitise a user query string.

    Args:
        query: Raw user input.

    Returns:
        Stripped query string.

    Raises:
        ValueError: If the query is empty or exceeds 4000 characters.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    if len(query) > 4000:
        raise ValueError(f"Query too long ({len(query)} chars). Max is 4000.")
    return query
