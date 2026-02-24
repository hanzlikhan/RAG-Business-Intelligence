"""
tests/test_llm.py
─────────────────
Pytest tests for the AI-BOS LLM integration.

Run:
    pytest tests/ -v
"""

import os
import sys
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def llm():
    """Build a shared LLM instance for all tests in this module."""
    from main import build_llm
    return build_llm()


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_response_not_empty(llm) -> None:
    """
    GIVEN a valid query
    WHEN the LLM is called directly
    THEN the response must be non-empty.
    """
    from utils import validate_query

    query = validate_query("What is RAG?")
    # Direct invoke call to test that the LLM is responsive
    response = llm.invoke(query)

    assert hasattr(response, 'content'), "Response should have content attribute"
    assert len(response.content) > 0, "Response content must not be empty"


def test_validate_query_empty() -> None:
    """
    GIVEN an empty string
    WHEN validate_query is called
    THEN it should raise ValueError.
    """
    from utils import validate_query

    with pytest.raises(ValueError, match="empty"):
        validate_query("   ")


def test_validate_query_too_long() -> None:
    """
    GIVEN a query longer than 4000 characters
    WHEN validate_query is called
    THEN it should raise ValueError.
    """
    from utils import validate_query

    with pytest.raises(ValueError, match="too long"):
        validate_query("x" * 4001)


def test_validate_query_valid() -> None:
    """
    GIVEN a normal query string
    WHEN validate_query is called
    THEN it should return the stripped string.
    """
    from utils import validate_query

    result = validate_query("  Hello  ")
    assert result == "Hello"


def test_build_llm_without_key(monkeypatch) -> None:
    """
    GIVEN no GOOGLE_API_KEY in environment
    WHEN build_llm is called
    THEN it should raise EnvironmentError.
    """
    from main import build_llm

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
        build_llm()
