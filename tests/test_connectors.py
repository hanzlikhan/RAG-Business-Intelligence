"""
tests/test_connectors.py
──────────────────────────
Unit tests for connectors.py multi-source ingestion pipeline.
Covers PII anonymization, CRM loading, auth fallbacks, and source tagging.
"""
import asyncio
import json
import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from connectors import (
    anonymize_text,
    anonymize_documents,
    load_crm_documents,
    fetch_gmail_messages_async,
    fetch_slack_messages_async,
    ingest_all_sources_async,
)

# ── PII Anonymizer Tests ────────────────────────────────────────────────────────

def test_pii_anonymizer_email():
    """Emails should be replaced with [ANONYMIZED_EMAIL]."""
    text = "Contact john.doe@company.com for details."
    result = anonymize_text(text)
    assert "[ANONYMIZED_EMAIL]" in result
    assert "john.doe@company.com" not in result

def test_pii_anonymizer_phone():
    """Phone numbers should be replaced with [ANONYMIZED_PHONE]."""
    text = "Call us at +1-800-555-0199 or (202) 555-0143."
    result = anonymize_text(text)
    assert "[ANONYMIZED_PHONE]" in result

def test_pii_anonymizer_ssn():
    """SSN patterns should be replaced with [ANONYMIZED_SSN]."""
    text = "Employee SSN: 123-45-6789"
    result = anonymize_text(text)
    assert "[ANONYMIZED_SSN]" in result
    assert "123-45-6789" not in result

def test_anonymize_documents():
    """anonymize_documents should strip PII from page_content of each doc."""
    docs = [
        Document(page_content="Email me at secret@corp.org", metadata={"source": "test"}),
        Document(page_content="No PII here.", metadata={"source": "test"}),
    ]
    result = anonymize_documents(docs)
    assert "[ANONYMIZED_EMAIL]" in result[0].page_content
    assert "secret@corp.org" not in result[0].page_content
    assert result[1].page_content == "No PII here."

# ── CRM Connector Tests ─────────────────────────────────────────────────────────

def test_crm_loader_json(tmp_path):
    """CRM JSON loader should return correct Document count with source tag."""
    crm_data = [
        {"name": "Acme Corp", "email": "acme@example.com", "notes": "Big client"},
        {"name": "Beta Ltd", "email": "beta@example.com", "notes": "Prospect"},
    ]
    crm_file = tmp_path / "crm.json"
    crm_file.write_text(json.dumps(crm_data))

    docs = load_crm_documents(str(crm_file))
    assert len(docs) == 2
    assert all(d.metadata["source"] == "crm" for d in docs)
    # Emails should be anonymized
    assert all("[ANONYMIZED_EMAIL]" in d.page_content for d in docs)

def test_crm_loader_csv(tmp_path):
    """CRM CSV loader should return correct Document count with source tag."""
    crm_file = tmp_path / "crm.csv"
    crm_file.write_text("name,email,notes\nAcme,acme@corp.com,Big deal\nBeta,beta@corp.com,Prospect")

    docs = load_crm_documents(str(crm_file))
    assert len(docs) == 2
    assert all(d.metadata["source"] == "crm" for d in docs)

def test_crm_loader_missing_file():
    """Should return placeholder doc when file not found."""
    docs = load_crm_documents("/nonexistent/path/crm.json")
    assert len(docs) == 1
    assert "[CRM_FILE_ERROR]" in docs[0].page_content
    assert docs[0].metadata["source"] == "crm"

def test_crm_loader_invalid_extension(tmp_path):
    """Should return error doc for unsupported file types."""
    bad_file = tmp_path / "crm.xml"
    bad_file.write_text("<record></record>")
    docs = load_crm_documents(str(bad_file))
    assert len(docs) == 1
    assert "[CRM_LOAD_ERROR]" in docs[0].page_content

# ── Source Tagging Tests ────────────────────────────────────────────────────────

def test_source_tagging_crm(tmp_path):
    """All CRM documents must have 'source' metadata key set to 'crm'."""
    data = [{"name": "Test", "notes": "Hello"}]
    f = tmp_path / "test.json"
    f.write_text(json.dumps(data))
    docs = load_crm_documents(str(f))
    for doc in docs:
        assert "source" in doc.metadata
        assert doc.metadata["source"] == "crm"

# ── Gmail Auth Fallback Tests ───────────────────────────────────────────────────

def test_gmail_auth_fallback_no_credentials():
    """Gmail connector should return a placeholder doc when credentials are missing."""
    result = asyncio.run(fetch_gmail_messages_async(
        token_path="/nonexistent/token.json",
        credentials_path="/nonexistent/credentials.json",
    ))
    assert len(result) >= 1
    assert "[GMAIL_AUTH_ERROR]" in result[0].page_content
    assert result[0].metadata["source"] == "gmail"

# ── Slack Auth Fallback Tests ───────────────────────────────────────────────────

def test_slack_auth_fallback_no_token():
    """Slack connector should return placeholder doc when token not set."""
    # Temporarily remove SLACK_BOT_TOKEN from env
    env_backup = os.environ.pop("SLACK_BOT_TOKEN", None)
    try:
        result = asyncio.run(fetch_slack_messages_async(channel_id="C12345"))
        assert len(result) == 1
        assert "[SLACK_AUTH_ERROR]" in result[0].page_content
        assert result[0].metadata["source"] == "slack"
    finally:
        if env_backup:
            os.environ["SLACK_BOT_TOKEN"] = env_backup

def test_slack_auth_fallback_no_channel():
    """Slack connector should return placeholder doc when channel ID not set."""
    env_backup = os.environ.pop("SLACK_BOT_TOKEN", None)
    try:
        os.environ["SLACK_BOT_TOKEN"] = "fake-token"
        env_channel_backup = os.environ.pop("SLACK_CHANNEL_ID", None)
        try:
            result = asyncio.run(fetch_slack_messages_async(channel_id=None))
            assert len(result) == 1
            assert "[SLACK_AUTH_ERROR]" in result[0].page_content
        finally:
            if env_channel_backup:
                os.environ["SLACK_CHANNEL_ID"] = env_channel_backup
    finally:
        if env_backup:
            os.environ["SLACK_BOT_TOKEN"] = env_backup
        else:
            os.environ.pop("SLACK_BOT_TOKEN", None)

# ── Multi-Source Diversity Tests ────────────────────────────────────────────────

def test_multi_source_diversity(tmp_path):
    """ingest_all_sources_async should return docs from multiple distinct sources."""
    crm_data = [{"name": "Acme", "deal": "Big contract for 500k"}]
    crm_file = tmp_path / "crm.json"
    crm_file.write_text(json.dumps(crm_data))

    # Run with gmail/slack disabled to avoid real API calls in tests
    docs = asyncio.run(ingest_all_sources_async(
        file_paths=None,
        gmail=False,
        slack=False,
        crm_path=str(crm_file),
    ))
    sources = {d.metadata.get("source") for d in docs}
    assert "crm" in sources
    assert len(docs) >= 1
