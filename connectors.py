"""
connectors.py
─────────────
Multi-source data connectors for the IntelForge RAG pipeline.
Supports: Gmail (OAuth2), Slack (Bot Token), CRM (JSON/CSV).

All connectors:
  - Return LangChain Document objects with source metadata tags.
  - Apply PII anonymization (emails, phones, SSN-patterns) before returning.
  - Handle auth errors gracefully with placeholder documents and clear logging.
"""

import asyncio
import csv
import json
import os
import re
import time
from typing import List, Optional

from langchain_core.documents import Document

from utils import get_logger

logger = get_logger(__name__)

# ── PII Anonymizer ─────────────────────────────────────────────────────────────

# Patterns for common PII
_EMAIL_PATTERN    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_MAILTO_PATTERN   = re.compile(r"mailto:[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_EMAIL_PATTERN = re.compile(r"(?:email|mail|to)=[a-zA-Z0-9._%+\-]+(?:%40|@)[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_PATTERN    = re.compile(r"\b(\+?\d[\d\s\-().]{7,14}\d)\b")
_SSN_PATTERN      = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

def anonymize_text(text: str) -> str:
    """
    Scrub PII from a block of text.
    Applies in order:
      1. URL-encoded emails (email=user%40domain.com) 
      2. Mailto links (mailto:user@domain.com)
      3. Standard emails
      4. SSN (before phone to prevent overlap)
      5. Phone numbers
    """
    text = _URL_EMAIL_PATTERN.sub("[ANONYMIZED_EMAIL_PARAM]", text)
    text = _MAILTO_PATTERN.sub("[ANONYMIZED_MAILTO]", text)
    text = _EMAIL_PATTERN.sub("[ANONYMIZED_EMAIL]", text)
    text = _SSN_PATTERN.sub("[ANONYMIZED_SSN]", text)   # SSN before phone!
    text = _PHONE_PATTERN.sub("[ANONYMIZED_PHONE]", text)
    return text

def anonymize_documents(docs: List[Document]) -> List[Document]:
    """Apply PII anonymization to a list of documents in-place."""
    for doc in docs:
        doc.page_content = anonymize_text(doc.page_content)
    return docs

# ── Gmail Connector ────────────────────────────────────────────────────────────

async def fetch_gmail_messages_async(
    max_results: int = 20,
    token_path: Optional[str] = None,
    credentials_path: str = "credentials.json",
) -> List[Document]:
    """
    Async fetch of Gmail messages using Google API Python Client with OAuth2.

    Reads from GMAIL_TOKEN_PATH env var (defaults to 'token.json').
    On any authentication or API failure, returns a placeholder document
    and logs the error — no crash propagated.

    Returns:
        List[Document] tagged with source='gmail' and sender/subject metadata.
    """
    token_path = token_path or os.getenv("GMAIL_TOKEN_PATH", "token.json")
    docs: List[Document] = []

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        import base64

        SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

        # Load or refresh credentials
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif os.path.exists(credentials_path):
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
                with open(token_path, "w") as token:
                    token.write(creds.to_json())
            else:
                raise FileNotFoundError(
                    f"Gmail credentials not found at '{credentials_path}'. "
                    "Download from Google Cloud Console and place in project root."
                )

        # Offload blocking API calls to thread pool
        loop = asyncio.get_event_loop()

        def _fetch_sync():
            service = build("gmail", "v1", credentials=creds)
            result = service.users().messages().list(
                userId="me", maxResults=max_results, labelIds=["INBOX"]
            ).execute()
            messages = result.get("messages", [])
            fetched = []
            for msg_ref in messages:
                raw = service.users().messages().get(
                    userId="me", id=msg_ref["id"], format="full"
                ).execute()
                headers = {h["name"]: h["value"] for h in raw["payload"].get("headers", [])}
                subject = headers.get("Subject", "(no subject)")
                sender = headers.get("From", "unknown")
                # Extract body
                body_text = ""
                payload = raw.get("payload", {})
                if "parts" in payload:
                    for part in payload["parts"]:
                        if part.get("mimeType") == "text/plain":
                            data = part["body"].get("data", "")
                            body_text = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
                            break
                elif "body" in payload and payload["body"].get("data"):
                    body_text = base64.urlsafe_b64decode(
                        payload["body"]["data"] + "=="
                    ).decode("utf-8", errors="ignore")

                fetched.append(Document(
                    page_content=f"Subject: {subject}\n\n{body_text}",
                    metadata={
                        "source": "gmail",
                        "sender": sender,
                        "subject": subject,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                ))
            return fetched

        docs = await loop.run_in_executor(None, _fetch_sync)
        logger.info("Gmail: fetched %d messages.", len(docs))

    except FileNotFoundError as e:
        logger.warning("Gmail auth skipped: %s", e)
        docs = [Document(
            page_content="[GMAIL_AUTH_ERROR] Gmail credentials not configured.",
            metadata={"source": "gmail", "error": str(e), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        )]
    except Exception as e:
        logger.error("Gmail connector error: %s", e)
        docs = [Document(
            page_content=f"[GMAIL_AUTH_ERROR] {e}",
            metadata={"source": "gmail", "error": str(e), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        )]

    return anonymize_documents(docs)

# ── Slack Connector ────────────────────────────────────────────────────────────

async def fetch_slack_messages_async(
    channel_id: Optional[str] = None,
    limit: int = 50,
) -> List[Document]:
    """
    Async fetch of Slack channel messages using slack-sdk.

    Reads SLACK_BOT_TOKEN from environment. If SLACK_CHANNEL_ID is set in
    .env it will be used as default channel. Falls back to placeholder on
    SlackApiError or missing token.

    Returns:
        List[Document] tagged with source='slack' and channel/user metadata.
    """
    token = os.getenv("SLACK_BOT_TOKEN")
    channel_id = channel_id or os.getenv("SLACK_CHANNEL_ID", "")
    docs: List[Document] = []

    if not token:
        logger.warning("SLACK_BOT_TOKEN not set. Returning placeholder.")
        return [Document(
            page_content="[SLACK_AUTH_ERROR] SLACK_BOT_TOKEN not configured in .env.",
            metadata={"source": "slack", "error": "missing token", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        )]

    if not channel_id:
        logger.warning("SLACK_CHANNEL_ID not set. Returning placeholder.")
        return [Document(
            page_content="[SLACK_AUTH_ERROR] SLACK_CHANNEL_ID not configured.",
            metadata={"source": "slack", "error": "missing channel_id", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        )]

    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError

        loop = asyncio.get_event_loop()

        def _fetch_sync():
            client = WebClient(token=token)
            # Auto-join the channel first (handles 'not_in_channel' error)
            try:
                client.conversations_join(channel=channel_id)
                logger.info("Slack: bot joined channel %s.", channel_id)
            except SlackApiError as join_err:
                err_code = join_err.response.get("error", "")
                if err_code not in ("already_in_channel", "method_not_supported_for_channel_type"):
                    logger.warning("Slack: could not join channel: %s", err_code)
            # Now read history
            response = client.conversations_history(channel=channel_id, limit=limit)
            messages = response.get("messages", [])
            fetched = []
            for msg in messages:
                text = msg.get("text", "")
                if not text.strip():
                    continue
                user = msg.get("user", "unknown")
                ts = msg.get("ts", "")
                fetched.append(Document(
                    page_content=text,
                    metadata={
                        "source": "slack",
                        "channel": channel_id,
                        "user": user,
                        "slack_ts": ts,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                ))
            return fetched

        docs = await loop.run_in_executor(None, _fetch_sync)
        logger.info("Slack: fetched %d messages from channel %s.", len(docs), channel_id)

    except Exception as e:
        logger.error("Slack connector error: %s", e)
        docs = [Document(
            page_content=f"[SLACK_AUTH_ERROR] {e}",
            metadata={"source": "slack", "error": str(e), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        )]

    return anonymize_documents(docs)

# ── CRM Connector ──────────────────────────────────────────────────────────────

def load_crm_documents(file_path: str) -> List[Document]:
    """
    Load CRM data from a JSON or CSV file.

    JSON format expected: list of records, e.g.
      [{"name": "Acme Corp", "email": "acme@example.com", "notes": "..."}]

    CSV format: Standard CSV where each row becomes a Document.

    Returns:
        List[Document] tagged with source='crm' and file path metadata.
    """
    if not os.path.exists(file_path):
        logger.warning("CRM file not found: %s. Returning placeholder.", file_path)
        return [Document(
            page_content=f"[CRM_FILE_ERROR] File not found: {file_path}",
            metadata={"source": "crm", "error": "file_not_found", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        )]

    ext = os.path.splitext(file_path)[1].lower()
    docs: List[Document] = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                records = [records]
            for i, record in enumerate(records):
                text = "\n".join(f"{k}: {v}" for k, v in record.items())
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": "crm",
                        "file": file_path,
                        "record_index": i,
                        "timestamp": timestamp,
                    }
                ))
        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    text = "\n".join(f"{k}: {v}" for k, v in row.items())
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": "crm",
                            "file": file_path,
                            "record_index": i,
                            "timestamp": timestamp,
                        }
                    ))
        else:
            raise ValueError(f"CRM connector supports JSON or CSV only. Got: {ext}")

        logger.info("CRM: loaded %d records from %s.", len(docs), file_path)

    except Exception as e:
        logger.error("CRM connector error for %s: %s", file_path, e)
        docs = [Document(
            page_content=f"[CRM_LOAD_ERROR] {e}",
            metadata={"source": "crm", "error": str(e), "timestamp": timestamp}
        )]

    return anonymize_documents(docs)

# ── Unified Multi-Source Ingester ──────────────────────────────────────────────

async def ingest_all_sources_async(
    file_paths: Optional[List[str]] = None,
    gmail: bool = True,
    slack: bool = True,
    slack_channel_id: Optional[str] = None,
    crm_path: Optional[str] = None,
    gmail_max_results: int = 20,
    slack_limit: int = 50,
) -> List[Document]:
    """
    Master ingestion function. Concurrently fetches from all configured
    sources, applies PII anonymization, and returns a unified list of
    tagged Documents ready for Pinecone indexing.

    Args:
        file_paths:         Local file paths (PDF/TXT/CSV/MD) to ingest.
        gmail:              Enable Gmail connector.
        slack:              Enable Slack connector.
        slack_channel_id:   Slack channel to fetch from.
        crm_path:           Path to CRM JSON or CSV file.
        gmail_max_results:  Max Gmail messages to fetch.
        slack_limit:        Max Slack messages to fetch.

    Returns:
        Combined List[Document] from all sources.
    """
    all_docs: List[Document] = []
    tasks = []

    # 1. File ingestion (local files via rag.py pipeline)
    if file_paths:
        try:
            from rag import ingest_documents_async
            tasks.append(ingest_documents_async(file_paths))
        except Exception as e:
            logger.error("File ingestion task setup failed: %s", e)

    # 2. Gmail
    if gmail:
        tasks.append(fetch_gmail_messages_async(max_results=gmail_max_results))

    # 3. Slack
    if slack:
        tasks.append(fetch_slack_messages_async(
            channel_id=slack_channel_id, limit=slack_limit
        ))

    # Run all async tasks concurrently
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error("Source ingestion error: %s", result)
            elif isinstance(result, list):
                all_docs.extend(result)

    # 4. CRM (synchronous, no async needed)
    if crm_path:
        crm_docs = load_crm_documents(crm_path)
        all_docs.extend(crm_docs)

    logger.info(
        "Multi-source ingestion complete: %d total documents from %d sources.",
        len(all_docs),
        sum([
            1 if file_paths else 0,
            1 if gmail else 0,
            1 if slack else 0,
            1 if crm_path else 0,
        ])
    )

    return all_docs
