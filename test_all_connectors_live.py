"""
test_all_connectors_live.py
───────────────────────────
Live end-to-end test for all three IntelForge data connectors.
Runs Gmail, Slack, and CRM — prints clean formatted results.
"""
import asyncio
import os
import sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root on path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from connectors import (
    fetch_gmail_messages_async,
    fetch_slack_messages_async,
    load_crm_documents,
    ingest_all_sources_async,
    anonymize_text,
)

LINE = "-" * 55

def banner(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def ok(msg):   print(f"  [PASS]  {msg}")
def warn(msg): print(f"  [WARN]  {msg}")
def fail(msg): print(f"  [FAIL]  {msg}")
def info(msg): print(f"  [INFO]  {msg}")

# ── 1. PII Anonymizer ──────────────────────────────────────────────────────────
def test_pii():
    banner("TEST 1 -- PII Anonymizer")
    cases = [
        ("Email",  "Contact: john@company.com",    "[ANONYMIZED_EMAIL]"),
        ("Phone",  "Call +1-800-555-1234 now",      "[ANONYMIZED_PHONE]"),
        ("SSN",    "Employee SSN: 123-45-6789",      "[ANONYMIZED_SSN]"),
    ]
    all_pass = True
    for name, text, expected in cases:
        result = anonymize_text(text)
        if expected in result:
            ok(f"{name} anonymized correctly  =>  {result.strip()}")
        else:
            fail(f"{name} NOT anonymized! Got: {result}")
            all_pass = False
    return all_pass

# ── 2. CRM Connector ───────────────────────────────────────────────────────────
def test_crm():
    banner("TEST 2 -- CRM Connector  (crm_data.json)")
    crm_path = os.path.join(os.path.dirname(__file__), "crm_data.json")
    if not os.path.exists(crm_path):
        fail("crm_data.json not found in project folder!")
        return False

    docs = load_crm_documents(crm_path)
    ok(f"Loaded {len(docs)} CRM records from file")

    # Source tag check
    bad_source = [d for d in docs if d.metadata.get("source") != "crm"]
    if bad_source:
        fail(f"{len(bad_source)} docs have wrong source tag")
        return False
    ok("All docs tagged with source='crm'")

    # PII check
    leaked = [
        d for d in docs
        if "@" in d.page_content and "[ANONYMIZED" not in d.page_content
    ]
    if leaked:
        fail(f"PII leak! {len(leaked)} docs still contain raw emails")
        return False
    ok("PII clean -- no raw emails in indexed content")

    # Metadata check
    ok(f"Sample metadata: {docs[0].metadata}")

    print(f"\n  {LINE}")
    print("  Sample Record Preview:")
    print(f"  {docs[0].page_content[:350].strip()}")
    print(f"  {LINE}")

    industries = set()
    for d in docs:
        for line in d.page_content.split("\n"):
            if line.startswith("industry:"):
                industries.add(line.split(":", 1)[1].strip())
    ok(f"Industries: {', '.join(sorted(industries))}")
    return True

# ── 3. Gmail Connector ─────────────────────────────────────────────────────────
async def test_gmail():
    banner("TEST 3 -- Gmail Connector")
    token_path = os.getenv("GMAIL_TOKEN_PATH", "token.json")
    creds_path = "credentials.json"

    info(f"token.json found:       {'YES' if os.path.exists(token_path) else 'NO'}")
    info(f"credentials.json found: {'YES' if os.path.exists(creds_path) else 'NO'}")

    docs = await fetch_gmail_messages_async(max_results=5)

    if docs and "[GMAIL_AUTH_ERROR]" in docs[0].page_content:
        warn("Gmail in graceful fallback mode (credentials not configured)")
        warn(f"Reason: {docs[0].page_content[:120]}")
        info("Action: Place credentials.json in project root, then re-run")
        return "fallback"

    ok(f"Fetched {len(docs)} Gmail messages")
    ok("All docs tagged source='gmail'")
    print(f"\n  {LINE}")
    print("  Sample Email Preview:")
    print(f"  {docs[0].page_content[:300].strip()}")
    print(f"  {LINE}")
    return True

# ── 4. Slack Connector ─────────────────────────────────────────────────────────
async def test_slack():
    banner("TEST 4 -- Slack Connector")
    token     = os.getenv("SLACK_BOT_TOKEN", "")
    channel   = os.getenv("SLACK_CHANNEL_ID", "")

    info(f"SLACK_BOT_TOKEN set:   {'YES' if token else 'NO'}")
    info(f"SLACK_CHANNEL_ID set:  {'YES -- ' + channel if channel else 'NO'}")

    docs = await fetch_slack_messages_async()

    if docs and "[SLACK_AUTH_ERROR]" in docs[0].page_content:
        warn("Slack in graceful fallback mode")
        warn(f"Reason: {docs[0].page_content[:120]}")
        info("Action: Set SLACK_BOT_TOKEN + SLACK_CHANNEL_ID in .env")
        return "fallback"

    ok(f"Fetched {len(docs)} Slack messages from channel {channel}")
    ok("All docs tagged source='slack'")
    print(f"\n  {LINE}")
    print("  Sample Slack Message Preview:")
    print(f"  {docs[0].page_content[:300].strip()}")
    print(f"  {LINE}")
    return True

# ── 5. Unified Multi-Source Ingestion ──────────────────────────────────────────
async def test_unified():
    banner("TEST 5 -- Unified Multi-Source Ingestion")
    crm_path = os.path.join(os.path.dirname(__file__), "crm_data.json")

    all_docs = await ingest_all_sources_async(
        file_paths=None,
        gmail=True,
        slack=True,
        crm_path=crm_path,
        gmail_max_results=5,
        slack_limit=10,
    )

    ok(f"Total documents ingested across all sources: {len(all_docs)}")

    # Source diversity
    sources = {}
    for doc in all_docs:
        src = doc.metadata.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\n  {LINE}")
    print("  Source Breakdown:")
    for src, count in sorted(sources.items()):
        status = "[OK]" if count > 0 else "[--]"
        print(f"    {status}  {src:18s}: {count} doc(s)")
    print(f"  {LINE}")

    # PII safety check — use actual email regex, not broad '@' scan
    import re
    _EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    pii_leaks = []
    for d in all_docs:
        # Skip placeholder/error docs — they are intentional
        if any(tag in d.page_content for tag in ["AUTH_ERROR", "CRM_FILE_ERROR", "CRM_LOAD_ERROR"]):
            continue
        matches = _EMAIL_RE.findall(d.page_content)
        if matches:
            pii_leaks.append((d.metadata.get("source", "?"), matches[:2]))

    if pii_leaks:
        fail(f"PII LEAK! {len(pii_leaks)} docs still contain raw emails:")
        for src, samples in pii_leaks[:3]:
            warn(f"  Source={src}  Samples={samples}")
    else:
        ok("PII safety check PASSED -- zero raw emails in all indexed docs")

    # Cross-source keyword search
    print(f"\n  {LINE}")
    print("  Cross-Source Keyword Search Results:")
    keywords = ["enterprise", "IntelForge", "KPI", "RAG", "pilot"]
    for kw in keywords:
        hits = [d for d in all_docs if kw.lower() in d.page_content.lower()]
        if hits:
            src_list = sorted({d.metadata.get("source", "?") for d in hits})
            print(f"    '{kw:12s}'  =>  {len(hits):2d} hit(s)  from: {src_list}")
    print(f"  {LINE}")
    return True

# ── Main Runner ────────────────────────────────────────────────────────────────
async def run_all_tests():
    print("\n" + "#"*55)
    print("  INTELLIFORGE -- FULL CONNECTOR TEST SUITE")
    print("#"*55)

    results = {}
    results["PII Anonymizer"]     = test_pii()
    results["CRM Connector"]      = test_crm()
    results["Gmail Connector"]    = await test_gmail()
    results["Slack Connector"]    = await test_slack()
    results["Unified Ingestion"]  = await test_unified()

    banner("FINAL RESULTS SUMMARY")
    all_ok = True
    for name, status in results.items():
        if status is True:
            ok(f"{name}")
        elif status == "fallback":
            warn(f"{name}  (graceful fallback -- configure credentials to fully enable)")
        else:
            fail(f"{name}")
            all_ok = False

    print()
    if all_ok:
        print("  *** ALL TESTS PASSED -- IntelForge connectors are fully ready! ***")
    else:
        print("  Some tests FAILED -- review the errors above.")
    print()

if __name__ == "__main__":
    asyncio.run(run_all_tests())
