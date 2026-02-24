"""
slack_debug.py - Quick Slack diagnostic tool
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("SLACK_BOT_TOKEN")
channel_id = os.getenv("SLACK_CHANNEL_ID")

print(f"Token starts with: {token[:15]}..." if token else "NO TOKEN!")
print(f"Channel ID: {channel_id}")
print()

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

client = WebClient(token=token)

# 1. Test auth
try:
    auth = client.auth_test()
    print(f"[OK] Bot Name:      {auth['user']}")
    print(f"[OK] Team:          {auth['team']}")
    print(f"[OK] Bot User ID:   {auth['user_id']}")
    print()
except SlackApiError as e:
    print(f"[FAIL] Auth test failed: {e.response['error']}")
    sys.exit(1)

# 2. List channels bot can see
print("Channels bot has access to:")
try:
    result = client.conversations_list(types="public_channel,private_channel", limit=20)
    channels = result.get("channels", [])
    if not channels:
        print("  (none found - bot may not be in any channel)")
    for ch in channels:
        marker = "  <-- TARGET" if ch['id'] == channel_id else ""
        print(f"  {ch['id']}  #{ch['name']}{marker}")
except SlackApiError as e:
    print(f"  Error listing channels: {e.response['error']}")

# 3. Try to join the target channel (if public)
print(f"\nTrying to join channel {channel_id}...")
try:
    client.conversations_join(channel=channel_id)
    print("[OK] Bot joined channel!")
    
    # Now try reading messages
    result = client.conversations_history(channel=channel_id, limit=5)
    msgs = result.get("messages", [])
    print(f"[OK] Messages fetched: {len(msgs)}")
    for m in msgs[:2]:
        print(f"   - {m.get('text','')[:80]}")
except SlackApiError as e:
    print(f"[INFO] Join result: {e.response['error']}")
