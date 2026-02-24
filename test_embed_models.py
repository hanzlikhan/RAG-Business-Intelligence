import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

r = requests.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    params={"key": api_key},
    timeout=20
)
if r.status_code == 200:
    print("Available Embedding Models:")
    for m in r.json().get("models", []):
        methods = m.get("supportedGenerationMethods", [])
        if "embedContent" in methods or "batchEmbedContents" in methods:
            print(f"  {m['name']} | methods: {methods}")
else:
    print(f"Error: {r.status_code} - {r.text[:200]}")
