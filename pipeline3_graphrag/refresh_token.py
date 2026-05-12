"""
Fetch a fresh TigerGraph bearer token and save it to tg_token.txt.
Run this once at startup, or whenever the token expires.

Usage:
    python pipeline3_graphrag/refresh_token.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline3_graphrag.pipeline import fetch_token, TG_TOKEN_FILE

if __name__ == "__main__":
    print(f"Fetching token from TigerGraph ...")
    try:
        token = fetch_token()
        print(f"Token saved to {TG_TOKEN_FILE}")
        print(f"Preview: {token[:20]}...")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
