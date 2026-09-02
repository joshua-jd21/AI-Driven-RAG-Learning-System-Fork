#!/usr/bin/env python3
"""Quick NVIDIA NIM auth + chat smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from modules.config import (
    NVIDIA_API_KEY,
    NVIDIA_PLANNER_MODEL,
    NVIDIA_REPAIR_MODEL,
)

CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


MODELS = (NVIDIA_PLANNER_MODEL, NVIDIA_REPAIR_MODEL)


def mask_key(key: str) -> str:
    """Return a non-sensitive preview suitable for a terminal smoke test."""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def check(model: str) -> bool:
    try:
        response = requests.post(
            CHAT_URL,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 8,
            },
            timeout=30,
        )
        body = response.text[:160].replace("\n", " ")
        print(f"model={model} status={response.status_code} body={body}")
        return response.ok
    except requests.RequestException as exc:
        print(f"model={model} status=REQUEST_ERROR body={str(exc)[:160]}")
        return False


def main() -> None:
    if not NVIDIA_API_KEY:
        print("NVIDIA_API_KEY missing in .env")
        sys.exit(1)
    print(f"Using key: {mask_key(NVIDIA_API_KEY)}")
    results = [check(model) for model in MODELS]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
