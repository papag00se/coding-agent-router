from __future__ import annotations

import json
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("TEST_SERVER_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
prompt = (Path(__file__).resolve().parent / "prompts" / "01_local_reasoning.txt").read_text(encoding="utf-8")

payload = {
    "model": "router-gateway",
    "max_tokens": 1200,
    "system": "You are a helpful coding and reasoning assistant.",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "metadata": {"source": "anthropic_gateway_test.py"}
}

resp = requests.post(f"{base_url}/v1/messages", json=payload, timeout=1800)
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))
