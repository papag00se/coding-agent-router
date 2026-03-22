from __future__ import annotations

import argparse
import json
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_file", help="Path to a .txt prompt file, for example scripts/prompts/01_local_reasoning.txt")
    parser.add_argument("--preferred-backend", default=None)
    parser.add_argument("--base-url", default=os.getenv("TEST_SERVER_BASE_URL", "http://127.0.0.1:8080"))
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    payload = {
        "prompt": prompt,
        "metadata": {"source": "smoke_test.py", "prompt_file": args.prompt_file},
    }
    if args.preferred_backend:
        payload["preferred_backend"] = args.preferred_backend

    response = requests.post(f"{args.base_url.rstrip('/')}/invoke", json=payload, timeout=1800)
    response.raise_for_status()
    body = response.json()

    print("=== ROUTE DECISION ===")
    print(json.dumps(body.get("route_decision", {}), indent=2))
    print("\n=== BACKEND MODEL ===")
    print(body.get("backend_model"))
    print("\n=== OUTPUT ===\n")
    print(body.get("output_text", ""))


if __name__ == "__main__":
    main()
