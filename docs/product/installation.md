# Installation

[Docs Index](../index.md) | [Product Index](./index.md)

## Prereqs

- Python 3.11+
- Ollama on each machine that will host a local model
- Network reachability from the router host to each Ollama host
- Optional: Codex CLI installed locally

## Pull Local Models

Example commands:

```bash
# router box
ollama pull qwen3:8b-q4_K_M

# coding box
ollama pull qwen3-coder:30b-a3b-q4_K_M

# reasoning box
ollama pull qwen3:14b
```

If `qwen3:14b` is too slow or too large on the smaller box, use:

```bash
ollama pull qwen3:8b-q4_K_M
```

## Create a Virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure Environment

```bash
cp .env.example .env
```

Then edit `.env`. At minimum, set:

- `ROUTER_OLLAMA_BASE_URL`
- `CODER_OLLAMA_BASE_URL`
- `REASONER_OLLAMA_BASE_URL`
- model names if you changed them
- `CODEX_CMD` and `CODEX_WORKDIR` if you want the Codex CLI backend

More detail lives in [Specification: Configuration](../spec/configuration.md).

## Start the Service

```bash
source .venv/bin/activate
./scripts/run_server.sh
```

## Health Check

```bash
curl http://127.0.0.1:8080/health
```

## Smoke Tests

Reasoning path:

```bash
python scripts/smoke_test.py scripts/prompts/01_local_reasoning.txt
```

Coding path:

```bash
python scripts/smoke_test.py scripts/prompts/02_local_coding.txt
```

Codex escalation path:

```bash
python scripts/smoke_test.py scripts/prompts/03_cloud_escalation.txt
```

To bypass routing during testing:

```bash
python scripts/smoke_test.py scripts/prompts/02_local_coding.txt --preferred-backend local_coder
python scripts/smoke_test.py scripts/prompts/01_local_reasoning.txt --preferred-backend local_reasoner
python scripts/smoke_test.py scripts/prompts/03_cloud_escalation.txt --preferred-backend codex_cli
```

Anthropic-style gateway check:

```bash
python scripts/anthropic_gateway_test.py
```
