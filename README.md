# Local Agent Router Service

A small starter project for a **local-first coding/reasoning stack** with an **AI router** in front of separate backends.

## What this does

- runs a **router service** you control
- uses a **small local router model** to choose a backend per task
- sends coding tasks to a **local coding model**
- sends reasoning tasks to a **local reasoning model**
- can optionally escalate to an experimental **Codex CLI** backend
- exposes a simple `/invoke` endpoint for direct testing
- exposes a minimal **Anthropic `/v1/messages`-compatible** endpoint so you can later place it in front of Claude-style workflows

## Recommended hardware mapping

- **GTX 1080 8GB** -> router model -> `qwen3:8b-q4_K_M`
- **RTX 3080 10GB** -> coding model -> `qwen3-coder:30b-a3b-q4_K_M`
- **RTX 3080 8GB** -> reasoning model -> `qwen3:14b` or fallback `qwen3:8b`

## What is inside

- `app/` -> FastAPI router service
- `app/prompts/` -> application prompt templates loaded by the runtime
- `scripts/` -> run and test helpers
- `scripts/prompts/` -> prompt files that exercise reasoning, coding, and cloud escalation lanes
- `docs/` -> install + architecture docs
- `.env.example` -> all environment variables you need
- `app/clients/ollama_client.py` uses keep-alive HTTP sessions with configurable pool sizing.

## Fast start

1. Copy `.env.example` to `.env`
2. Edit the model URLs, model names, and API keys
3. Install requirements
4. Start the service
5. Run a smoke test prompt

Start command:

```bash
./scripts/run_server.sh
```

If you prefer to run directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Long-running requests

If routing or execution needs several minutes, raise timeout values in `.env`:

- `ROUTER_TIMEOUT_SECONDS`
- `CODER_TIMEOUT_SECONDS`
- `REASONER_TIMEOUT_SECONDS`
- `CODEX_TIMEOUT_SECONDS`

The Ollama clients also use keep-alive HTTP pools. Tune with:

- `OLLAMA_CONNECT_TIMEOUT_SECONDS`
- `OLLAMA_POOL_CONNECTIONS`
- `OLLAMA_POOL_MAXSIZE`

See [the docs index](/home/jesse/src/coding-agent-router/docs/index.md) first, then start with [installation](/home/jesse/src/coding-agent-router/docs/product/installation.md).
