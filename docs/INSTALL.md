# Install

## 1. Prereqs

- Python 3.11+
- Ollama on each machine that will host a local model
- network reachability from the router-service host to each Ollama host
- optional: Codex CLI installed locally

## 2. Pull local models

Example commands:

```bash
# router box
ollama pull qwen3:8b-q4_K_M

# coding box
ollama pull qwen3-coder:30b-a3b-q4_K_M

# reasoning box
ollama pull qwen3:14b
```

If `qwen3:14b` is too slow or too big on the 3080 8GB box, fall back to:

```bash
ollama pull qwen3:8b-q4_K_M
```

## 3. Create a virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Configure env vars

```bash
cp .env.example .env
```

Then edit `.env`.

At minimum, set:

- `ROUTER_OLLAMA_BASE_URL`
- `CODER_OLLAMA_BASE_URL`
- `REASONER_OLLAMA_BASE_URL`
- model names if you changed them
- `CODEX_CMD` and `CODEX_WORKDIR` if you want the Codex CLI backend

## 5. Start the service

```bash
source .venv/bin/activate
./scripts/run_server.sh
```

## 6. Health check

```bash
curl http://127.0.0.1:8080/health
```

## 7. Run a direct smoke test

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

If you want to bypass routing and force a backend while testing:

```bash
python scripts/smoke_test.py scripts/prompts/02_local_coding.txt --preferred-backend local_coder
python scripts/smoke_test.py scripts/prompts/01_local_reasoning.txt --preferred-backend local_reasoner
python scripts/smoke_test.py scripts/prompts/03_cloud_escalation.txt --preferred-backend codex_cli
```

## 8. Anthropic-style gateway test

```bash
python scripts/anthropic_gateway_test.py
```

That verifies the `/v1/messages` endpoint.
