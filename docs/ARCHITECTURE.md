# Architecture

## System Shape

The repository implements two related HTTP services:

- the full router in [`app/main.py`](/home/jesse/src/coding-agent-router/app/main.py)
- the compaction companion in [`app/compaction_main.py`](/home/jesse/src/coding-agent-router/app/compaction_main.py)

Both share the same routing core in [`app/router.py`](/home/jesse/src/coding-agent-router/app/router.py).

## High-Level Flow

```text
client
  -> compatibility layer / native invoke
  -> routing digest + route selection
  -> local_coder | local_reasoner | codex_cli
  -> compatibility response mapping
```

## Main Components

### API Surface

The runtime accepts:

- native `/invoke`
- Anthropic `/v1/messages`
- OpenAI `/v1/chat/completions`
- OpenAI `/v1/responses`
- Ollama `/api/chat`
- internal `/internal/compact`

Request and response translation lives in [`app/compat.py`](/home/jesse/src/coding-agent-router/app/compat.py).

### Route Selection

The router builds a compact digest rather than sending full history to the router model. The digest includes:

- latest user prompt
- trajectory summary payload
- token and structure metrics
- configured context constraints
- eligible routes

Route selection then:

- honors explicit backend overrides
- removes local backends that cannot fit the request
- bypasses inference when the routing digest itself is too large
- falls back deterministically if the router output is invalid

### Execution Backends

Supported backends:

- `local_coder` through Ollama
- `local_reasoner` through Ollama
- `codex_cli` through a local `codex exec` subprocess

### Tool Handling

Tool handling is implemented primarily for the local coder lane.

The system:

- normalizes Anthropic-style tools into Ollama format
- recovers embedded tool JSON from models like Devstral
- exposes a reduced alias tool surface for the Responses API

### Compaction and Handoff

The compaction subsystem:

- pre-cleans structured transcript items before compaction
- preserves the newest raw turn outside compaction
- chunks older transcript items at item boundaries
- extracts durable session state via a compactor model
- merges chunk state deterministically
- runs a constrained final refinement pass over merged state plus recent raw turns
- reattaches the newest raw turn after refinement
- writes Markdown and JSON handoff artifacts
- rebuilds a Codex-ready prompt for later work

### Prompt Files

Application prompts are stored under [`app/prompts`](/home/jesse/src/coding-agent-router/app/prompts) and loaded through [`app/prompt_loader.py`](/home/jesse/src/coding-agent-router/app/prompt_loader.py).

Static prompts such as router, coder, reasoner, and compactor instructions are plain Markdown files.

Dynamic prompts such as the Codex support prompt and compacted-flow prompt use explicit placeholder replacement for sections like current request, durable memory, and structured handoff.

## Service Modes

### Full Router

Used when the service should terminate client traffic and execute locally.

### Compaction Companion

Used when inline local compaction should run locally, while ordinary `/v1/responses` traffic is proxied upstream.

## Further Docs

- [`FEATURE_SET.md`](/home/jesse/src/coding-agent-router/docs/FEATURE_SET.md)
- [`PRD.md`](/home/jesse/src/coding-agent-router/docs/PRD.md)
- [`SPEC.md`](/home/jesse/src/coding-agent-router/docs/SPEC.md)
