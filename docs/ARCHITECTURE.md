# Architecture

## System Shape

The repository currently implements two related services:

- a full router service in [`app/main.py`](/home/jesse/src/coding-agent-router/app/main.py)
- a compaction companion service in [`app/compaction_main.py`](/home/jesse/src/coding-agent-router/app/compaction_main.py)

Both services share the same routing core in [`app/router.py`](/home/jesse/src/coding-agent-router/app/router.py).

## High-Level Flow

```text
client
  -> compatibility layer / native invoke
  -> routing digest + route selection
  -> local coder | local reasoner | codex cli
  -> compatibility response mapping
```

For long-running app-server sessions:

```text
app-server client
  -> /app-server/ws
  -> persisted thread state
  -> optional thread compaction
  -> durable memory + structured handoff
  -> follow-up turns with compacted context
```

## Main Components

### API and Compatibility Surface

The API layer accepts:

- native `/invoke`
- Anthropic `/v1/messages`
- OpenAI `/v1/chat/completions`
- OpenAI `/v1/responses`
- Ollama `/api/chat`
- app-server WebSocket traffic

Request and response translation is implemented in [`app/compat.py`](/home/jesse/src/coding-agent-router/app/compat.py).

### Route Selection

The router does not send the full conversation to the router model. It builds a smaller metrics digest that includes:

- latest user prompt
- trajectory summary payload
- token and structure metrics
- context-window constraints
- eligible routes

Route selection then:

- honors explicit backend overrides
- removes local routes that exceed backend context limits
- bypasses inference when the digest is itself too large
- falls back deterministically if router output is invalid
- loads route-selection prompt text from [`app/prompts`](/home/jesse/src/coding-agent-router/app/prompts)

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
- chunks older transcript items with overlap at item boundaries
- extracts durable session state via a compactor model
- merges chunk state deterministically
- runs a constrained final refinement pass over merged state plus recent raw turns
- reattaches the newest raw turn after refinement
- writes Markdown and JSON handoff artifacts
- rebuilds a Codex-ready prompt for later work

### Prompt Files

Application prompts are stored as Markdown files under [`app/prompts`](/home/jesse/src/coding-agent-router/app/prompts) and loaded through [`prompt_loader.py`](/home/jesse/src/coding-agent-router/app/prompt_loader.py).

Static prompts such as router, coder, reasoner, and compactor system instructions are plain `.md` files.

Dynamic prompts such as the Codex support prompt and the app-server compacted-flow prompt use explicit placeholder replacement for sections like current request, durable memory, and structured handoff.

### App-Server Bridge

The bridge persists per-thread state, serves turns over WebSocket, and can replace older raw history with compacted handoff state after `thread/compact/start`.

## Service Modes

### Full Router

Used when the service itself should terminate client traffic and execute locally.

### Compaction Companion

Used when app-server and inline local compaction should run locally, while ordinary `/v1/responses` traffic is proxied upstream.

## Further Docs

- [`FEATURE_SET.md`](/home/jesse/src/coding-agent-router/docs/FEATURE_SET.md)
- [`PRD.md`](/home/jesse/src/coding-agent-router/docs/PRD.md)
- [`SPEC.md`](/home/jesse/src/coding-agent-router/docs/SPEC.md)
