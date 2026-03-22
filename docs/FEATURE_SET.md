# Reverse-Engineered Feature Set

## Product Summary

This repository implements a local-first AI gateway for coding workflows. It is not just a router. The current product is a combination of:

- a FastAPI service that accepts multiple client API shapes
- a routing layer that chooses between local coding, local reasoning, and optional Codex CLI execution
- a transcript compaction pipeline that produces durable handoff state for long-running coding sessions
- a Codex app-server bridge that persists thread state and can inject compacted context back into later turns
- an optional compaction-only proxy service that locally handles inline compaction requests and proxies ordinary `/v1/responses` traffic upstream

## Operating Modes

### 1. Full Router Service

Implemented in [`app/main.py`](/home/jesse/src/coding-agent-router/app/main.py).

Primary capabilities:

- health and discovery endpoints
- direct `/invoke` execution
- Anthropic-style `/v1/messages`
- OpenAI-style `/v1/chat/completions`
- OpenAI Responses `/v1/responses`
- Ollama-style `/api/chat`
- Codex app-server WebSocket bridge at `/app-server/ws`
- internal transcript compaction endpoint at `/internal/compact`

### 2. Compaction Companion Service

Implemented in [`app/compaction_main.py`](/home/jesse/src/coding-agent-router/app/compaction_main.py).

Primary capabilities:

- same health and discovery endpoints
- same `/invoke` and `/internal/compact`
- app-server bridge at `/app-server/ws`
- `/v1/responses` as a split-mode endpoint:
  - inline compaction requests are served locally
  - non-compaction requests are proxied to `OPENAI_PASSTHROUGH_BASE_URL`
- explicit rejection of `/v1/messages`, `/v1/chat/completions`, and `/api/chat`

## Implemented Capabilities

### Multi-Protocol Ingress

The service currently accepts and normalizes:

- a native router request model via `/invoke`
- Anthropic Messages requests via `/v1/messages`
- OpenAI Chat Completions requests via `/v1/chat/completions`
- OpenAI Responses requests via `/v1/responses`
- Ollama Chat requests via `/api/chat`

Compatibility behavior is implemented in [`app/compat.py`](/home/jesse/src/coding-agent-router/app/compat.py).

### Route Selection Across Three Backends

The only supported execution routes are:

- `local_coder`
- `local_reasoner`
- `codex_cli`

Routing behavior in [`app/router.py`](/home/jesse/src/coding-agent-router/app/router.py):

- supports a caller-supplied `preferred_backend` override
- builds a compact routing digest rather than passing full conversation history to the router model
- computes numeric task metrics from prompt and trajectory
- loads router task and router system prompt text from files in [`app/prompts`](/home/jesse/src/coding-agent-router/app/prompts)
- removes local backends whose configured context windows are too small for the request
- skips the router model entirely when routing payload size exceeds router context
- falls back deterministically when the router output is invalid
- prefers local backends over Codex CLI when a fallback is needed

### Separate Local Coding and Reasoning Lanes

Implemented with independent Ollama clients and settings for:

- router model
- coder model
- reasoner model
- compactor model

Each lane has its own:

- base URL
- timeout
- context window
- temperature

### Codex CLI Escalation

When enabled, the router can execute `codex exec` as a subprocess through [`app/clients/codex_client.py`](/home/jesse/src/coding-agent-router/app/clients/codex_client.py).

Behavior:

- forces `model_provider` and `model` flags into the Codex CLI command
- optionally changes working directory from request metadata
- injects structured compaction handoff content into the prompt when a session id is present
- returns stdout or stderr as the backend-visible output

### Tool-Aware Coding Lane

The local coder lane supports structured tool calls when tools are present on Anthropic-style input.

Implemented behavior:

- Anthropic-style tool schemas are normalized for Ollama
- assistant `tool_use` content is converted into Ollama tool calls
- user `tool_result` content is converted into Ollama tool messages
- tool responses are mapped back into Anthropic/OpenAI/Responses formats
- Devstral-style tool JSON embedded in plain text is recovered into structured tool calls

Relevant files:

- [`app/tool_adapter.py`](/home/jesse/src/coding-agent-router/app/tool_adapter.py)
- [`app/tool_surface.py`](/home/jesse/src/coding-agent-router/app/tool_surface.py)

### Responses API Tool Surface Reduction

For OpenAI Responses requests, the service intentionally presents a smaller tool surface:

- `rg` becomes `search_text`
- `exec_command` becomes `run_shell`
- `write_stdin` becomes `send_terminal_input`
- `update_plan` becomes `set_plan`
- custom `apply_patch` becomes `write_patch`

Returned tool calls are mapped back to the original names before the final Responses payload is emitted.

### Streaming Support

Current streaming behavior is uneven by transport and should be understood as implemented, not as implied by endpoint names.

- `/v1/responses` in the full router supports progressive SSE driven by backend stream events
- `/v1/responses` in the compaction companion supports progressive SSE for inline compaction and byte-stream passthrough for upstream responses
- `/v1/chat/completions` streaming is synthesized from a completed response, not true token streaming
- `/api/chat` streaming is synthesized from a completed response, not true token streaming
- `/v1/messages` has no streaming path

### Transcript Compaction and Durable Handoff

The repository includes a dedicated compaction subsystem under [`app/compaction/`](/home/jesse/src/coding-agent-router/app/compaction).

Implemented flow:

- split a transcript into compactable history and recent raw turns
- chunk large histories with overlap
- extract chunk-local durable state with a compactor model in JSON mode
- merge chunk extractions into a session-level state
- render five durable memory documents
- persist JSON and Markdown artifacts to disk
- build a Codex-ready handoff flow with memory, structured handoff, recent raw turns, and current request

Persisted durable memory files:

- `TASK_STATE.md`
- `DECISIONS.md`
- `FAILURES_TO_AVOID.md`
- `NEXT_STEPS.md`
- `SESSION_HANDOFF.md`

### Prompt File Architecture

The runtime now keeps application prompts out of Python source.

Implemented behavior:

- prompt bodies live under [`app/prompts`](/home/jesse/src/coding-agent-router/app/prompts)
- prompt files are loaded through [`app/prompt_loader.py`](/home/jesse/src/coding-agent-router/app/prompt_loader.py)
- dynamic prompt templates use explicit placeholder replacement, for example `{{SYSTEM_SECTION}}` and `{{CURRENT_REQUEST}}`
- unresolved placeholders raise an error instead of silently leaking template tokens into runtime prompts

### App-Server Thread Bridge

The app-server bridge in [`app/app_server.py`](/home/jesse/src/coding-agent-router/app/app_server.py) implements a JSON-RPC-like WebSocket interface for Codex app-server clients.

Implemented methods:

- `initialize`
- `thread/start`
- `turn/start`
- `thread/compact/start`
- `initialized`

Implemented thread behavior:

- creates persisted thread state on disk
- keeps message history per thread
- appends compacted memory into the system prompt after compaction
- replaces older raw history with compacted `recent_raw_turns`
- can force `preferred_backend=codex_cli` in compaction-only mode

### Inline Compaction Sentinel

The compaction companion detects inline local compaction from a sentinel string, default `<<<LOCAL_COMPACT>>>`.

Implemented rules:

- checks `instructions`
- checks only the current user turn in Responses input
- ignores sentinel text that appears only in historical tool output

### Persistence and Observability

State is persisted under:

- `state/app_server`
- `state/compaction`

Transport logging is appended to:

- `state/compaction_transport.jsonl`

When `LOG_COMPACTION_PAYLOADS=true`, the transport log also includes full before/after compaction payload bodies for compaction operations.

### Local Deployment Artifacts

Operational helpers are included for:

- `uvicorn` startup via [`scripts/run_server.sh`](/home/jesse/src/coding-agent-router/scripts/run_server.sh)
- smoke tests via [`scripts/smoke_test.py`](/home/jesse/src/coding-agent-router/scripts/smoke_test.py)
- Anthropic gateway checks via [`scripts/anthropic_gateway_test.py`](/home/jesse/src/coding-agent-router/scripts/anthropic_gateway_test.py)
- systemd unit generation via [`scripts/render_systemd_units.py`](/home/jesse/src/coding-agent-router/scripts/render_systemd_units.py)
- proxying `codex app-server` traffic via [`scripts/codex_app_server_proxy.py`](/home/jesse/src/coding-agent-router/scripts/codex_app_server_proxy.py)

## Explicit Product Limits

The current implementation does not provide:

- a real multi-tenant gateway
- auth, rate limiting, quotas, or billing controls
- a general tool execution runtime inside the router service
- guaranteed fidelity to Claude Code or full Anthropic protocol parity
- true backend streaming for OpenAI Chat Completions or Ollama chat compatibility endpoints
- retry orchestration, circuit breaking, or advanced failover
- direct execution against arbitrary cloud model providers other than Codex CLI subprocess use and Responses passthrough

## Dormant or Partially Wired Surfaces

These surfaces exist in code or config but are not meaningfully wired into the runtime today:

- `APP_SERVER_MODE` exists in settings but runtime mode is chosen directly in code
- `ENABLE_INCREMENTAL_COMPACTION` exists but is not consulted by compaction logic
- `COMPACTOR_MERGE_BATCH_SIZE` exists but the merger does not batch
- `DEFAULT_CLOUD_BACKEND` exists but routing hardcodes `codex_cli`
- `FAIL_OPEN` exists but routing and transport code do not use it

## Reverse-Engineered Product Boundary

The implemented product is best described as:

> a local coding-workflow gateway that normalizes several client protocols, routes work between local specialist models and an optional Codex CLI lane, and preserves long session continuity through compaction-backed handoff state.

It is not currently a general-purpose enterprise LLM gateway.
