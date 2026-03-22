# Technical Specification

## Purpose

This specification describes the current implementation, not an aspirational redesign.

## Runtime Topology

### Primary Components

1. API server
   Implemented by [`app/main.py`](/home/jesse/src/coding-agent-router/app/main.py) and [`app/compaction_main.py`](/home/jesse/src/coding-agent-router/app/compaction_main.py).
2. Routing service
   Implemented by [`app/router.py`](/home/jesse/src/coding-agent-router/app/router.py).
3. Backend clients
   Implemented by [`app/clients/ollama_client.py`](/home/jesse/src/coding-agent-router/app/clients/ollama_client.py) and [`app/clients/codex_client.py`](/home/jesse/src/coding-agent-router/app/clients/codex_client.py).
4. Compatibility layer
   Implemented by [`app/compat.py`](/home/jesse/src/coding-agent-router/app/compat.py).
5. Tool translation layer
   Implemented by [`app/tool_adapter.py`](/home/jesse/src/coding-agent-router/app/tool_adapter.py) and [`app/tool_surface.py`](/home/jesse/src/coding-agent-router/app/tool_surface.py).
6. App-server bridge
   Implemented by [`app/app_server.py`](/home/jesse/src/coding-agent-router/app/app_server.py).
7. Transcript compaction subsystem
   Implemented under [`app/compaction/`](/home/jesse/src/coding-agent-router/app/compaction).
8. Prompt loading and rendering
   Implemented by [`app/prompt_loader.py`](/home/jesse/src/coding-agent-router/app/prompt_loader.py) with prompt files in [`app/prompts`](/home/jesse/src/coding-agent-router/app/prompts).

## Configuration Model

Configuration is loaded from environment variables through [`app/config.py`](/home/jesse/src/coding-agent-router/app/config.py).

### Active Routing Settings

- `ROUTER_OLLAMA_BASE_URL`
- `ROUTER_MODEL`
- `ROUTER_NUM_CTX`
- `ROUTER_TEMPERATURE`
- `ROUTER_TIMEOUT_SECONDS`
- `CODER_OLLAMA_BASE_URL`
- `CODER_MODEL`
- `CODER_NUM_CTX`
- `CODER_TEMPERATURE`
- `CODER_TIMEOUT_SECONDS`
- `REASONER_OLLAMA_BASE_URL`
- `REASONER_MODEL`
- `REASONER_NUM_CTX`
- `REASONER_TEMPERATURE`
- `REASONER_TIMEOUT_SECONDS`
- `ENABLE_LOCAL_CODER`
- `ENABLE_LOCAL_REASONER`
- `ENABLE_CODEX_CLI`
- `CODEX_CMD`
- `CODEX_WORKDIR`
- `CODEX_EXEC_MODEL_PROVIDER`
- `CODEX_EXEC_MODEL`
- `CODEX_TIMEOUT_SECONDS`

### Active Compaction Settings

- `COMPACTOR_OLLAMA_BASE_URL`
- `COMPACTOR_MODEL`
- `COMPACTOR_NUM_CTX`
- `COMPACTOR_TEMPERATURE`
- `COMPACTOR_TIMEOUT_SECONDS`
- `COMPACTOR_TARGET_CHUNK_TOKENS`
- `COMPACTOR_MAX_CHUNK_TOKENS`
- `COMPACTOR_OVERLAP_TOKENS`
- `COMPACTOR_KEEP_RAW_TOKENS`
- `COMPACTION_STATE_DIR`
- `INLINE_COMPACT_SENTINEL`
- `OPENAI_PASSTHROUGH_BASE_URL`
- `APP_SERVER_STATE_DIR`

### Present but Not Operationally Significant

- `APP_SERVER_MODE`
- `ENABLE_INCREMENTAL_COMPACTION`
- `COMPACTOR_MERGE_BATCH_SIZE`
- `DEFAULT_CLOUD_BACKEND`
- `FAIL_OPEN`

## Request Models

Core Pydantic models are defined in [`app/models.py`](/home/jesse/src/coding-agent-router/app/models.py).

### Native Invoke

`InvokeRequest` fields:

- `prompt: str`
- `system: str | None`
- `preferred_backend: str | None`
- `metadata: dict`

### Anthropic Messages

`AnthropicMessagesRequest` fields:

- `model: str | None`
- `max_tokens: int`
- `system: str | list[dict] | None`
- `messages: list[Message]`
- `tools: list[dict] | None`
- `metadata: dict | None`

### Internal Compaction

`CompactRequest` fields:

- `session_id: str`
- `items: list[dict]`
- `current_request: str`
- `repo_context: dict`
- `refresh_if_needed: bool`

## Endpoint Specification

### Full Router Service

Defined in [`app/main.py`](/home/jesse/src/coding-agent-router/app/main.py).

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/health` | Returns static health plus configured model names and Codex toggle |
| `GET` | `/api/version` | Returns synthetic Ollama version payload |
| `GET` | `/api/tags` | Returns synthetic Ollama model list |
| `GET` | `/v1/models` | Returns synthetic OpenAI model list |
| `POST` | `/invoke` | Native routing and execution |
| `POST` | `/internal/compact` | Runs transcript compaction |
| `POST` | `/v1/messages` | Anthropic-compatible request/response mapping |
| `POST` | `/v1/chat/completions` | OpenAI Chat compatibility mapping |
| `POST` | `/v1/responses` | OpenAI Responses compatibility mapping |
| `POST` | `/api/chat` | Ollama chat compatibility mapping |
| `WS` | `/app-server/ws` | Codex app-server bridge |

### Compaction Companion Service

Defined in [`app/compaction_main.py`](/home/jesse/src/coding-agent-router/app/compaction_main.py).

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/health` | Same as full router plus `app_server_mode=compaction_only` |
| `GET` | `/api/version` | Synthetic Ollama discovery |
| `GET` | `/api/tags` | Synthetic Ollama discovery |
| `GET` | `/v1/models` | Synthetic OpenAI discovery |
| `POST` | `/invoke` | Native routing and execution |
| `POST` | `/internal/compact` | Runs transcript compaction |
| `POST` | `/v1/messages` | Rejected with HTTP 409 |
| `POST` | `/v1/chat/completions` | Rejected with HTTP 409 |
| `POST` | `/v1/responses` | Inline local compaction or upstream passthrough |
| `POST` | `/api/chat` | Rejected with HTTP 409 |
| `WS` | `/app-server/ws` | Codex app-server bridge with forced Codex backend |

## Routing Algorithm

Implemented in [`app/router.py`](/home/jesse/src/coding-agent-router/app/router.py).

### Prompt Sources

The routing layer uses prompt files rather than long embedded literals.

Current prompt files:

- [`router_system.md`](/home/jesse/src/coding-agent-router/app/prompts/router_system.md)
- [`router_task.md`](/home/jesse/src/coding-agent-router/app/prompts/router_task.md)
- [`coder_system.md`](/home/jesse/src/coding-agent-router/app/prompts/coder_system.md)
- [`reasoner_system.md`](/home/jesse/src/coding-agent-router/app/prompts/reasoner_system.md)
- [`conversation_compactor_system.md`](/home/jesse/src/coding-agent-router/app/prompts/conversation_compactor_system.md)

### Input Normalization

Anthropic-shaped requests are flattened into:

- `system`
- `prompt`
- `user_prompt`
- `trajectory`

Text flattening preserves text blocks and JSON-stringifies non-text blocks.

### Metrics Extraction

The router builds a digest containing:

- task instruction string
- eligible route list
- user prompt
- trajectory
- quantitative metrics from [`app/task_metrics.py`](/home/jesse/src/coding-agent-router/app/task_metrics.py)

Current metrics include counts such as:

- prompt chars, lines, tokens
- trajectory chars, lines, tokens
- message counts by role
- tool call count
- command count
- command output token count
- file reference counts
- code block counts
- JSON block counts
- diff line count
- error line count
- stack trace count
- prior failure count
- question count
- metadata key count
- backend request tokens
- router payload tokens
- router request tokens
- configured context limits

### Route Eligibility Rules

1. Start from enabled backends.
2. If `preferred_backend` is set, return it immediately with confidence `1.0`.
3. Remove `local_reasoner` if backend request tokens exceed `REASONER_NUM_CTX`.
4. Remove `local_coder` if backend request tokens exceed `CODER_NUM_CTX`.
5. If router request tokens exceed `ROUTER_NUM_CTX`, bypass router-model inference and fall back to:
   - `codex_cli` if available
   - otherwise first local fallback
6. If no eligible local route remains, fall back to:
   - `codex_cli` if available
   - otherwise first local fallback
7. If exactly one route remains, select it without LLM inference.
8. Otherwise ask the router Ollama model to return JSON with:
   - `route`
   - `confidence`
   - `reason`
9. If parsing fails or the route is invalid, use deterministic fallback order:
   - `local_coder`
   - `local_reasoner`
   - `codex_cli`

## Backend Dispatch

### Local Coder

Uses `OllamaClient.chat()` or `chat_stream()` with:

- coder model
- coder temperature
- coder context window
- optional system prompt
- optional normalized tools

Special handling:

- Devstral-like embedded tool JSON is recovered into structured tool calls

### Local Reasoner

Uses `OllamaClient.chat()` or `chat_stream()` with:

- reasoner model
- reasoner temperature
- reasoner context window
- no tool surface

### Codex CLI

Uses `CodexCLIClient.exec_prompt()` with:

- `codex exec`
- `--skip-git-repo-check`
- configured `model_provider`
- configured `model`
- optional `-C <workdir>`

Prompt construction:

- if compaction session state exists, render a structured Codex handoff prompt
- otherwise pass `system + prompt`

## Prompt Loader Specification

Implemented in [`app/prompt_loader.py`](/home/jesse/src/coding-agent-router/app/prompt_loader.py).

### `load_prompt(name)`

Behavior:

- reads `<app>/prompts/<name>`
- caches prompt text in memory
- strips leading and trailing file whitespace

### `render_prompt(name, replacements)`

Behavior:

- loads the template from `app/prompts`
- replaces placeholders of the form `{{TOKEN_NAME}}`
- raises `ValueError` if any placeholders remain unresolved

### Dynamic Prompt Templates

Current dynamic templates:

- [`codex_support_prompt.md`](/home/jesse/src/coding-agent-router/app/prompts/codex_support_prompt.md)
- [`app_server_compacted_flow.md`](/home/jesse/src/coding-agent-router/app/prompts/app_server_compacted_flow.md)

Current replacement tokens include:

- `SYSTEM_SECTION`
- `DURABLE_MEMORY_BLOCKS`
- `STRUCTURED_HANDOFF`
- `RECENT_RAW_TURNS`
- `CURRENT_REQUEST`

## Compatibility Layer

Implemented in [`app/compat.py`](/home/jesse/src/coding-agent-router/app/compat.py).

### OpenAI Chat Input

System messages are folded into Anthropic `system`.

Assistant tool calls are transformed into Anthropic `tool_use` blocks.

Tool result messages with role `tool` are transformed into Anthropic `tool_result` blocks.

### Ollama Chat Input

Mapped similarly to Anthropic request shape, with `options.num_predict` translated into `max_tokens` when present.

### OpenAI Responses Input

Supports:

- string input
- message items
- `function_call`
- `function_call_output`
- `instructions`
- request metadata
- request tools with alias translation

### OpenAI Chat Output

Returns an OpenAI Chat Completions payload with:

- one assistant message
- optional function tool calls
- usage from backend token counts

Streaming note:

- the streaming variant emits one content chunk plus a terminal chunk and `[DONE]`
- it is not backed by progressive token streaming

### OpenAI Responses Output

Returns:

- `object=response`
- `output` items containing assistant text and/or function calls
- `output_text` prefixed with a route banner when route metadata exists
- `parallel_tool_calls=true` when tool calls are present

Responses streaming note:

- this is the only fully progressive compatibility stream in the main router
- emitted event types include `response.created`, `response.in_progress`, `response.output_item.added`, `response.output_text.delta`, `response.function_call_arguments.done`, and `response.completed`

### Ollama Output

Returns:

- assistant text
- optional `tool_calls`
- Ollama-style usage fields

Streaming note:

- stream mode emits one NDJSON line derived from the completed response

## Tool Translation Rules

### Anthropic to Ollama

Defined in [`app/tool_adapter.py`](/home/jesse/src/coding-agent-router/app/tool_adapter.py).

- Anthropic tools become Ollama function tools
- assistant `tool_use` becomes Ollama `tool_calls`
- user `tool_result` becomes Ollama `tool` messages

### Responses Alias Surface

Defined in [`app/tool_surface.py`](/home/jesse/src/coding-agent-router/app/tool_surface.py).

| Original | Alias | Mode |
| --- | --- | --- |
| `rg` | `search_text` | identity |
| `exec_command` | `run_shell` | exec command field remap |
| `write_stdin` | `send_terminal_input` | stdin field remap |
| `update_plan` | `set_plan` | plan payload remap |
| `apply_patch` | `write_patch` | patch wrapper |

Mapped tool calls are restored to original names before final Responses output.

## App-Server Bridge Specification

Implemented in [`app/app_server.py`](/home/jesse/src/coding-agent-router/app/app_server.py).

### Supported Methods

- `initialize`
- `thread/start`
- `turn/start`
- `thread/compact/start`
- `initialized`

### Thread State

Persisted per thread as JSON under `APP_SERVER_STATE_DIR`.

Stored fields include:

- thread metadata
- approval policy
- cwd
- model
- model provider
- sandbox policy
- base instructions
- developer instructions
- turns
- history items
- compacted flow

Prompt rendering note:

- compacted follow-up context is rendered from [`app_server_compacted_flow.md`](/home/jesse/src/coding-agent-router/app/prompts/app_server_compacted_flow.md)

### Turn Execution

For `turn/start`:

1. flatten client input items into user text
2. build system prompt from base instructions and developer instructions
3. append rendered compacted flow if present
4. set metadata:
   - `compaction_session_id`
   - `session_id`
   - `preferred_backend` in compaction-only mode
   - `cwd`
5. execute through `invoke_from_anthropic`
6. emit `thread/status/changed`, `turn/started`, `item/started`, `item/completed`, `turn/completed`
7. persist updated state

### Thread Compaction

For `thread/compact/start`:

1. gather items from history and prior compacted flow
2. derive current request from latest user turn or thread preview
3. call transcript compaction
4. rebuild compacted flow
5. replace `history_items` with `recent_raw_turns`
6. persist state
7. emit `thread/compacted`

## Compaction Subsystem Specification

### Chunking

Implemented in [`app/compaction/chunking.py`](/home/jesse/src/coding-agent-router/app/compaction/chunking.py).

Behavior:

- estimates tokens heuristically with `len(text) / 4`
- retains newest turns as raw tail based on `COMPACTOR_KEEP_RAW_TOKENS`
- chunks older transcript items by target token budget
- carries overlap forward into the next chunk

### Extraction

Implemented in [`app/compaction/extractor.py`](/home/jesse/src/coding-agent-router/app/compaction/extractor.py).

Behavior:

- calls the compactor model in Ollama JSON mode
- supplies a strict extraction system prompt
- validates that the response is a JSON object
- validates into `ChunkExtraction`

Prompt source:

- [`compaction_extraction_system.md`](/home/jesse/src/coding-agent-router/app/prompts/compaction_extraction_system.md)

### Merge

Implemented in [`app/compaction/merger.py`](/home/jesse/src/coding-agent-router/app/compaction/merger.py).

Behavior:

- latest non-empty scalar wins
- repo state dictionaries are shallow-merged with later values overwriting earlier values
- list fields are deduplicated case-insensitively while preserving first occurrence order
- latest non-empty plan list wins

### Durable Memory Rendering

Implemented in [`app/compaction/durable_memory.py`](/home/jesse/src/coding-agent-router/app/compaction/durable_memory.py).

Rendered files:

- `TASK_STATE.md`
- `DECISIONS.md`
- `FAILURES_TO_AVOID.md`
- `NEXT_STEPS.md`
- `SESSION_HANDOFF.md`

### Handoff Flow

Implemented in [`app/compaction/handoff.py`](/home/jesse/src/coding-agent-router/app/compaction/handoff.py).

Output fields:

- `durable_memory`
- `structured_handoff`
- `recent_raw_turns`
- `current_request`

Rendering note:

- the final Codex support prompt is rendered from [`codex_support_prompt.md`](/home/jesse/src/coding-agent-router/app/prompts/codex_support_prompt.md)

### Storage Layout

Implemented in [`app/compaction/storage.py`](/home/jesse/src/coding-agent-router/app/compaction/storage.py).

Per-session structure:

- `<session>/chunks/chunk-<n>.json`
- `<session>/merged-state.json`
- `<session>/handoff.json`
- `<session>/TASK_STATE.md`
- `<session>/DECISIONS.md`
- `<session>/FAILURES_TO_AVOID.md`
- `<session>/NEXT_STEPS.md`
- `<session>/SESSION_HANDOFF.md`

## Passthrough Proxy Rules

Compaction companion passthrough behavior:

- preserves request body
- preserves most inbound headers
- strips `host`, `content-length`, and `connection`
- forwards to `OPENAI_PASSTHROUGH_BASE_URL + /responses`
- streams bytes through unchanged when upstream streaming succeeds
- logs transport metadata to `state/compaction_transport.jsonl`

## Discovery Endpoint Semantics

Discovery endpoints are compatibility shims, not backend truth:

- `/v1/models` returns one synthetic `gpt-5.4` model entry unless a Codex cache file exists
- `/api/version` returns `0.0.0-router`
- `/api/tags` returns one synthetic `router` model tag

## Deployment Artifacts

Systemd unit generation is defined in [`scripts/render_systemd_units.py`](/home/jesse/src/coding-agent-router/scripts/render_systemd_units.py).

Generated units:

- [`deploy/systemd/coding-agent-router.service`](/home/jesse/src/coding-agent-router/deploy/systemd/coding-agent-router.service)
- [`deploy/systemd/coding-agent-router-compaction.service`](/home/jesse/src/coding-agent-router/deploy/systemd/coding-agent-router-compaction.service)

The generated units pin:

- `uvicorn`
- host `127.0.0.1`
- dedicated ports `8080` and `8081`
- explicit Ollama endpoints
- explicit Codex CLI path

## Verified Behavioral Invariants from Tests

The test suite asserts all of the following:

- route enum is locked to three values
- local fallback prefers `local_coder`
- router bypasses inference when digest size exceeds router context
- coder tool recovery works for embedded and streaming tool JSON
- Responses tool alias mapping round-trips
- app-server thread state survives reload from disk
- compaction writes handoff and durable memory files
- compaction-only mode rejects unsupported transports
- inline compaction sentinel detection ignores historical tool output

## Known Technical Gaps

- true streaming parity does not exist across all compatibility endpoints
- some environment variables advertise behavior that the runtime does not implement
- there is no internal execution engine for returned tool calls
