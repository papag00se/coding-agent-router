# Routing Specification

[Docs Index](../index.md) | [Specification Index](./index.md)

Implemented in [`app/router.py`](/home/jesse/src/coding-agent-router/app/router.py).

## Route Set

Supported execution backends:

- `local_coder`
- `local_reasoner`
- `codex_cli`

## Input Normalization

Anthropic-shaped requests are flattened into:

- `system`
- `prompt`
- `user_prompt`
- `trajectory`

Text flattening preserves text blocks and JSON-stringifies non-text blocks.

## Metrics Extraction

Routing builds a compact digest containing:

- task instruction string
- eligible route list
- user prompt
- trajectory
- quantitative metrics from [`app/task_metrics.py`](/home/jesse/src/coding-agent-router/app/task_metrics.py)

Current metrics include prompt size, trajectory size, message-role counts, tool-call counts, command counts, command-output size, file-reference counts, diff size, error lines, stack traces, prior failures, and configured context limits.

## Route Eligibility Rules

1. Start from enabled backends.
2. If `preferred_backend` is set, return it immediately with confidence `1.0`.
3. Remove `local_reasoner` if backend request tokens exceed `REASONER_NUM_CTX`.
4. Remove `local_coder` if backend request tokens exceed `CODER_NUM_CTX`.
5. If router request tokens exceed `ROUTER_NUM_CTX`, bypass router-model inference and fall back to `codex_cli` if available, otherwise the first local fallback.
6. If no eligible local route remains, fall back to `codex_cli` if available, otherwise the first local fallback.
7. If exactly one route remains, select it without LLM inference.
8. Otherwise ask the router model to return JSON containing `route`, `confidence`, and `reason`.
9. If parsing fails or the route is invalid, use deterministic fallback order: `local_coder`, `local_reasoner`, `codex_cli`.

## Backend Dispatch

### Local Coder

Uses `OllamaClient.chat()` or `chat_stream()` with:

- coder model
- coder temperature
- coder context window
- optional system prompt
- optional normalized tools

Devstral-style embedded tool JSON is recovered into structured tool calls.

### Local Reasoner

Uses `OllamaClient.chat()` or `chat_stream()` with the configured reasoner model, temperature, and context window, without a structured tool surface.

### Codex CLI

Uses `CodexCLIClient.exec_prompt()` with:

- `codex exec`
- `--skip-git-repo-check`
- configured `model_provider`
- configured `model`
- optional `-C <workdir>`

If compaction state exists for the session, the router injects the structured Codex handoff prompt. Otherwise it passes `system + prompt`.
