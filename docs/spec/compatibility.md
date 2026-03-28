# Compatibility Specification

[Docs Index](../index.md) | [Specification Index](./index.md)

Implemented in [`app/compat.py`](/home/jesse/src/coding-agent-router/app/compat.py).

## Request Models

Core Pydantic models live in [`app/models.py`](/home/jesse/src/coding-agent-router/app/models.py).

Important shapes:

- `InvokeRequest`
- `AnthropicMessagesRequest`
- `CompactRequest`

## Input Mapping

### OpenAI Chat Input

- System messages are folded into Anthropic `system`
- Assistant tool calls become Anthropic `tool_use` blocks
- Tool role messages become Anthropic `tool_result` blocks

### Ollama Chat Input

Mapped similarly to the Anthropic request shape, with `options.num_predict` translated into `max_tokens` when present.

### OpenAI Responses Input

Supports:

- string input
- message items
- `function_call`
- `function_call_output`
- `instructions`
- request metadata
- request tools with alias translation

## Output Mapping

### OpenAI Chat Output

Returns an OpenAI Chat Completions payload with:

- one assistant message
- optional function tool calls
- usage from backend token counts

Streaming is synthesized from a completed response rather than backed by progressive token generation, and the stream emits keepalive comments while waiting on the backend response.

### OpenAI Responses Output

Returns:

- `object=response`
- `output` items containing assistant text and or function calls
- `output_text` prefixed with a route banner when route metadata exists
- `parallel_tool_calls=true` when tool calls are present

The full router's `/v1/responses` stream is the only fully progressive compatibility stream in the main service. Its sibling `/v1/chat/completions` stream is synthesized from a completed response and uses keepalive comments while it waits.

### Ollama Output

Returns assistant text, optional `tool_calls`, and Ollama-style usage fields. Stream mode emits NDJSON derived from a completed response and blank-line keepalives while it waits on the backend response.

## Tool Translation

### Anthropic to Ollama

Defined in [`app/tool_adapter.py`](/home/jesse/src/coding-agent-router/app/tool_adapter.py).

- Anthropic tools become Ollama function tools
- Assistant `tool_use` becomes Ollama `tool_calls`
- User `tool_result` becomes Ollama `tool` messages

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
