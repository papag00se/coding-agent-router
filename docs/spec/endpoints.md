# Endpoint Specification

[Docs Index](../index.md) | [Specification Index](./index.md)

## Full Router Service

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

## Compaction Companion Service

Defined in [`app/compaction_main.py`](/home/jesse/src/coding-agent-router/app/compaction_main.py).

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/health` | Same as full router |
| `GET` | `/api/version` | Synthetic Ollama discovery |
| `GET` | `/api/tags` | Synthetic Ollama discovery |
| `GET` | `/v1/models` | Synthetic OpenAI discovery |
| `POST` | `/invoke` | Native routing and execution |
| `POST` | `/internal/compact` | Runs transcript compaction |
| `POST` | `/v1/messages` | Rejected with HTTP 409 |
| `POST` | `/v1/chat/completions` | Rejected with HTTP 409 |
| `POST` | `/v1/responses` | Inline local compaction or upstream passthrough |
| `POST` | `/api/chat` | Rejected with HTTP 409 |

## Additional `/v1/responses` Companion Behavior

- Inline compaction is detected from the configured sentinel in `instructions` or the current user turn
- Ordinary passthrough requests are proxied to `OPENAI_PASSTHROUGH_BASE_URL`
- Qualifying passthrough requests may be rewritten from `gpt-5.4` to `CODEX_SPARK_MODEL`
- Spark qualification uses the latest significant `function_call_output`, a tokenizer-based request-size estimate, and `CODEX_SPARK_QUALIFIED_RATE`
