# Endpoint Specification

[Docs Index](../index.md) | [Specification Index](./index.md)

## Full Router Service

Defined in [`app/main.py`](/home/jesse/src/coding-agent-router/app/main.py).

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/health` | Returns static health plus configured model names and Codex toggle |
| `GET` | `/api/version` | Returns synthetic Ollama version payload |
| `GET` | `/api/tags` | Returns synthetic Ollama model list |
| `GET` | `/internal/metrics` | Returns persisted transport and compaction counters plus estimated token-savings |
| `GET` | `/v1/models` | Returns synthetic OpenAI model list |
| `POST` | `/invoke` | Native routing and execution |
| `POST` | `/internal/compact` | Runs transcript compaction |
| `POST` | `/v1/messages` | Anthropic-compatible request/response mapping |
| `POST` | `/v1/chat/completions` | OpenAI Chat compatibility mapping with SSE keepalive while waiting |
| `POST` | `/v1/responses` | OpenAI Responses compatibility mapping |
| `POST` | `/api/chat` | Ollama chat compatibility mapping with blank-line keepalive while waiting |

## Compaction Companion Service

Defined in [`app/compaction_main.py`](/home/jesse/src/coding-agent-router/app/compaction_main.py).

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/health` | Same as full router |
| `GET` | `/api/version` | Synthetic Ollama discovery |
| `GET` | `/api/tags` | Synthetic Ollama discovery |
| `GET` | `/internal/metrics` | Returns persisted transport and compaction counters plus estimated token-savings |
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
- Qualifying passthrough requests may be rewritten from `gpt-5.4` to `CODEX_SPARK_MODEL` or `CODEX_MINI_MODEL`
- Image-bearing passthrough requests stay off the Spark lane; Spark rewrites are only used for image-free payloads, and Spark-targeted image requests are rewritten to `CODEX_MINI_MODEL`
- Mini rewrites remain available for qualifying image-bearing payloads
- Spark qualification uses a deterministic recent-payload classifier, a tokenizer-based request-size estimate, and `CODEX_SPARK_QUALIFIED_RATE`
- Spark rewrites normalize GPT-5.4-only request fields such as assistant-message `phase` markers and unsupported reasoning-effort values before forwarding to `CODEX_SPARK_MODEL`
- Mini selection is deterministic and currently covers Spark-oversize requests plus bounded reasoning-heavy analysis and investigation shapes
- The current Spark classifier covers retrieval, direct file reads, targeted tests, polling, localized edit loops, bounded test-fix loops, diff follow-ups, stacktrace triage, small refactors, and simple bounded synthesis
- Spark quota exhaustion retries the same request on `CODEX_MINI_MODEL` and keeps subsequent Spark-targeted requests on mini until the recorded reset time expires
- Inline local compaction validates the rendered Codex handoff shape before returning it
- If local inline compaction fails or the rendered handoff is invalid, the companion reruns the same normalize/chunk/extract/refine compaction pipeline on `CODEX_SPARK_MODEL`
- If the inline Spark fallback is blocked or unavailable, the companion retries that same pipeline on `CODEX_MINI_MODEL`
