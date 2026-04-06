# Configuration Specification

[Docs Index](../index.md) | [Specification Index](./index.md)

Configuration is loaded from environment variables through [`app/config.py`](/home/jesse/src/coding-agent-router/app/config.py).

## Routing Settings

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

## Compaction Settings

- `COMPACTOR_OLLAMA_BASE_URL`
- `COMPACTOR_MODEL`
- `COMPACTOR_NUM_CTX`
- `COMPACTOR_BURST_NUM_CTX`
- `COMPACTOR_TEMPERATURE`
- `COMPACTOR_TIMEOUT_SECONDS`
- `COMPACTOR_TARGET_CHUNK_TOKENS`
- `COMPACTOR_MAX_CHUNK_TOKENS`
- `COMPACTOR_MAX_PROMPT_TOKENS`
- `COMPACTOR_OVERLAP_TOKENS`
- `COMPACTOR_KEEP_RAW_TOKENS`
- `COMPACTION_STATE_DIR`
- `LOG_COMPACTION_PAYLOADS`
- `INLINE_COMPACT_SENTINEL`
- `OPENAI_PASSTHROUGH_BASE_URL`
- `ANTHROPIC_PASSTHROUGH_BASE_URL`
- `CODEX_MINI_MODEL`
- `CODEX_SPARK_MODEL`
- `CODEX_SPARK_QUALIFIED_RATE`

## Other Present Settings

These exist in config but are not operationally significant in the current runtime:

- `ENABLE_INCREMENTAL_COMPACTION`
- `COMPACTOR_MERGE_BATCH_SIZE`
- `DEFAULT_CLOUD_BACKEND`
- `FAIL_OPEN`

## Notes

`OPENAI_PASSTHROUGH_BASE_URL` is the hosted Codex/OpenAI upstream for ordinary non-compaction traffic when the compaction companion is used as an OAuth proxy. In ChatGPT-auth mode, that should normally be `https://chatgpt.com/backend-api/codex`.

`ANTHROPIC_PASSTHROUGH_BASE_URL` is the upstream Anthropic API for non-compaction Claude Code traffic (default `https://api.anthropic.com`). Set `ANTHROPIC_BASE_URL` in your Claude Code environment to point at the companion service, and non-compaction requests will be proxied upstream through this URL. Requests containing the inline compaction sentinel are handled locally via the shared Ollama queue.

Local Ollama requests are serialized per `base_url` by the client gate. `OLLAMA_POOL_CONNECTIONS` and `OLLAMA_POOL_MAXSIZE` only tune HTTP connection reuse.

`LOG_COMPACTION_PAYLOADS=true` adds full before and after compaction payloads to `state/compaction_transport.jsonl`. Leave it off unless you explicitly want raw payloads on disk.

Transport and rewrite counters are persisted separately in `state/compaction_metrics.json`. Those counters are updated directly from runtime transport events, include estimated token-savings for Spark, mini, and local compaction paths, and continue to work even when `LOG_COMPACTION_PAYLOADS` is off. Spark rewrite stats also distinguish ordinary mini preemption from Spark quota blocks. During bootstrap from older transport logs, local compaction savings are recovered from the intercepted `before_payload` when present; if an old completed local compaction cannot be paired back to a request, the bootstrap uses a `250000`-token default.

`COMPACTOR_BURST_NUM_CTX` lets extractor and refiner requests temporarily raise context above the normal `COMPACTOR_NUM_CTX`. The default behavior keeps compaction at `16384` context but may expand a specific request to `17408` when the default window would leave too little room for valid JSON output.

`COMPACTOR_TARGET_CHUNK_TOKENS` is the hard extraction chunk-content limit, not a soft target. The current default is `10000`.

`COMPACTOR_MAX_CHUNK_TOKENS` is a legacy compatibility knob. In the current runtime it cannot raise extraction chunk size above `COMPACTOR_TARGET_CHUNK_TOKENS`; the effective hard chunk limit is the lower of the two values.

`COMPACTOR_MAX_PROMPT_TOKENS` is the hard ceiling for the full extraction/refinement request estimate, including the system prompt, the compact transcript payload, and the strict structured-output JSON schema. The current default is `12256`, which keeps compaction well below the upstream request limit once the request envelope is counted.

Refinement also has a built-in per-iteration cap of about `8000` recent-raw tokens. That cap is code-level behavior, not an environment-variable knob.

`CODEX_MINI_MODEL` is the upstream model id used when a non-compaction `/v1/responses` passthrough request is rewritten from `gpt-5.4` to the middle-tier mini lane.

Mini selection is deterministic. The current policy rewrites `gpt-5.4` passthrough requests to `CODEX_MINI_MODEL` when either:

- the tokenizer-based estimated request size is above the Spark cap of `114688`
- the recent payload matches a bounded reasoning-heavy category such as bounded investigation, cross-file analysis, medium diff follow-up, medium test triage, or multi-file refactor

Repo-wide or architecture-scale requests are intentionally left on full `gpt-5.4`.

`CODEX_SPARK_MODEL` is the upstream model id used when a qualifying non-compaction `/v1/responses` passthrough request is rewritten from `gpt-5.4` to Spark.

Spark rewrites are skipped for payloads that carry image inputs. If a request already targets `CODEX_SPARK_MODEL` and carries images, the proxy rewrites it to `CODEX_MINI_MODEL` instead.

When a request is rewritten to `CODEX_SPARK_MODEL`, the proxy also normalizes request fields that are specific to newer GPT-5.4 payloads before forwarding the request to Spark. Today that includes stripping assistant-message `phase` markers and coercing unsupported reasoning-effort values onto Spark-supported settings.

`CODEX_SPARK_QUALIFIED_RATE` controls the fraction of qualifying passthrough calls sent to Spark. Qualification currently requires:

- original model `gpt-5.4`
- tokenizer-based estimated request size at or under `114688`
- a deterministic bounded-work classifier match from the recent payload
- stable-hash sampling below the configured rate

Current Spark categories are:

- `file_read`
- `search_inventory`
- `targeted_test`
- `polling`
- `test_fix_loop`
- `diff_followup`
- `stacktrace_triage`
- `localized_edit`
- `small_refactor`
- `simple_synthesis`

Current mini-only categories are:

- `oversize_for_spark`
- `bounded_investigation`
- `cross_file_analysis`
- `medium_diff_followup`
- `medium_test_triage`
- `multi_file_refactor`

Spark quota exhaustion is handled by a persisted circuit breaker under `COMPACTION_STATE_DIR` at `spark_quota.json` (default `state/compaction/spark_quota.json`). When a Spark-targeted request gets a quota or rate-limit response, the companion records the reset hint if the upstream sends one, retries the same request on mini, and keeps subsequent Spark-targeted requests on mini until the recorded reset time expires. If the upstream does not send a reset hint, the breaker uses a short probe interval and backs off between failed probes. The next qualifying request after expiry re-probes Spark automatically. The same breaker is also consulted by inline compaction Spark fallback, which retries the chunked compaction pipeline on `CODEX_MINI_MODEL` when Spark is blocked or unavailable.
