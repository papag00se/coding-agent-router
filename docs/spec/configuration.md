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
- `COMPACTOR_OVERLAP_TOKENS`
- `COMPACTOR_KEEP_RAW_TOKENS`
- `COMPACTION_STATE_DIR`
- `LOG_COMPACTION_PAYLOADS`
- `INLINE_COMPACT_SENTINEL`
- `OPENAI_PASSTHROUGH_BASE_URL`
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

`LOG_COMPACTION_PAYLOADS=true` adds full before and after compaction payloads to `state/compaction_transport.jsonl`. Leave it off unless you explicitly want raw payloads on disk.

`COMPACTOR_BURST_NUM_CTX` lets extractor and refiner requests temporarily raise context above the normal `COMPACTOR_NUM_CTX`. The default behavior keeps compaction at `16384` context but may expand a specific request to `17408` when the default window would leave too little room for valid JSON output.

Refinement also has a built-in per-iteration cap of about `8000` recent-raw tokens. That cap is code-level behavior, not an environment-variable knob.

`CODEX_SPARK_MODEL` is the upstream model id used when a qualifying non-compaction `/v1/responses` passthrough request is rewritten from `gpt-5.4` to Spark.

`CODEX_SPARK_QUALIFIED_RATE` controls the fraction of qualifying passthrough calls sent to Spark. Qualification currently requires:

- original model `gpt-5.4`
- tokenizer-based estimated request size at or under `114688`
- latest significant input item is a qualifying `function_call_output`
- stable-hash sampling below the configured rate
