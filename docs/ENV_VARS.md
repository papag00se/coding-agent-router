# Environment variables

## Required for local-only operation

- `ROUTER_OLLAMA_BASE_URL`
- `ROUTER_MODEL`
- `CODER_OLLAMA_BASE_URL`
- `CODER_MODEL`
- `REASONER_OLLAMA_BASE_URL`
- `REASONER_MODEL`

## Optional Codex CLI backend

- `ENABLE_CODEX_CLI=true`
- `CODEX_CMD=codex`
- `CODEX_WORKDIR=/path/to/repo`
- `CODEX_TIMEOUT_SECONDS`

## Optional compactor

- `COMPACTOR_OLLAMA_BASE_URL`
- `COMPACTOR_MODEL`
- `COMPACTOR_NUM_CTX`
- `COMPACTOR_TEMPERATURE`
- `COMPACTOR_TIMEOUT_SECONDS`
- `COMPACTOR_TARGET_CHUNK_TOKENS`
- `COMPACTOR_MAX_CHUNK_TOKENS`
- `COMPACTOR_OVERLAP_TOKENS`
- `COMPACTOR_KEEP_RAW_TOKENS`
- `COMPACTOR_RESPONSE_HEADROOM_TOKENS`
- `COMPACTOR_MERGE_BATCH_SIZE`
- `COMPACTION_STATE_DIR`
- `ENABLE_INCREMENTAL_COMPACTION`
- `LOG_COMPACTION_PAYLOADS`
- `INLINE_COMPACT_SENTINEL`
- `OPENAI_PASSTHROUGH_BASE_URL`
- `CODEX_SPARK_MODEL`
- `CODEX_SPARK_QUALIFIED_RATE`

`OPENAI_PASSTHROUGH_BASE_URL` is the hosted Codex/OpenAI upstream that ordinary non-compaction traffic is proxied to when the compaction router is used as an OAuth proxy. In ChatGPT-auth mode, that should normally be `https://chatgpt.com/backend-api/codex`, not `https://api.openai.com/v1`.

`LOG_COMPACTION_PAYLOADS=true` adds full before/after compaction payloads to `state/compaction_transport.jsonl`. Leave it off unless you explicitly want that data on disk.

`CODEX_SPARK_MODEL` is the upstream model id used when a qualifying non-compaction `/v1/responses` passthrough call is rewritten from `gpt-5.4` to Spark.

`CODEX_SPARK_QUALIFIED_RATE` controls the fraction of qualifying passthrough calls that are sent to Spark. The router only considers requests under a rough `100k` token payload size whose latest tool-output shape is retrieval, listing, direct file read, targeted test, or polling. `1.0` means all qualifying calls go to Spark. `0.0` disables the rewrite. The default is `0.2`.

## Routing toggles

- `ENABLE_LOCAL_CODER`
- `ENABLE_LOCAL_REASONER`
- `ENABLE_CODEX_CLI`
- `DEFAULT_CLOUD_BACKEND`

## Context and temperature tuning

The important defaults are already set. Change them only if you have a reason.

- router: low temperature, smaller context
- coder: low temperature, medium context
- reasoner: low temperature, medium context

Suggested starting values:

- router -> `temperature=0.0`, `num_ctx=4096`
- coder -> `temperature=0.1`, `num_ctx=12000`
- reasoner -> `temperature=0.1`, `num_ctx=12000`
