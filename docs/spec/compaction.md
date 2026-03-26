# Compaction Specification

[Docs Index](../index.md) | [Specification Index](./index.md)

Implemented under [`app/compaction/`](/home/jesse/src/coding-agent-router/app/compaction).

## Pipeline

1. Normalize transcript items before compaction
2. Preserve the newest top-level raw turn outside compacted history
3. Split older compactable history from recent raw turns
4. Chunk compactable history at item boundaries using prompt-aware request sizing
5. Extract chunk-local durable state through the compactor model
6. Merge chunk states deterministically
7. Extract recent-turn state and merge it onto merged state using `8000`-token recent-raw iterations
8. Reattach the preserved newest raw turn
9. Render and persist durable handoff artifacts

## Normalization

Current normalization behavior:

- strips `encrypted_content` recursively
- removes attachment-like content blocks from compactable history
- drops historical `tool_result` and `function_call_output` blocks from compactable history
- keeps prior machine-generated continuation summaries out of extractor chunks and carries them forward raw for refinement/final handoff
- strips the known Codex bootstrap `instructions` block for inline compaction requests
- preserves the newest top-level turn raw for later reattachment

If a compactable item still cannot fit inside extraction prompt budget, it is skipped from chunk extraction and carried forward raw instead.

For inline compaction fallback, Spark uses this same normalized transcript rather than receiving a single replay of the original raw thread payload.

## Chunking

Implemented in [`app/compaction/chunking.py`](/home/jesse/src/coding-agent-router/app/compaction/chunking.py).

Behavior:

- keeps a recent raw tail based on `COMPACTOR_KEEP_RAW_TOKENS`
- chunks older items by prompt budget rather than transcript-size heuristics alone
- enforces `COMPACTOR_TARGET_CHUNK_TOKENS` as a hard chunk-content limit instead of treating it as a soft target
- enforces `COMPACTOR_MAX_PROMPT_TOKENS` as the hard full-request ceiling for extraction and refinement estimates
- uses full extraction and refinement request estimation to decide boundaries
- chunks at transcript item boundaries rather than slicing message content
- carries overlap forward into the next chunk

## Extraction

Implemented in [`app/compaction/extractor.py`](/home/jesse/src/coding-agent-router/app/compaction/extractor.py).

Behavior:

- calls the compactor model with schema-constrained structured output
- uses a strict Responses-compatible JSON schema where object nodes are closed with `additionalProperties: false`
- represents `repo_state` facts as key/value entry arrays in model output, then normalizes them back into runtime dictionaries
- sends a compact ordered event stream instead of the original verbose transcript JSON
- preserves chronology while stripping tool-call wrapper noise such as `type`, `id`, `yield_time_ms`, and `max_output_tokens`
- encodes empty `write_stdin` polls as compact `poll` events instead of empty input blobs
- supplies a strict extraction system prompt
- estimates full request size with a tokenizer-based prompt counter
- keeps `COMPACTOR_NUM_CTX` as the normal request context but may burst to `COMPACTOR_BURST_NUM_CTX`
- keeps the estimated full extraction request at or under `COMPACTOR_MAX_PROMPT_TOKENS`
- avoids requests that cannot fit an estimated safe request budget within context
- validates responses into `ChunkExtraction`

Prompt source:

- [`compaction_extraction_system.md`](/home/jesse/src/coding-agent-router/app/prompts/compaction_extraction_system.md)

## Merge

Implemented in [`app/compaction/merger.py`](/home/jesse/src/coding-agent-router/app/compaction/merger.py).

Behavior:

- latest non-empty scalar wins
- repo-state dictionaries are shallow-merged with later values overwriting earlier values
- list fields are deduplicated case-insensitively while prioritizing newer entries
- latest non-empty plan list wins

## Refinement

Implemented in [`app/compaction/refiner.py`](/home/jesse/src/coding-agent-router/app/compaction/refiner.py).

Behavior:

- takes deterministic merged state as the source of truth
- extracts recent-turn durable state and merges it deterministically onto the existing merged state
- sends recent raw turns to the refiner as the same compact ordered event stream used by extraction
- estimates full request size with a tokenizer-based prompt counter
- keeps the estimated full refinement request at or under `COMPACTOR_MAX_PROMPT_TOKENS`
- constrains refinement output with a strict recent-state extraction schema instead of a diff/patch schema
- represents `repo_state` facts as key/value entry arrays in model output, then normalizes them back into runtime dictionaries
- caps each refinement iteration to about `8000` tokens of recent raw context even if the prompt budget could fit more
- may skip refinement entirely when the base merged-state prompt already exceeds refinement budget
- skips oversize recent-raw items that still cannot fit into a refinement iteration
- rejects malformed or non-JSON model output
- merges accepted recent-state extraction fields deterministically onto merged state

Prompt source:

- [`compaction_refinement_system.md`](/home/jesse/src/coding-agent-router/app/prompts/compaction_refinement_system.md)

## Spark Fallback

If local inline compaction fails or the rendered Codex handoff is invalid, the companion reruns compaction on `CODEX_SPARK_MODEL` through the same chunked extraction and refinement pipeline. Spark therefore sees the same compact event-stream chunks and bounded refinement windows as the local compactor, not the full original inline-compaction request replayed as one upstream call. The Spark compaction client uses the upstream streaming `/responses` contract and reconstructs the final completed response from SSE events.

## Persistence

Artifacts are written under `COMPACTION_STATE_DIR/<session_id>/`.

Persisted files include:

- chunk extraction JSON
- `merged-state.json`
- `refined-state.json`
- `handoff.json`
- `TASK_STATE.md`
- `DECISIONS.md`
- `FAILURES_TO_AVOID.md`
- `NEXT_STEPS.md`
- `SESSION_HANDOFF.md`

## Codex Handoff Validation

Before a compacted handoff is rendered for Codex, the runtime validates that:

- the handoff still matches the `SessionHandoff`-compatible structured schema
- the structured handoff and recent raw turns remain JSON-serializable
- each durable-memory entry still has a non-empty filename

If a stored handoff cannot be loaded or rendered safely for the local Codex CLI lane, the router skips the compacted handoff and falls back to the plain `system + prompt` path instead of crashing the request.
