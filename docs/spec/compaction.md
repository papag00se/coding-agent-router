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
7. Apply bounded refinement patches over merged state using recent raw turns in `8000`-token iterations
8. Reattach the preserved newest raw turn
9. Render and persist durable handoff artifacts

## Normalization

Current normalization behavior:

- strips `encrypted_content` recursively
- removes attachment-like content blocks from compactable history
- drops historical `tool_result` and `function_call_output` blocks from compactable history
- strips the known Codex bootstrap `instructions` block for inline compaction requests
- preserves the newest top-level turn raw for later reattachment

If a compactable item still cannot fit inside extraction prompt budget, it is skipped from chunk extraction and carried forward raw instead.

## Chunking

Implemented in [`app/compaction/chunking.py`](/home/jesse/src/coding-agent-router/app/compaction/chunking.py).

Behavior:

- keeps a recent raw tail based on `COMPACTOR_KEEP_RAW_TOKENS`
- chunks older items by prompt budget rather than transcript-size heuristics alone
- uses full extraction and refinement request estimation to decide boundaries
- chunks at transcript item boundaries rather than slicing message content
- carries overlap forward into the next chunk

## Extraction

Implemented in [`app/compaction/extractor.py`](/home/jesse/src/coding-agent-router/app/compaction/extractor.py).

Behavior:

- calls the compactor model in Ollama JSON mode
- supplies a strict extraction system prompt
- estimates full request size with a tokenizer-based prompt counter
- keeps `COMPACTOR_NUM_CTX` as the normal request context but may burst to `COMPACTOR_BURST_NUM_CTX`
- refuses requests that cannot leave the minimum output budget for valid JSON
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
- requests a bounded patch rather than a full state rewrite
- estimates full request size with a tokenizer-based prompt counter
- caps each refinement iteration to about `8000` tokens of `recent_raw_turns` even if the prompt budget could fit more
- may skip refinement entirely when the base merged-state prompt already exceeds refinement budget
- skips oversize recent-raw items that still cannot fit into a refinement iteration
- rejects malformed or non-JSON model output
- applies accepted patch fields deterministically onto merged state

Prompt source:

- [`compaction_refinement_system.md`](/home/jesse/src/coding-agent-router/app/prompts/compaction_refinement_system.md)

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
