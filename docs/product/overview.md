# Product Overview

[Docs Index](../index.md) | [Product Index](./index.md)

## Summary

This repository implements a local-first gateway for coding workflows. It accepts several client protocol shapes, routes work between local models and an optional Codex CLI lane, and preserves long-running session continuity through transcript compaction.

The product has two runtime modes:

- the full router in [`app/main.py`](/home/jesse/src/coding-agent-router/app/main.py)
- the compaction companion in [`app/compaction_main.py`](/home/jesse/src/coding-agent-router/app/compaction_main.py)

Both share the same routing core in [`app/router.py`](/home/jesse/src/coding-agent-router/app/router.py).

## Main Capabilities

- Multi-protocol ingress for native `/invoke`, Anthropic `/v1/messages`, OpenAI `/v1/chat/completions`, OpenAI `/v1/responses`, and Ollama `/api/chat`
- Deterministic routing across `local_coder`, `local_reasoner`, and `codex_cli`
- Local Ollama requests are serialized one at a time per service endpoint
- Structured tool handling on the local coder lane
- Durable transcript compaction with persisted handoff artifacts
- Compaction companion mode that intercepts inline local compaction while proxying ordinary Responses traffic upstream
- Selective passthrough rewrite from `gpt-5.4` to `gpt-5.3-codex-spark` or `gpt-5.4-mini` for qualifying non-compaction Responses calls, with image-bearing requests kept off the Spark lane
- Spark quota failover to `gpt-5.4-mini` with automatic re-probe after the recorded reset time, including inline compaction Spark fallback

## Product Boundary

This is not a general-purpose hosted gateway. The implemented product is best understood as a local coding-workflow router with durable compaction-backed handoff state.

## Current Limits

- Streaming behavior is not uniform across all compatibility endpoints
- Structured tool semantics are implemented primarily for the local coder lane
- Model discovery endpoints are synthetic compatibility surfaces, not authoritative inventory
- Compaction is structurally constrained, but durable-state quality still depends on the compactor model's extraction quality
