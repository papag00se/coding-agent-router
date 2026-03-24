# Product Requirements

[Docs Index](../index.md) | [Product Index](./index.md)

## Goals

- Provide one local service boundary for coding and reasoning traffic
- Route work across local coding, local reasoning, and optional Codex CLI execution
- Preserve compatibility with several upstream request formats
- Support tool-aware coding requests on the local coder lane
- Preserve session continuity through durable transcript compaction
- Allow ordinary Responses traffic to keep flowing upstream while local inline compaction is intercepted

## Functional Requirements

- The service accepts native router requests through `/invoke`
- The service accepts Anthropic Messages, OpenAI Chat, OpenAI Responses, and Ollama Chat compatibility requests
- The only execution routes are `local_coder`, `local_reasoner`, and `codex_cli`
- Callers can override route selection with a preferred backend
- Local backends are excluded when their configured context windows cannot fit the request
- The router bypasses route-model inference when the routing digest itself is too large
- The local coder lane preserves structured tool calls across protocol boundaries
- The Responses surface exposes an alias tool set and maps it back to original tool names
- Long transcripts are compacted into persisted durable memory plus a structured handoff
- The compaction companion can intercept inline compaction locally while proxying ordinary `/v1/responses` traffic upstream
- Qualifying non-compaction `gpt-5.4` passthrough calls can be rewritten to Codex Spark based on request shape, request size, and rollout rate

## Non-Functional Requirements

- Local-first operation with Codex CLI as an optional extra lane
- Environment-variable configuration and simple systemd-friendly deployment
- Restart-safe on-disk compaction artifacts
- Compact route selection rather than full-history route prompts
- Deterministic fallback behavior when route inference is malformed
- Prompt-aware compaction sizing with explicit output headroom and bounded context bursting

## Known Gaps

- Real token streaming parity does not exist across every protocol shape
- Some config variables remain present but dormant
- Tool handling is lane-specific rather than universal
- Compaction output quality still depends on the compactor model, even though the pipeline is more structurally constrained than before
