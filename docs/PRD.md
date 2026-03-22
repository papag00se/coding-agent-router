# Product Requirements Document

## Scope

This PRD is reverse engineered from the implementation in `app/`, `scripts/`, `deploy/`, and `tests/`. It describes the product that currently exists, the requirements that are clearly implemented, and the requirements that are only partial or implied.

## Product Name

Coding Agent Router

## Problem Statement

Users running coding agents locally need a single gateway that can:

- accept traffic from multiple client API shapes
- choose an execution backend appropriate to the task
- preserve long-running coding context without resending the entire transcript
- optionally escalate into Codex CLI when local routing or local context windows are not enough
- fit into Codex app-server style workflows

## Target Users

### Local-First AI Operator

Needs to split work across small local models and minimize cloud dependence.

### Coding-Agent Integrator

Needs one endpoint that can sit in front of coding clients speaking Anthropic, OpenAI, or Ollama-like protocols.

### Long-Running Session Operator

Needs transcript state to survive large histories and thread restarts.

## Product Goals

- Provide one local service boundary for coding and reasoning traffic.
- Separate local coding and reasoning lanes behind a routing policy.
- Preserve compatibility with several upstream request formats.
- Support tool-aware coding requests where the backend can produce structured tool calls.
- Preserve session continuity through durable transcript compaction.
- Fit into Codex app-server flows, including explicit thread compaction.
- Allow an operator to keep ordinary Responses traffic pointed at upstream while intercepting local compaction behavior.

## Non-Goals

- Perfect protocol parity with Claude Code or any hosted provider
- Multi-tenant hosting
- Authentication and billing
- Rich observability, tracing, or cost accounting
- Dynamic discovery of all actual backend models
- A built-in tool execution engine
- Full failover orchestration across providers

## Functional Requirements

### FR1. Multi-Protocol Request Ingress

The product must accept:

- native router requests through `/invoke`
- Anthropic Messages through `/v1/messages`
- OpenAI Chat Completions through `/v1/chat/completions`
- OpenAI Responses through `/v1/responses`
- Ollama Chat through `/api/chat`

Status: Implemented.

### FR2. Deterministic Route Set

The product must route work only to:

- `local_coder`
- `local_reasoner`
- `codex_cli`

Status: Implemented.

### FR3. Preferred Backend Override

The caller must be able to bypass route selection by specifying a preferred backend.

Status: Implemented.

### FR4. Context-Aware Eligibility

The routing layer must exclude local backends whose configured context windows are too small for the request.

Status: Implemented.

### FR5. Router Payload Guardrail

If the routing digest itself exceeds router context, the service must bypass router-model inference and choose a safe fallback route.

Status: Implemented.

### FR6. Local Coding Lane

The system must support a dedicated local coding backend with its own base URL, model, timeout, context, and temperature settings.

Status: Implemented.

### FR7. Local Reasoning Lane

The system must support a dedicated local reasoning backend with independent configuration.

Status: Implemented.

### FR8. Optional Codex CLI Escalation

The system must optionally run Codex CLI as a subprocess and pass repo-local working directory context.

Status: Implemented.

### FR9. Tool-Aware Coder Responses

When the request includes tools and the chosen route is the local coder, the service must preserve structured tool calls across protocol boundaries.

Status: Implemented for the local coder lane.

### FR10. Tool Surface Translation for Responses API

When serving the OpenAI Responses API, the system must be able to expose a smaller alias tool surface and map returned tool calls back to original tool names.

Status: Implemented.

### FR11. Progressive Responses Streaming

The product must support progressive SSE output for the OpenAI Responses API in the full router path.

Status: Implemented.

### FR12. App-Server Thread Lifecycle

The product must support starting threads, running turns, compacting threads, and persisting thread state for app-server clients.

Status: Implemented.

### FR13. Transcript Compaction

The product must compact long transcripts into persisted durable memory plus a structured handoff.

Status: Implemented.

### FR14. Codex-Oriented Handoff Reconstruction

For sessions with compaction state, the product must be able to reconstruct a Codex-ready prompt containing durable memory, structured handoff, recent raw turns, and the current request.

Status: Implemented.

### FR15. Compaction Companion Mode

The product must support a mode where:

- app-server traffic is local
- inline compaction requests are local
- ordinary `/v1/responses` traffic is proxied upstream
- other compatibility endpoints are explicitly rejected

Status: Implemented.

### FR16. Disk Persistence

The product must persist:

- app-server thread state
- chunk extractions
- merged compaction state
- durable memory markdown
- handoff JSON
- transport logs

Status: Implemented.

### FR17. File-Backed Application Prompts

The product must keep application prompts as separate files under `app/prompts` rather than embedding long prompt bodies directly in Python modules.

Status: Implemented.

### FR18. Template-Based Dynamic Prompt Rendering

The product must support dynamic prompt sections by explicit placeholder replacement instead of scattered inline string assembly.

Status: Implemented.

## Non-Functional Requirements

### NFR1. Local-First Operation

The default design should support fully local routing and local backend execution, with Codex CLI as an optional extra lane.

Status: Implemented.

### NFR2. Simple Deployment

The product should run as a small Python service with environment-variable configuration and basic systemd support.

Status: Implemented.

### NFR3. Stable On-Disk State

Thread and compaction artifacts should be restart-safe and reloadable from disk.

Status: Implemented.

### NFR4. Bounded Router Decision Cost

Route selection should operate on a compact digest rather than the full task payload.

Status: Implemented.

### NFR5. Clear Failure Behavior

If route inference is malformed, the product should fall back deterministically rather than fail closed.

Status: Implemented.

## Current Gaps Between Product Appearance and Product Reality

These are important requirements clarifications because they are easy to misunderstand from endpoint names or config names alone.

### GR1. Streaming Is Not Uniform

- `/v1/responses` is truly progressive in the full router
- `/v1/chat/completions` streaming is synthesized from a completed response
- `/api/chat` streaming is synthesized from a completed response

Requirement implication:

- if real token streaming parity is required across all protocols, that is not yet delivered

### GR2. Some Config Knobs Are Not Active Product Requirements Yet

The following settings exist but are not wired into active behavior:

- `APP_SERVER_MODE`
- `ENABLE_INCREMENTAL_COMPACTION`
- `COMPACTOR_MERGE_BATCH_SIZE`
- `DEFAULT_CLOUD_BACKEND`
- `FAIL_OPEN`

Requirement implication:

- these should be treated as future or abandoned product ideas, not current supported functionality

### GR3. Model Discovery Is Synthetic

`/v1/models` and `/api/tags` do not enumerate actual local backends. They expose compatibility placeholders.

Requirement implication:

- clients must not rely on these endpoints as authoritative backend inventory

### GR4. Tool Handling Is Lane-Specific

Structured tool handling is implemented only for the local coder path. The reasoner lane and Codex CLI lane do not expose equivalent structured tool semantics.

Requirement implication:

- clients needing tool-driven coding behavior should ensure requests land on `local_coder`

## Acceptance Criteria

The product should be considered functionally correct when:

- a native `/invoke` request can route and execute against the intended backend
- the same task can enter through Anthropic, OpenAI Chat, OpenAI Responses, or Ollama compatibility endpoints
- structured tool calls survive a local coder round trip
- route selection excludes backends that cannot fit the request context
- a long transcript can be compacted into durable memory and reloaded later
- app-server clients can resume a persisted thread after restart
- compaction-only mode can intercept inline compaction while proxying ordinary Responses traffic upstream

## Risks

- compatibility endpoints can be mistaken for full protocol implementations
- route selection quality depends on heuristic metrics and a router model returning valid JSON
- compaction fidelity depends on LLM extraction accuracy
- durable memory is useful but not authoritative ground truth
- synthetic model discovery can confuse clients that expect backend enumeration

## Requirement Backlog Suggested by the Implementation

These are the most obvious next requirements if the product is to mature beyond its current shape.

- make streaming semantics consistent across all compatibility endpoints
- either wire or delete dormant config toggles
- decide whether model discovery should remain synthetic or become truthful
- define explicit protocol-compatibility targets and test matrices
