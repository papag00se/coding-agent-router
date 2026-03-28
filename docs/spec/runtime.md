# Runtime Specification

[Docs Index](../index.md) | [Specification Index](./index.md)

## Runtime Topology

Primary components:

1. API server in [`app/main.py`](/home/jesse/src/coding-agent-router/app/main.py) and [`app/compaction_main.py`](/home/jesse/src/coding-agent-router/app/compaction_main.py)
2. Routing service in [`app/router.py`](/home/jesse/src/coding-agent-router/app/router.py)
3. Backend clients in [`app/clients/ollama_client.py`](/home/jesse/src/coding-agent-router/app/clients/ollama_client.py) and [`app/clients/codex_client.py`](/home/jesse/src/coding-agent-router/app/clients/codex_client.py)
4. Compatibility layer in [`app/compat.py`](/home/jesse/src/coding-agent-router/app/compat.py)
5. Tool translation layer in [`app/tool_adapter.py`](/home/jesse/src/coding-agent-router/app/tool_adapter.py) and [`app/tool_surface.py`](/home/jesse/src/coding-agent-router/app/tool_surface.py)
6. Transcript compaction subsystem under [`app/compaction/`](/home/jesse/src/coding-agent-router/app/compaction)
7. Prompt loading and rendering in [`app/prompt_loader.py`](/home/jesse/src/coding-agent-router/app/prompt_loader.py)

`OllamaClient` serializes in-flight calls per `base_url` with a shared file lock, so router and compaction processes queue behind the same local Ollama service instead of hitting it concurrently.

## High-Level Flow

```text
client
  -> compatibility layer / native invoke
  -> routing digest + route selection
  -> local_coder | local_reasoner | codex_cli
  -> compatibility response mapping
```

## Service Modes

### Full Router

Used when the service terminates client traffic and executes locally.

### Compaction Companion

Used when inline local compaction should run locally while ordinary `/v1/responses` traffic is proxied upstream.

The compaction companion also owns the selective Spark and mini passthrough rewrite for qualifying non-compaction Responses requests.
Inline compaction uses the same Spark breaker and can drop to mini when the Spark lane is blocked or unavailable.

## Prompt Files

Prompt files live under [`app/prompts`](/home/jesse/src/coding-agent-router/app/prompts).

Static prompt files include router, coder, reasoner, and compactor instructions. Dynamic templates include:

- [`codex_support_prompt.md`](/home/jesse/src/coding-agent-router/app/prompts/codex_support_prompt.md)
- [`compacted_flow.md`](/home/jesse/src/coding-agent-router/app/prompts/compacted_flow.md)

`render_prompt` validates only placeholders that exist in the template itself. Literal `{{...}}` sequences inside replacement content are preserved as plain text, which keeps code snippets and handoff payloads from tripping template validation.
