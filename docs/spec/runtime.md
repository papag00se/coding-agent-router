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

The compaction companion also owns the selective Codex Spark passthrough rewrite for qualifying non-compaction Responses requests.

## Prompt Files

Prompt files live under [`app/prompts`](/home/jesse/src/coding-agent-router/app/prompts).

Static prompt files include router, coder, reasoner, and compactor instructions. Dynamic templates include:

- [`codex_support_prompt.md`](/home/jesse/src/coding-agent-router/app/prompts/codex_support_prompt.md)
- [`compacted_flow.md`](/home/jesse/src/coding-agent-router/app/prompts/compacted_flow.md)
