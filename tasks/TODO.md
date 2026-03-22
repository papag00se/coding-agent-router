# TODO

## Status Legend
- `[ ]` pending
- `[-]` in progress
- `[x]` done
- `[b]` blocked

## Queue

- `[x]` `ARC-1` Convert compaction mode into an app-server-only integration point
  - Scope: remove ordinary-turn execution from the compaction service design and make `thread/compact/start` the only interception target.
  - Depends on: none
  - Done when:
    - the intended architecture is documented in code/task state
    - the compaction service no longer depends on nested `codex exec` for ordinary extension turns
    - the extension integration point is explicitly the app-server protocol

- `[x]` `ARC-2` Keep normal extension turns on native Codex/OpenAI
  - Scope: ensure ordinary turns bypass local routing in compaction mode and continue in the original Codex session.
  - Depends on: `ARC-1`
  - Done when:
    - normal extension turns do not spawn nested Codex sessions
    - normal extension turns do not route to local reasoner/coder in compaction mode
    - ordinary turn progress remains native to the original session

- `[b]` `ARC-3` Replace only the compaction step with local compaction
  - Scope: on `thread/compact/start`, run the local compactor and feed reduced context back into the same app-server thread.
  - Depends on: `ARC-1`, `ARC-2`
  - Done when:
    - a real compaction trigger stays inside the same thread
    - local compaction artifacts are used for post-compaction context
    - the original thread continues after compaction without flattened proxy output

- `[b]` `ARC-4` Preserve native feedback and session semantics
  - Scope: make sure item lifecycle, thread ids, turn ids, and progress UI behave like native Codex while compaction is intercepted.
  - Depends on: `ARC-3`
  - Done when:
    - the extension shows native in-session progress during and after compaction
    - `contextCompaction` / `thread/compacted` semantics remain intact
    - no reconnect or decode-body regressions are introduced

- `[b]` `ARC-5` Add deterministic tests for the app-server compaction path
  - Scope: add regression tests for normal-turn passthrough, compaction interception, and post-compaction follow-up behavior.
  - Depends on: `ARC-2`, `ARC-3`, `ARC-4`
  - Done when:
    - tests fail if ordinary turns start spawning nested sessions again
    - tests fail if compaction stops using the local compactor
    - tests fail if post-compaction turns lose session continuity

- `[b]` `ARC-6` Validate the extension path end to end
  - Scope: prove the real VS Code extension path behaves correctly under a long session that reaches compaction.
  - Depends on: `ARC-5`
  - Done when:
    - a real extension session reaches compaction
    - local compaction is confirmed from logs
    - the session continues afterward in the same thread

- `[b]` `ARC-7` Add app-server local reasoning for narrowly safe non-tool turns
  - Scope: allow specific analysis/reasoning turns to route to `local_reasoner` while keeping tool-heavy or stateful turns on native Codex.
  - Depends on: `ARC-6`
  - Done when:
    - routing rules are explicit and narrow
    - no fake tool text leaks into the session
    - tests cover the allowed and disallowed turn classes

- `[b]` `ARC-8` Add app-server local coder for carefully constrained coding turns
  - Scope: support local coding turns only after protocol, tool, and session continuity rules are proven safe.
  - Depends on: `ARC-7`
  - Done when:
    - local coder turns preserve tool semantics or are explicitly limited away from unsafe tool flows
    - the original session still shows native progress
    - tests cover the coder routing boundary and failure mode
