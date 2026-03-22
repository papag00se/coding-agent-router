# USER_ACTIONS_CHECKLIST

## App-Server Compaction Blocker

- Decide whether to allow unsupported reverse engineering of the native Codex app-server thread state / persisted rollout format in order to inject local-compacted history into the same live thread.
- If the answer is `no`, the supported architecture stops at:
  - native app-server passthrough for normal turns
  - no local replacement of native compaction
- If the answer is `yes`, the next work item is a separate unsupported investigation task against Codex internals rather than normal router implementation work.
