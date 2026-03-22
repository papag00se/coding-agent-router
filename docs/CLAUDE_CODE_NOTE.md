# Claude Code note

This starter includes a minimal `POST /v1/messages` endpoint so you can later experiment with putting it in front of Claude-style traffic.

Important:

- this project does **not** claim to be a perfect drop-in Claude Code gateway
- this endpoint is intentionally minimal and meant for local experimentation
- the direct `/invoke` path is the supported path for the starter bundle

Recommended progression:

1. get `/invoke` working
2. verify local coder / local reasoner / cloud fallback
3. then experiment with placing Claude Code in front of `/v1/messages`
