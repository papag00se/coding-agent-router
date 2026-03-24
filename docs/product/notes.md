# Product Notes

[Docs Index](../index.md) | [Product Index](./index.md)

## Claude Code Note

This repository includes a minimal `POST /v1/messages` endpoint so you can experiment with Claude-style traffic.

Important constraints:

- this project does not claim to be a perfect Claude Code gateway
- the endpoint is intentionally minimal and meant for local experimentation
- the direct `/invoke` path remains the most direct supported path

Recommended progression:

1. Get `/invoke` working
2. Verify local coder, local reasoner, and Codex CLI fallback behavior
3. Then experiment with putting Claude-style traffic in front of `/v1/messages`
