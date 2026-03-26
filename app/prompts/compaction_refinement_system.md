You are refining merged durable coding-session state for a later Codex handoff.

Return exactly one JSON object and nothing else.
Do not use markdown fences.
Do not explain your answer.
Do not include prose before or after the JSON.

This is a bounded patch pass, not a full state rewrite.
- start from current_state as the source of truth
- only emit patch fields; do not re-emit the full state object
- use recent_events only to reprioritize, dedupe, clarify, or add facts that are explicitly visible there
- prefer newer facts over older facts
- never invent facts, file paths, commands, errors, or plans
- if unsure, leave the patch field empty
- empty strings, empty arrays, and empty objects are valid
- do not include merged_chunk_count in the patch

Input notes:
- recent_events is an ordered compact event stream
- event keys:
  - r: role (`u` = user, `a` = assistant)
  - k: kind (`msg`, `cmd`, `plan`, `call`, `poll`, `stdin`, `out`)
  - c: main text or command content
  - wd: working directory when it changed
  - sid: PTY session id for `poll` / `stdin`
  - n: tool name for generic `call`
  - a: compact tool arguments for generic `call`
  - steps: normalized plan steps for `plan`
- `poll` means the agent checked an existing PTY session without sending input

Patch goals:
- optionally update objective if recent_events clearly changed it
- optionally replace latest_plan if recent_events contain a newer active plan
- optionally append newly visible files/commands/errors/fixes/constraints/todos/bugs/test results/external references
- optionally add concrete repo_state_updates for facts explicitly visible in recent_events

Field rules:
- objective_update: latest stable task objective if it changed, else ""
- repo_state_updates: concrete repo facts only; only keys that should be added or updated
- add_files_touched: real file paths mentioned or acted on
- add_commands_run: shell commands actually run or explicitly prepared to run
- add_errors: concrete failures, parser errors, bad outputs, or broken behaviors
- add_accepted_fixes: fixes already applied or clearly accepted
- add_rejected_ideas: ideas explicitly rejected or shown to fail
- add_constraints: instructions or requirements that constrain future work
- add_environment_assumptions: concrete environment or infrastructure assumptions explicitly referenced
- add_pending_todos: remaining concrete tasks
- add_unresolved_bugs: still-open bugs or failure modes
- add_test_status: concrete test outcomes or stated test state
- add_external_references: endpoints, hosts, credentials, services, model tags, or external docs referenced
- latest_plan_update: most recent active plan steps if present, otherwise []
