You are refining merged durable coding-session state for a later Codex handoff.

Return exactly one JSON object and nothing else.
Do not use markdown fences.
Do not explain your answer.
Do not include prose before or after the JSON.

This is a constrained refinement pass:
- start from current_state as the source of truth
- use recent_raw_turns only to reprioritize, dedupe, clarify, or add facts that are explicitly visible there
- prefer newer facts over older facts
- never invent facts, file paths, commands, errors, or plans
- if unsure, keep the existing current_state value or omit the fact
- empty strings, empty arrays, and empty objects are valid

Refinement goals:
- keep the latest stable objective
- promote the most recent active plan into latest_plan
- keep unresolved work current and deduped
- keep failures and rejected ideas distinct
- keep repo_state concrete and factual
- preserve merged_chunk_count from current_state

Field rules:
- objective: latest stable task objective
- repo_state: concrete repo facts only
- files_touched: real file paths mentioned or acted on
- commands_run: shell commands actually run or explicitly prepared to run
- errors: concrete failures, parser errors, bad outputs, or broken behaviors
- accepted_fixes: fixes already applied or clearly accepted
- rejected_ideas: ideas explicitly rejected or shown to fail
- constraints: instructions or requirements that constrain future work
- environment_assumptions: concrete environment or infrastructure assumptions explicitly referenced
- pending_todos: remaining concrete tasks
- unresolved_bugs: still-open bugs or failure modes
- test_status: concrete test outcomes or stated test state
- external_references: endpoints, hosts, credentials, services, model tags, or external docs referenced
- latest_plan: most recent active plan steps if present, otherwise []
- merged_chunk_count: copy from current_state unless the payload explicitly says otherwise
