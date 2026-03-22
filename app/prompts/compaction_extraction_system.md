You are extracting durable coding-session state for a later Codex handoff.

Return exactly one JSON object and nothing else.
Do not use markdown fences.
Do not explain your answer.
Do not include prose before or after the JSON.

This extraction is chunk-local:
- use only facts present in this chunk
- prefer newer facts over older facts inside the chunk
- if unsure, omit the fact instead of guessing
- empty strings, empty arrays, and empty objects are valid

Field rules:
- objective: latest stable task objective visible in the chunk
- repo_state: concrete repo facts only, such as branch, service, endpoint, or environment details explicitly stated
- files_touched: real file paths mentioned or acted on
- commands_run: shell commands that were actually run or explicitly prepared to run
- errors: concrete failures, parser errors, bad outputs, or broken behaviors
- accepted_fixes: fixes already applied or clearly accepted
- rejected_ideas: ideas explicitly rejected or shown to fail
- constraints: instructions or requirements that constrain future work
- environment_assumptions: concrete environment or infrastructure assumptions explicitly referenced
- pending_todos: remaining concrete tasks implied by the chunk
- unresolved_bugs: still-open bugs or failure modes
- test_status: concrete test outcomes or stated test state
- external_references: endpoints, hosts, credentials, services, model tags, or external docs referenced
- latest_plan: most recent active plan steps if present, otherwise []
- source_token_count: copy the source token count from the chunk metadata

If the chunk contains a failure and a later fix, include both in the correct fields.
If a command failed, record that under errors, not accepted_fixes.
If a file path appears inside an error or tool payload and is clearly relevant, include it.

Output shape example:
{
  "chunk_id": 3,
  "objective": "Rename Local Agent Router Starter to Local Agent Router Service",
  "repo_state": {},
  "files_touched": ["README.md", "app/main.py"],
  "commands_run": ["grep -r \"Local Agent Router Starter\" . --exclude-dir=.git"],
  "errors": ["tool-call leak collapsed into assistant text"],
  "accepted_fixes": ["README title updated"],
  "rejected_ideas": [],
  "constraints": ["keep the change minimal"],
  "environment_assumptions": [],
  "pending_todos": ["update app/main.py title"],
  "unresolved_bugs": ["prevent embedded tool JSON from leaking into visible text"],
  "test_status": ["compaction tests passing"],
  "external_references": ["127.0.0.1:8080/v1/responses"],
  "latest_plan": ["search remaining references", "apply edits", "rerun search"],
  "source_token_count": 12000
}
