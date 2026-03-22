from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.compaction.models import ChunkExtraction
from app.compaction.service import CompactionService
from app.compaction.storage import CompactionStorage


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "codex_support_transcript.json"


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = []

    def extract_chunk(self, chunk, repo_context=None):
        self.calls.append((chunk, repo_context))
        return ChunkExtraction(
            chunk_id=chunk.chunk_id,
            objective="Rename Local Agent Router Starter to Local Agent Router Service",
            repo_state={"repo": (repo_context or {}).get("repo", "unknown")},
            files_touched=["README.md", "app/main.py", "tests/test_tool_adapter.py"],
            commands_run=[
                "grep -r \"Local Agent Router Starter\" . --exclude-dir=.git",
                "find . -type f -name \"*.md\" -o -name \"*.py\" | xargs grep -n \"Local Agent Router Starter\" 2>/dev/null",
            ],
            errors=["tool-call leak collapsed into assistant text"],
            accepted_fixes=["README title updated"],
            rejected_ideas=["one-shot full transcript summary"],
            constraints=["keep the change minimal", "update tests with behavior changes"],
            pending_todos=["update app/main.py title", "update tests/test_tool_adapter.py assertions"],
            unresolved_bugs=["prevent embedded tool JSON from leaking into visible text"],
            test_status=["compaction tests passing"],
            external_references=["127.0.0.1:8080/v1/responses"],
            latest_plan=["search remaining references", "apply edits", "rerun search"],
            source_token_count=chunk.token_count,
        )


class TestHandoffFlow(unittest.TestCase):
    def test_service_builds_ordered_codex_handoff_flow_from_fixture(self):
        items = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        extractor = _FakeExtractor()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, storage=CompactionStorage(Path(tmpdir)))
            service.compact_transcript(
                "fixture-session",
                items,
                current_request="Finish the remaining rename and rerun the search.",
                repo_context={"repo": "coding-agent-router"},
            )

            flow = service.build_codex_handoff_flow("fixture-session")

        self.assertIsNotNone(flow)
        self.assertEqual(
            [item["name"] for item in flow.durable_memory],
            [
                "TASK_STATE.md",
                "DECISIONS.md",
                "FAILURES_TO_AVOID.md",
                "NEXT_STEPS.md",
                "SESSION_HANDOFF.md",
            ],
        )
        self.assertEqual(
            flow.structured_handoff["stable_task_definition"],
            "Rename Local Agent Router Starter to Local Agent Router Service",
        )
        self.assertEqual(flow.current_request, "Finish the remaining rename and rerun the search.")
        self.assertEqual(flow.recent_raw_turns[-1]["content"], "Update app/main.py and tests/test_tool_adapter.py, then rerun the search.")
        self.assertIn("tool-call leak collapsed into assistant text", flow.structured_handoff["failures_to_avoid"])
