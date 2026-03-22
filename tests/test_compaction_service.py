from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.compaction.models import ChunkExtraction
from app.compaction.service import CompactionService
from app.compaction.storage import CompactionStorage
from app.task_metrics import estimate_tokens

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "codex_support_transcript_large.json"


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = []

    def extract_chunk(self, chunk, repo_context=None):
        self.calls.append((chunk, repo_context))
        return ChunkExtraction(
            chunk_id=chunk.chunk_id,
            objective=f"objective {chunk.chunk_id}",
            files_touched=[f"file-{chunk.chunk_id}.py"],
            accepted_fixes=[f"fix {chunk.chunk_id}"],
            pending_todos=[f"todo {chunk.chunk_id}"],
            latest_plan=[f"plan {chunk.chunk_id}"],
            source_token_count=chunk.token_count,
        )


class TestCompactionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.large_items = __import__("json").loads(FIXTURE_PATH.read_text())

    def test_compact_transcript_writes_handoff_and_memory(self):
        extractor = _FakeExtractor()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, storage=CompactionStorage(Path(tmpdir)))
            items = [{"role": "user", "content": "x" * 4000} for _ in range(4)]

            handoff = service.compact_transcript(
                "session-1",
                items,
                current_request="finish rename",
                repo_context={"repo": "coding-agent-router"},
            )

            self.assertTrue(extractor.calls)
            self.assertEqual(handoff.stable_task_definition, f"objective {len(extractor.calls)}")
            self.assertEqual(handoff.current_request, "finish rename")
            self.assertTrue((Path(tmpdir) / "session-1" / "handoff.json").exists())
            self.assertTrue((Path(tmpdir) / "session-1" / "TASK_STATE.md").exists())

    def test_refresh_if_needed_uses_existing_handoff_for_small_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CompactionStorage(Path(tmpdir))
            extractor = _FakeExtractor()
            service = CompactionService(extractor=extractor, storage=storage)

            service.compact_transcript("session-2", [{"role": "user", "content": "x" * 4000}], current_request="rename")
            loaded = service.refresh_if_needed("session-2", [{"role": "user", "content": "short"}], current_request="rename")

            self.assertEqual(loaded.current_request, "rename")
            self.assertEqual(len(extractor.calls), 1)

    def test_load_latest_handoff_round_trips_after_compaction(self):
        extractor = _FakeExtractor()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, storage=CompactionStorage(Path(tmpdir)))
            service.compact_transcript("session-3", [{"role": "user", "content": "x" * 4000}], current_request="rename")
            loaded = service.load_latest_handoff("session-3")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.current_request, "rename")

    def test_large_fixture_is_big_enough_to_force_multiple_chunks(self):
        chars = FIXTURE_PATH.stat().st_size
        approx_tokens = estimate_tokens(self.large_items)

        self.assertGreater(chars, 200_000)
        self.assertGreater(approx_tokens, 40_000)

    def test_compact_transcript_handles_large_fixture_without_llm(self):
        extractor = _FakeExtractor()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, storage=CompactionStorage(Path(tmpdir)))
            handoff = service.compact_transcript(
                "large-session",
                self.large_items,
                current_request="Finish the remaining router stabilization tasks.",
                repo_context={"repo": "coding-agent-router"},
            )

            self.assertGreater(len(extractor.calls), 2)
            self.assertEqual(handoff.current_request, "Finish the remaining router stabilization tasks.")
            self.assertEqual(handoff.stable_task_definition, f"objective {len(extractor.calls)}")
            self.assertIn(f"fix {len(extractor.calls)}", handoff.key_decisions)
            self.assertTrue(handoff.recent_raw_turns)
            self.assertIn("Iteration 180", handoff.recent_raw_turns[-1]["content"])
            self.assertTrue((Path(tmpdir) / "large-session" / "handoff.json").exists())
            self.assertTrue((Path(tmpdir) / "large-session" / "SESSION_HANDOFF.md").exists())
