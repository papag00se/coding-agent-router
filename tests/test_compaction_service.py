from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.compaction.models import ChunkExtraction
from app.compaction.service import CompactionService
from app.compaction.storage import CompactionStorage
from app.task_metrics import estimate_tokens

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "codex_support_transcript_large.json"


class _FakeExtractor:
    def __init__(self) -> None:
        self.model = "qwen-test"
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


class _FakeRefiner:
    def __init__(self) -> None:
        self.model = "qwen-test"
        self.calls = []

    def refine_state(self, state, recent_raw_turns, *, current_request, repo_context=None):
        self.calls.append(
            {
                "state": state,
                "recent_raw_turns": recent_raw_turns,
                "current_request": current_request,
                "repo_context": repo_context,
            }
        )
        if recent_raw_turns:
            state.objective = state.objective or current_request
            state.latest_plan = [current_request]
        return state


class TestCompactionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.large_items = __import__("json").loads(FIXTURE_PATH.read_text())

    def test_compact_transcript_writes_handoff_and_memory(self):
        extractor = _FakeExtractor()
        refiner = _FakeRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, refiner=refiner, storage=CompactionStorage(Path(tmpdir)))
            items = [{"role": "user", "content": "x" * 4000} for _ in range(4)]
            with patch(
                "app.compaction.service.settings",
                SimpleNamespace(
                    compactor_keep_raw_tokens=500,
                    compactor_target_chunk_tokens=1200,
                    compactor_max_chunk_tokens=1600,
                    compactor_overlap_tokens=200,
                    compactor_num_ctx=4000,
                    compactor_response_headroom_tokens=512,
                ),
            ):
                handoff = service.compact_transcript(
                    "session-1",
                    items,
                    current_request="finish rename",
                    repo_context={"repo": "coding-agent-router"},
                )

            self.assertTrue(extractor.calls)
            self.assertEqual(handoff.stable_task_definition, f"objective {len(extractor.calls)}")
            self.assertEqual(handoff.current_request, "finish rename")
            self.assertTrue(refiner.calls)
            self.assertTrue((Path(tmpdir) / "session-1" / "handoff.json").exists())
            self.assertTrue((Path(tmpdir) / "session-1" / "TASK_STATE.md").exists())
            self.assertTrue((Path(tmpdir) / "session-1" / "refined-state.json").exists())

    def test_refresh_if_needed_uses_existing_handoff_for_small_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CompactionStorage(Path(tmpdir))
            extractor = _FakeExtractor()
            refiner = _FakeRefiner()
            service = CompactionService(extractor=extractor, refiner=refiner, storage=storage)
            with patch(
                "app.compaction.service.settings",
                SimpleNamespace(
                    compactor_keep_raw_tokens=500,
                    compactor_target_chunk_tokens=1200,
                    compactor_max_chunk_tokens=1600,
                    compactor_overlap_tokens=200,
                    compactor_num_ctx=4000,
                    compactor_response_headroom_tokens=512,
                ),
            ):
                service.compact_transcript(
                    "session-2",
                    [{"role": "user", "content": "x" * 4000} for _ in range(4)],
                    current_request="rename",
                )
                loaded = service.refresh_if_needed("session-2", [{"role": "user", "content": "short"}], current_request="rename")

            self.assertEqual(loaded.current_request, "rename")
            self.assertGreaterEqual(len(extractor.calls), 1)

    def test_load_latest_handoff_round_trips_after_compaction(self):
        extractor = _FakeExtractor()
        refiner = _FakeRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, refiner=refiner, storage=CompactionStorage(Path(tmpdir)))
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
        refiner = _FakeRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, refiner=refiner, storage=CompactionStorage(Path(tmpdir)))
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

    def test_compact_transcript_keeps_latest_item_raw_and_out_of_extractions(self):
        extractor = _FakeExtractor()
        refiner = _FakeRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, refiner=refiner, storage=CompactionStorage(Path(tmpdir)))
            handoff = service.compact_transcript(
                "session-raw-tail",
                [
                    {"role": "user", "content": "older request"},
                    {"role": "assistant", "content": "older reply"},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "latest tool output"}]},
                ],
                current_request="older request",
            )

            extracted_items = [item for chunk, _repo_context in extractor.calls for item in chunk.items]

        self.assertEqual(handoff.recent_raw_turns[-1]["content"][0]["type"], "tool_result")
        self.assertNotIn(handoff.recent_raw_turns[-1], extracted_items)

    def test_compact_transcript_does_not_fallback_to_compacting_latest_oversize_item(self):
        extractor = _FakeExtractor()
        refiner = _FakeRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, refiner=refiner, storage=CompactionStorage(Path(tmpdir)))
            handoff = service.compact_transcript(
                "session-oversize-tail",
                [{"role": "user", "content": "x" * 80000}],
                current_request="rename",
            )

        self.assertEqual(extractor.calls, [])
        self.assertEqual(handoff.recent_raw_turns, [{"role": "user", "content": "x" * 80000}])

    def test_compact_transcript_refines_recent_raw_turns_in_multiple_passes(self):
        extractor = _FakeExtractor()
        refiner = _FakeRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, refiner=refiner, storage=CompactionStorage(Path(tmpdir)))
            with patch(
                "app.compaction.service.settings",
                SimpleNamespace(
                    compactor_keep_raw_tokens=4000,
                    compactor_target_chunk_tokens=300,
                    compactor_max_chunk_tokens=600,
                    compactor_overlap_tokens=0,
                    compactor_num_ctx=1200,
                    compactor_response_headroom_tokens=128,
                ),
            ):
                handoff = service.compact_transcript(
                    "session-refine-passes",
                    [
                        {"role": "user", "content": "older compactable context " + ("c" * 1200)},
                        {"role": "assistant", "content": "recent raw turn one " + ("a" * 1200)},
                        {"role": "user", "content": "recent raw turn two " + ("b" * 1200)},
                        {"role": "assistant", "content": "newest raw preserved"},
                    ],
                    current_request="finish rename",
                )

        self.assertGreaterEqual(len(refiner.calls), 2)
        self.assertEqual(handoff.recent_raw_turns[-1], {"role": "assistant", "content": "newest raw preserved"})

    def test_compact_transcript_preserves_prompt_oversize_items_raw(self):
        extractor = _FakeExtractor()
        refiner = _FakeRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, refiner=refiner, storage=CompactionStorage(Path(tmpdir)))
            with patch(
                "app.compaction.service.settings",
                SimpleNamespace(
                    compactor_keep_raw_tokens=0,
                    compactor_target_chunk_tokens=1200,
                    compactor_max_chunk_tokens=1600,
                    compactor_overlap_tokens=0,
                    compactor_num_ctx=2000,
                    compactor_response_headroom_tokens=256,
                ),
            ), patch(
                "app.compaction.service.estimate_extraction_request_tokens",
                side_effect=lambda chunk, repo_context=None, model=None: 2600 if any(item["content"] == "oversize" for item in chunk.items) else 800,
            ):
                handoff = service.compact_transcript(
                    "session-prompt-oversize",
                    [
                        {"role": "user", "content": "small"},
                        {"role": "assistant", "content": "oversize"},
                        {"role": "user", "content": "small-two"},
                        {"role": "assistant", "content": "latest raw"},
                    ],
                    current_request="continue",
                )

        extracted_contents = [item["content"] for chunk, _repo_context in extractor.calls for item in chunk.items]
        self.assertEqual(extracted_contents, ["small", "small-two"])
        self.assertIn({"role": "assistant", "content": "oversize"}, handoff.recent_raw_turns)

    def test_compact_transcript_skips_refinement_items_that_cannot_fit_minimum_output_budget(self):
        extractor = _FakeExtractor()
        refiner = _FakeRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = CompactionService(extractor=extractor, refiner=refiner, storage=CompactionStorage(Path(tmpdir)))
            with patch(
                "app.compaction.service.settings",
                SimpleNamespace(
                    compactor_keep_raw_tokens=24,
                    compactor_target_chunk_tokens=1200,
                    compactor_max_chunk_tokens=1600,
                    compactor_overlap_tokens=0,
                    compactor_num_ctx=2000,
                    compactor_response_headroom_tokens=512,
                ),
            ), patch(
                "app.compaction.service.estimate_refinement_request_tokens",
                side_effect=lambda state, recent_raw_turns, current_request, repo_context=None, model=None: (
                    1800 if recent_raw_turns and recent_raw_turns[0]["content"] == "oversize raw turn" else 900
                ),
            ):
                handoff = service.compact_transcript(
                    "session-refine-oversize",
                    [
                        {"role": "user", "content": "compactable context " + ("x" * 1200)},
                        {"role": "assistant", "content": "oversize raw turn"},
                        {"role": "user", "content": "small raw turn"},
                        {"role": "assistant", "content": "latest raw"},
                    ],
                    current_request="continue",
                )

        self.assertEqual(len(refiner.calls), 1)
        self.assertEqual(
            [item["content"] for item in refiner.calls[0]["recent_raw_turns"]],
            ["small raw turn", "latest raw"],
        )
        self.assertIn({"role": "assistant", "content": "oversize raw turn"}, handoff.recent_raw_turns)
