from __future__ import annotations

import unittest

from app.compaction.models import ChunkExtraction, SessionHandoff, TranscriptChunk


class TestCompactionModels(unittest.TestCase):
    def test_chunk_extraction_defaults_are_structured(self):
        extraction = ChunkExtraction(chunk_id=1)

        self.assertEqual(extraction.chunk_id, 1)
        self.assertEqual(extraction.files_touched, [])
        self.assertEqual(extraction.repo_state, {})

    def test_session_handoff_carries_recent_raw_turns(self):
        handoff = SessionHandoff(
            stable_task_definition="rename service",
            recent_raw_turns=[{"role": "user", "content": "rename it"}],
        )

        self.assertEqual(handoff.stable_task_definition, "rename service")
        self.assertEqual(handoff.recent_raw_turns[0]["role"], "user")

    def test_transcript_chunk_tracks_indices(self):
        chunk = TranscriptChunk(
            chunk_id=2,
            start_index=4,
            end_index=7,
            token_count=123,
            items=[{"role": "user", "content": "hi"}],
        )

        self.assertEqual(chunk.start_index, 4)
        self.assertEqual(chunk.end_index, 7)
        self.assertEqual(chunk.token_count, 123)
