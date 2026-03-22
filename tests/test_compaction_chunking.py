from __future__ import annotations

import unittest

from app.compaction.chunking import chunk_transcript_items, split_recent_raw_turns


class TestCompactionChunking(unittest.TestCase):
    def test_chunk_transcript_items_handles_empty_and_large_single_item(self):
        self.assertEqual(
            chunk_transcript_items([], target_tokens=10, max_tokens=20, overlap_tokens=5),
            [],
        )

        chunks = chunk_transcript_items(
            [{"role": "user", "content": "x" * 20000}],
            target_tokens=100,
            max_tokens=200,
            overlap_tokens=50,
        )
        self.assertEqual(len(chunks), 1)
        self.assertGreater(chunks[0].token_count, 200)

    def test_chunk_transcript_items_applies_overlap(self):
        items = [{"role": "user", "content": "x" * 4000} for _ in range(4)]

        chunks = chunk_transcript_items(
            items,
            target_tokens=2200,
            max_tokens=2600,
            overlap_tokens=1200,
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].items, items[0:2])
        self.assertEqual(chunks[1].items, items[1:3])
        self.assertGreater(chunks[1].overlap_from_previous_tokens, 0)

    def test_split_recent_raw_turns_keeps_newest_tail(self):
        items = [
            {"role": "user", "content": "a" * 2000},
            {"role": "assistant", "content": "b" * 2000},
            {"role": "user", "content": "c" * 2000},
        ]

        compactable, recent = split_recent_raw_turns(items, keep_tokens=450)

        self.assertEqual(compactable, items[:2])
        self.assertEqual(recent, items[2:])

    def test_split_recent_raw_turns_returns_all_items_when_keep_tokens_non_positive(self):
        items = [{"role": "user", "content": "a"}]
        compactable, recent = split_recent_raw_turns(items, keep_tokens=0)
        self.assertEqual(compactable, items)
        self.assertEqual(recent, [])
