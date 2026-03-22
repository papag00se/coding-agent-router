from __future__ import annotations

import unittest

from app.compaction.merger import merge_states
from app.compaction.models import ChunkExtraction


class TestCompactionMerger(unittest.TestCase):
    def test_merge_states_keeps_latest_objective_and_dedupes_lists(self):
        merged = merge_states(
            [
                ChunkExtraction(
                    chunk_id=1,
                    objective="rename service",
                    files_touched=["README.md"],
                    constraints=["minimal change"],
                    latest_plan=["search", "edit"],
                    repo_state={"branch": "main"},
                ),
                ChunkExtraction(
                    chunk_id=2,
                    objective="rename service everywhere",
                    files_touched=["README.md", "app/main.py"],
                    constraints=["minimal change", "update tests"],
                    latest_plan=["search", "edit", "test"],
                    repo_state={"branch": "feature/router"},
                ),
            ]
        )

        self.assertEqual(merged.objective, "rename service everywhere")
        self.assertEqual(merged.files_touched, ["README.md", "app/main.py"])
        self.assertEqual(merged.constraints, ["minimal change", "update tests"])
        self.assertEqual(merged.latest_plan, ["search", "edit", "test"])
        self.assertEqual(merged.repo_state["branch"], "feature/router")
        self.assertEqual(merged.merged_chunk_count, 2)
