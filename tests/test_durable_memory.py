from __future__ import annotations

import unittest

from app.compaction.durable_memory import build_session_handoff, render_durable_memory
from app.compaction.models import MergedState


class TestDurableMemory(unittest.TestCase):
    def test_render_durable_memory_outputs_expected_sections(self):
        state = MergedState(
            objective="rename service",
            files_touched=["README.md"],
            accepted_fixes=["updated README title"],
            errors=["tool call leaked into text"],
            pending_todos=["update app title"],
            latest_plan=["search", "edit", "test"],
        )

        memory = render_durable_memory(state, [{"role": "user", "content": "rename it"}], "finish rename")

        self.assertIn("# Task State", memory.task_state)
        self.assertIn("README.md", memory.task_state)
        self.assertIn("# Decisions", memory.decisions)
        self.assertIn("updated README title", memory.decisions)
        self.assertIn("# Failures To Avoid", memory.failures_to_avoid)
        self.assertIn("# Session Handoff", memory.session_handoff)

    def test_build_session_handoff_maps_state_fields(self):
        state = MergedState(
            objective="rename service",
            repo_state={"branch": "main"},
            accepted_fixes=["updated README title"],
            pending_todos=["update app title"],
            unresolved_bugs=["tool loop"],
            latest_plan=["edit", "test"],
            errors=["parse failure"],
            rejected_ideas=["one-shot summary"],
        )

        handoff = build_session_handoff(state, [{"role": "assistant", "content": "done"}], "continue work")

        self.assertEqual(handoff.stable_task_definition, "rename service")
        self.assertEqual(handoff.repo_state["branch"], "main")
        self.assertEqual(handoff.unresolved_work, ["update app title", "tool loop"])
        self.assertEqual(handoff.failures_to_avoid, ["parse failure", "one-shot summary"])
