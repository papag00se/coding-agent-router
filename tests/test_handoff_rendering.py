from __future__ import annotations

import unittest

from app.compaction.handoff import render_codex_support_prompt
from app.compaction.models import CodexHandoffFlow


class TestHandoffRendering(unittest.TestCase):
    def test_render_codex_support_prompt_orders_sections(self):
        flow = CodexHandoffFlow(
            durable_memory=[
                {"name": "TASK_STATE.md", "content": "# Task State\n- rename service\n"},
                {"name": "DECISIONS.md", "content": "# Decisions\n- keep change minimal\n"},
            ],
            structured_handoff={"stable_task_definition": "rename service"},
            recent_raw_turns=[{"role": "user", "content": "old turn"}],
            current_request="Finish the rename.",
        )

        prompt = render_codex_support_prompt(flow, system="Be concise.")

        self.assertIn("System instructions:\nBe concise.", prompt)
        self.assertIn("### TASK_STATE.md", prompt)
        self.assertIn("Structured handoff:", prompt)
        self.assertIn('"stable_task_definition": "rename service"', prompt)
        self.assertIn("Recent raw turns:", prompt)
        self.assertTrue(prompt.rstrip().endswith("Finish the rename."))
