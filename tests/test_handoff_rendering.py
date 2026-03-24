from __future__ import annotations

import unittest

from app.compaction.handoff import build_codex_handoff_flow, render_codex_support_prompt, render_inline_compaction_summary
from app.compaction.models import CodexHandoffFlow, DurableMemorySet, SessionHandoff


class TestHandoffRendering(unittest.TestCase):
    def test_render_inline_compaction_summary_omits_machine_scaffold(self):
        flow = CodexHandoffFlow(
            durable_memory=[
                {"name": "TASK_STATE.md", "content": "# Task State\n## Objective\n- rename service\n"},
                {"name": "NEXT_STEPS.md", "content": "# Next Steps\n## Pending TODOs\n- rerun browser test\n"},
            ],
            structured_handoff={"stable_task_definition": "rename service"},
            recent_raw_turns=[{"role": "user", "content": "old turn"}],
            current_request="Finish the rename.",
        )

        summary = render_inline_compaction_summary(flow)

        self.assertIn("# Task State", summary)
        self.assertIn("# Next Steps", summary)
        self.assertIn("# Current Request\nFinish the rename.", summary)
        self.assertNotIn("Structured handoff:", summary)
        self.assertNotIn("Recent raw turns:", summary)

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

    def test_render_codex_support_prompt_does_not_repeat_current_request_from_stored_handoff(self):
        flow = CodexHandoffFlow(
            durable_memory=[
                {"name": "SESSION_HANDOFF.md", "content": "# Session Handoff\n## Stable Task Definition\n- rename service\n"},
            ],
            structured_handoff={"stable_task_definition": "rename service"},
            recent_raw_turns=[],
            current_request="Finish the rename.",
        )

        prompt = render_codex_support_prompt(flow)

        self.assertEqual(prompt.count("Finish the rename."), 1)

    def test_build_codex_handoff_flow_strips_legacy_current_request_section(self):
        flow = build_codex_handoff_flow(
            DurableMemorySet(
                session_handoff="# Session Handoff\n## Stable Task Definition\n- rename service\n## Current Request\n- stale oversized request\n",
            ),
            SessionHandoff(
                stable_task_definition="rename service",
                current_request="fresh request",
                recent_raw_turns=[{"role": "user", "content": "old turn"}],
            ),
            current_request="fresh request",
        )

        session_handoff = next(item["content"] for item in flow.durable_memory if item["name"] == "SESSION_HANDOFF.md")
        self.assertNotIn("stale oversized request", session_handoff)
        self.assertNotIn("Current Request", session_handoff)
