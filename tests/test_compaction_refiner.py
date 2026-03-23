from __future__ import annotations

import unittest

from app.compaction.models import MergedState
from app.compaction.prompts import REFINEMENT_SYSTEM_PROMPT, build_refinement_payload
from app.compaction.refiner import CompactionRefiner


class _FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, model, messages, *, temperature, num_ctx, max_tokens=None, system=None, response_format=None, tools=None):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "num_ctx": num_ctx,
                "max_tokens": max_tokens,
                "system": system,
                "response_format": response_format,
                "tools": tools,
            }
        )
        return {"message": {"content": self.content}}


class TestCompactionRefiner(unittest.TestCase):
    def test_refine_state_uses_json_mode_and_validates_output(self):
        client = _FakeClient('{"objective":"new objective","latest_plan":["step 1"]}')
        refiner = CompactionRefiner(client=client, model="qwen-test", temperature=0.0, num_ctx=12000)
        state = MergedState(objective="old objective", merged_chunk_count=4)

        result = refiner.refine_state(
            state,
            [{"role": "user", "content": "newer request"}],
            current_request="newer request",
            repo_context={"repo": "coding-agent-router"},
        )

        self.assertEqual(result.objective, "new objective")
        self.assertEqual(result.latest_plan, ["step 1"])
        self.assertEqual(result.merged_chunk_count, 4)
        self.assertEqual(client.calls[0]["response_format"], "json")
        self.assertIsNotNone(client.calls[0]["max_tokens"])

    def test_refine_state_normalizes_structured_fields(self):
        client = _FakeClient(
            (
                '{"repo_state":"working in /repo",'
                '"test_status":{"compaction":"passing"},'
                '"latest_plan":[{"step":"Retry compaction","status":"in_progress"}]}'
            )
        )
        refiner = CompactionRefiner(client=client, model="qwen-test", temperature=0.0, num_ctx=12000)

        result = refiner.refine_state(
            MergedState(merged_chunk_count=2),
            [{"role": "user", "content": "continue"}],
            current_request="continue",
        )

        self.assertEqual(result.repo_state, {"summary": "working in /repo"})
        self.assertEqual(result.test_status, ["compaction: passing"])
        self.assertEqual(result.latest_plan, ["Retry compaction [in_progress]"])
        self.assertEqual(result.merged_chunk_count, 2)

    def test_refine_state_rejects_non_json_output(self):
        refiner = CompactionRefiner(client=_FakeClient("not json"), model="qwen-test", temperature=0.0, num_ctx=12000)
        with self.assertRaisesRegex(ValueError, "non-JSON output: not json"):
            refiner.refine_state(MergedState(), [{"role": "user", "content": "x"}], current_request="x")

    def test_refinement_prompt_and_payload_are_explicit(self):
        payload = build_refinement_payload(
            MergedState(objective="rename"),
            [{"role": "assistant", "content": "recent raw"}],
            "finish rename",
            {"repo": "coding-agent-router"},
        )

        self.assertIn("Return exactly one JSON object and nothing else.", REFINEMENT_SYSTEM_PROMPT)
        self.assertIn("current_state", payload)
        self.assertIn("recent_raw_turns", payload)
        self.assertIn("required_keys", payload)
