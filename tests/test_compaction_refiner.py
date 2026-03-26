from __future__ import annotations

import json
import unittest
from unittest import mock

from app.compaction.models import MergedState
from app.compaction.prompts import REFINEMENT_SYSTEM_PROMPT, build_refinement_payload
from app.compaction.refiner import CompactionRefiner


class _FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, model, messages, *, temperature, num_ctx, max_tokens=None, system=None, response_format=None, think=None, tools=None):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "num_ctx": num_ctx,
                "max_tokens": max_tokens,
                "system": system,
                "response_format": response_format,
                "think": think,
                "tools": tools,
            }
        )
        return {"message": {"content": self.content}}


class TestCompactionRefiner(unittest.TestCase):
    def test_refine_state_uses_recent_state_schema_and_merges_output(self):
        client = _FakeClient('{"objective":"new objective","latest_plan":["step 1"]}')
        refiner = CompactionRefiner(
            client=client,
            model="qwen-test",
            temperature=0.0,
            num_ctx=12000,
        )
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
        self.assertEqual(client.calls[0]["response_format"]["title"], "RecentStateExtraction")
        self.assertFalse(client.calls[0]["response_format"]["additionalProperties"])
        self.assertEqual(client.calls[0]["response_format"]["properties"]["repo_state"]["type"], "array")
        self.assertFalse(
            client.calls[0]["response_format"]["properties"]["repo_state"]["items"]["additionalProperties"]
        )
        self.assertFalse(client.calls[0]["think"])
        self.assertIsNone(client.calls[0]["max_tokens"])

    def test_refine_state_response_budget_respects_remaining_context(self):
        refiner = CompactionRefiner(client=_FakeClient("{}"), model="qwen-test", temperature=0.0, num_ctx=16384)
        self.assertEqual(refiner._response_token_budget(1000), 15128)
        self.assertEqual(refiner._response_token_budget(14336), 1792)
        self.assertEqual(refiner._response_token_budget(15000), 1128)
        self.assertEqual(refiner._response_token_budget(17000), 0)

    def test_refine_state_uses_burst_context_when_default_window_would_starve_output(self):
        client = _FakeClient('{"objective":"ok"}')
        refiner = CompactionRefiner(client=client, model="qwen-test", temperature=0.0, num_ctx=16384)

        with mock.patch(
            "app.compaction.refiner.estimate_refinement_request_tokens",
            return_value=16000,
        ):
            refiner.refine_state(MergedState(), [{"role": "user", "content": "x"}], current_request="x")

        self.assertEqual(client.calls[0]["num_ctx"], 17408)
        self.assertIsNone(client.calls[0]["max_tokens"])

    def test_refine_state_normalizes_structured_fields_from_recent_state(self):
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

    def test_refine_state_normalizes_repo_state_entry_list(self):
        client = _FakeClient(
            (
                '{"repo_state":['
                '{"key":"branch","value":"main"},'
                '{"key":"endpoint","value":"127.0.0.1:8081/v1/responses"}'
                ']}'
            )
        )
        refiner = CompactionRefiner(client=client, model="qwen-test", temperature=0.0, num_ctx=12000)

        result = refiner.refine_state(
            MergedState(merged_chunk_count=2),
            [{"role": "user", "content": "continue"}],
            current_request="continue",
        )

        self.assertEqual(result.repo_state, {"branch": "main", "endpoint": "127.0.0.1:8081/v1/responses"})

    def test_refine_state_merges_recent_state_without_overwriting_existing_lists(self):
        client = _FakeClient(
            (
                '{"pending_todos":["verify preview minting"],'
                '"errors":["scope 7 hydration mismatch"],'
                '"external_references":["https://preview.api.handle.me/health"]}'
            )
        )
        refiner = CompactionRefiner(client=client, model="qwen-test", temperature=0.0, num_ctx=12000)
        state = MergedState(
            objective="close lcr-05b",
            pending_todos=["rerun scope 2"],
            errors=["old error"],
            merged_chunk_count=3,
        )

        result = refiner.refine_state(
            state,
            [{"role": "user", "content": "continue"}],
            current_request="continue",
        )

        self.assertEqual(result.objective, "close lcr-05b")
        self.assertEqual(result.pending_todos, ["verify preview minting", "rerun scope 2"])
        self.assertEqual(result.errors, ["scope 7 hydration mismatch", "old error"])
        self.assertEqual(result.external_references, ["https://preview.api.handle.me/health"])
        self.assertEqual(result.merged_chunk_count, 3)

    def test_refine_state_returns_original_state_for_empty_recent_extraction(self):
        refiner = CompactionRefiner(client=_FakeClient("{}"), model="qwen-test", temperature=0.0, num_ctx=12000)
        state = MergedState(objective="close lcr-05b", files_touched=["deploy.sh"], merged_chunk_count=5)

        result = refiner.refine_state(state, [{"role": "user", "content": "continue"}], current_request="continue")

        self.assertEqual(result, state)

    def test_refine_state_rejects_non_json_output(self):
        refiner = CompactionRefiner(client=_FakeClient("not json"), model="qwen-test", temperature=0.0, num_ctx=12000)
        with self.assertRaisesRegex(ValueError, "non-JSON output: not json"):
            refiner.refine_state(MergedState(), [{"role": "user", "content": "x"}], current_request="x")

    def test_refinement_prompt_and_payload_describe_recent_state_extraction(self):
        payload = build_refinement_payload(
            MergedState(objective="rename"),
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "write_stdin",
                            "input": {"session_id": 55, "chars": "", "yield_time_ms": 1000},
                        },
                        {
                            "type": "tool_use",
                            "name": "update_plan",
                            "input": {
                                "plan": [
                                    {"step": "inspect repo", "status": "in_progress"},
                                    {"step": "patch route", "status": "pending"},
                                ]
                            },
                        },
                    ],
                }
            ],
            "finish rename",
            {"repo": "coding-agent-router"},
        )
        parsed = json.loads(payload)

        self.assertIn("Return exactly one JSON object and nothing else.", REFINEMENT_SYSTEM_PROMPT)
        self.assertIn("This is a recent-state extraction pass, not a diff or patch pass.", REFINEMENT_SYSTEM_PROMPT)
        self.assertIn("recent_events", parsed)
        self.assertIn("required_keys", parsed["output_contract"])
        self.assertIn("objective", parsed["output_contract"]["required_keys"])
        self.assertNotIn("current_state", parsed)
        self.assertEqual(
            parsed["recent_events"],
            [
                {"r": "a", "k": "poll", "sid": 55},
                {"r": "a", "k": "plan", "steps": ["inspect repo [in_progress]", "patch route [pending]"]},
            ],
        )
