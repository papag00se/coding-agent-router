from __future__ import annotations

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
    def test_refine_state_uses_json_mode_and_validates_output(self):
        client = _FakeClient('{"objective_update":"new objective","latest_plan_update":["step 1"]}')
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
        self.assertEqual(client.calls[0]["response_format"], "json")
        self.assertFalse(client.calls[0]["think"])
        self.assertEqual(client.calls[0]["max_tokens"], 11003)

    def test_refine_state_response_budget_respects_remaining_context(self):
        refiner = CompactionRefiner(client=_FakeClient("{}"), model="qwen-test", temperature=0.0, num_ctx=16384)
        self.assertEqual(refiner._response_token_budget(1000), 15128)
        self.assertEqual(refiner._response_token_budget(14336), 1792)
        self.assertEqual(refiner._response_token_budget(15000), 1128)
        self.assertEqual(refiner._response_token_budget(17000), 0)

    def test_refine_state_uses_burst_context_when_default_window_would_starve_output(self):
        client = _FakeClient('{"objective_update":"ok"}')
        refiner = CompactionRefiner(client=client, model="qwen-test", temperature=0.0, num_ctx=16384)

        with mock.patch(
            "app.compaction.refiner.estimate_refinement_request_tokens",
            return_value=16000,
        ):
            refiner.refine_state(MergedState(), [{"role": "user", "content": "x"}], current_request="x")

        self.assertEqual(client.calls[0]["num_ctx"], 17408)
        self.assertEqual(client.calls[0]["max_tokens"], 1152)

    def test_refine_state_normalizes_structured_fields(self):
        client = _FakeClient(
            (
                '{"repo_state_updates":"working in /repo",'
                '"add_test_status":{"compaction":"passing"},'
                '"latest_plan_update":[{"step":"Retry compaction","status":"in_progress"}]}'
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

    def test_refine_state_applies_patch_without_overwriting_deterministic_merge(self):
        client = _FakeClient(
            (
                '{"add_pending_todos":["verify preview minting"],'
                '"add_errors":["scope 7 hydration mismatch"],'
                '"add_external_references":["https://preview.api.handle.me/health"]}'
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
        self.assertEqual(result.pending_todos, ["rerun scope 2", "verify preview minting"])
        self.assertEqual(result.errors, ["old error", "scope 7 hydration mismatch"])
        self.assertEqual(result.external_references, ["https://preview.api.handle.me/health"])
        self.assertEqual(result.merged_chunk_count, 3)

    def test_refine_state_ignores_unrelated_full_state_object(self):
        client = _FakeClient(
            (
                '{"objective":"Implement a secure, production-ready authentication system",'
                '"files_touched":["src/app/auth.config.ts"],'
                '"latest_plan":["Implement email verification flow [pending]"]}'
            )
        )
        refiner = CompactionRefiner(client=client, model="qwen-test", temperature=0.0, num_ctx=12000)
        state = MergedState(objective="close lcr-05b", files_touched=["deploy.sh"], merged_chunk_count=5)

        result = refiner.refine_state(
            state,
            [{"role": "user", "content": "continue"}],
            current_request="continue",
        )

        self.assertEqual(result.objective, "close lcr-05b")
        self.assertEqual(result.files_touched, ["deploy.sh"])
        self.assertEqual(result.latest_plan, [])
        self.assertEqual(result.merged_chunk_count, 5)

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
        self.assertIn("This is a bounded patch pass, not a full state rewrite.", REFINEMENT_SYSTEM_PROMPT)
        self.assertIn("current_state", payload)
        self.assertIn("recent_raw_turns", payload)
        self.assertIn("required_keys", payload)
        self.assertIn("objective_update", payload)
