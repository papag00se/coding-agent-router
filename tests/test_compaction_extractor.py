from __future__ import annotations

import json
import unittest
from unittest import mock

from app.compaction.extractor import CompactionExtractor
from app.compaction.models import TranscriptChunk
from app.compaction.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_payload


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


class TestCompactionExtractor(unittest.TestCase):
    def test_extract_chunk_uses_json_mode_and_validates_output(self):
        client = _FakeClient(
            '{"objective":"rename service","files_touched":["README.md"],"constraints":["keep change minimal"]}'
        )
        extractor = CompactionExtractor(
            client=client,
            model="qwen-test",
            temperature=0.0,
            num_ctx=12000,
        )
        chunk = TranscriptChunk(chunk_id=3, start_index=0, end_index=1, token_count=250, items=[{"role": "user", "content": "rename it"}])

        result = extractor.extract_chunk(chunk, {"repo": "coding-agent-router"})

        self.assertEqual(result.chunk_id, 3)
        self.assertEqual(result.source_token_count, 250)
        self.assertEqual(result.files_touched, ["README.md"])
        self.assertEqual(client.calls[0]["response_format"]["title"], "ChunkExtraction")
        self.assertFalse(client.calls[0]["response_format"]["additionalProperties"])
        self.assertEqual(client.calls[0]["response_format"]["properties"]["repo_state"]["type"], "array")
        self.assertFalse(
            client.calls[0]["response_format"]["properties"]["repo_state"]["items"]["additionalProperties"]
        )
        self.assertFalse(client.calls[0]["think"])
        self.assertIsNone(client.calls[0]["max_tokens"])

    def test_extract_chunk_response_budget_respects_remaining_context(self):
        extractor = CompactionExtractor(client=_FakeClient("{}"), model="qwen-test", temperature=0.0, num_ctx=16384)
        self.assertEqual(extractor._response_token_budget(1000), 15128)
        self.assertEqual(extractor._response_token_budget(14336), 1792)
        self.assertEqual(extractor._response_token_budget(15000), 1128)
        self.assertEqual(extractor._response_token_budget(17000), 0)

    def test_extract_chunk_uses_burst_context_when_default_window_would_starve_output(self):
        client = _FakeClient('{"objective":"ok"}')
        extractor = CompactionExtractor(client=client, model="qwen-test", temperature=0.0, num_ctx=16384)
        chunk = TranscriptChunk(chunk_id=9, start_index=0, end_index=1, token_count=10, items=[{"role": "user", "content": "small"}])

        with mock.patch(
            "app.compaction.extractor.estimate_extraction_request_tokens",
            return_value=16000,
        ):
            extractor.extract_chunk(chunk)

        self.assertEqual(client.calls[0]["num_ctx"], 17408)
        self.assertIsNone(client.calls[0]["max_tokens"])

    def test_extract_chunk_rejects_non_json_output(self):
        extractor = CompactionExtractor(client=_FakeClient("not json"), model="qwen-test", temperature=0.0, num_ctx=12000)
        chunk = TranscriptChunk(chunk_id=1, start_index=0, end_index=1, token_count=50, items=[{"role": "user", "content": "rename it"}])

        with self.assertRaisesRegex(ValueError, "non-JSON output: not json"):
            extractor.extract_chunk(chunk)

    def test_extract_chunk_normalizes_structured_test_status_and_plan(self):
        client = _FakeClient(
            (
                '{"objective":"stabilize compaction",'
                '"test_status":{"lcr-01":"closed","lcr-07a":"open"},'
                '"latest_plan":['
                '{"step":"Inspect repo/session state","status":"in_progress"},'
                '{"step":"Implement upstream fix","status":"pending"}'
                ']}'
            )
        )
        extractor = CompactionExtractor(client=client, model="qwen-test", temperature=0.0, num_ctx=12000)
        chunk = TranscriptChunk(chunk_id=2, start_index=0, end_index=1, token_count=100, items=[{"role": "user", "content": "fix it"}])

        result = extractor.extract_chunk(chunk)

        self.assertEqual(result.test_status, ["lcr-01: closed", "lcr-07a: open"])
        self.assertEqual(
            result.latest_plan,
            ["Inspect repo/session state [in_progress]", "Implement upstream fix [pending]"],
        )

    def test_extract_chunk_normalizes_string_repo_state(self):
        client = _FakeClient('{"repo_state":"Working in /repo with regions us-west-2 and us-east-1"}')
        extractor = CompactionExtractor(client=client, model="qwen-test", temperature=0.0, num_ctx=12000)
        chunk = TranscriptChunk(chunk_id=4, start_index=0, end_index=1, token_count=80, items=[{"role": "user", "content": "summarize state"}])

        result = extractor.extract_chunk(chunk)

        self.assertEqual(result.repo_state, {"summary": "Working in /repo with regions us-west-2 and us-east-1"})

    def test_extract_chunk_normalizes_repo_state_entry_list(self):
        client = _FakeClient(
            '{"repo_state":[{"key":"branch","value":"feature/router"},{"key":"service","value":"compaction"}]}'
        )
        extractor = CompactionExtractor(client=client, model="qwen-test", temperature=0.0, num_ctx=12000)
        chunk = TranscriptChunk(chunk_id=5, start_index=0, end_index=1, token_count=80, items=[{"role": "user", "content": "summarize state"}])

        result = extractor.extract_chunk(chunk)

        self.assertEqual(result.repo_state, {"branch": "feature/router", "service": "compaction"})

    def test_extraction_prompt_is_explicit_and_payload_declares_contract(self):
        chunk = TranscriptChunk(
            chunk_id=1,
            start_index=0,
            end_index=2,
            token_count=50,
            items=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "exec_command",
                            "input": {
                                "cmd": "sed -n '1,40p' app/main.py",
                                "workdir": "/repo",
                                "yield_time_ms": 1000,
                                "max_output_tokens": 1200,
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "write_stdin",
                            "input": {
                                "session_id": 123,
                                "chars": "",
                                "yield_time_ms": 1000,
                            },
                        }
                    ],
                },
                {"role": "user", "content": "rename it"},
            ],
        )
        payload = build_extraction_payload(chunk, {"repo": "coding-agent-router"})
        parsed = json.loads(payload)

        self.assertIn("Return exactly one JSON object and nothing else.", EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("output_contract", parsed)
        self.assertIn("required_keys", parsed["output_contract"])
        self.assertEqual(parsed["chunk"]["id"], 1)
        self.assertEqual(
            parsed["chunk"]["events"][:3],
            [
                {"r": "a", "k": "cmd", "c": "sed -n '1,40p' app/main.py", "wd": "/repo"},
                {"r": "a", "k": "poll", "sid": 123},
                {"r": "u", "k": "msg", "c": "rename it"},
            ],
        )
