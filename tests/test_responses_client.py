from __future__ import annotations

import unittest
import requests

from app.clients.responses_client import ResponsesClient


class _FakeResponse:
    def __init__(self, payload=None, *, text: str = "", status_code: int = 200, lines=None):
        self.payload = payload or {}
        self.text = text
        self.status_code = status_code
        self.raise_called = False
        self.lines = lines or []

    def raise_for_status(self):
        self.raise_called = True

    def json(self):
        return self.payload

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _FakeSession:
    def __init__(self):
        self.calls = []
        self.response = None

    def post(self, url, json=None, headers=None, timeout=None, stream=False):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout, "stream": stream})
        return self.response


class TestResponsesClient(unittest.TestCase):
    def test_chat_builds_expected_responses_payload(self) -> None:
        client = ResponsesClient(
            "https://chatgpt.com/backend-api/codex/",
            (1, 2),
            headers={"authorization": "Bearer test-token"},
        )
        fake_session = _FakeSession()
        fake_session.response = _FakeResponse(
            lines=[
                'event: response.output_text.delta',
                'data: {"type":"response.output_text.delta","delta":"{\\"objective\\":\\"ok\\"}"}',
                "",
                'event: response.completed',
                'data: {"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"{\\"objective\\":\\"ok\\"}"}]}],"usage":{"input_tokens":11,"output_tokens":7}}}',
                "",
            ],
        )
        client.session = fake_session

        body = client.chat(
            "gpt-5.3-codex-spark",
            [{"role": "user", "content": '{"chunk":{"id":1}}'}],
            temperature=0.0,
            num_ctx=16384,
            max_tokens=512,
            system="Return JSON only.",
            response_format={"title": "ChunkExtraction", "type": "object", "properties": {"objective": {"type": "string"}}},
            think=False,
        )

        self.assertEqual(body["message"]["content"], '{"objective":"ok"}')
        self.assertEqual(body["prompt_eval_count"], 11)
        self.assertEqual(body["eval_count"], 7)
        call = fake_session.calls[0]
        self.assertEqual(call["url"], "https://chatgpt.com/backend-api/codex/responses")
        self.assertEqual(call["headers"]["authorization"], "Bearer test-token")
        self.assertEqual(call["json"]["model"], "gpt-5.3-codex-spark")
        self.assertEqual(call["json"]["instructions"], "Return JSON only.")
        self.assertEqual(call["json"]["max_output_tokens"], 512)
        self.assertTrue(call["json"]["stream"])
        self.assertEqual(call["json"]["text"]["format"]["type"], "json_schema")
        self.assertEqual(call["json"]["text"]["format"]["name"], "chunkextraction")
        self.assertTrue(call["stream"])
        self.assertEqual(
            call["json"]["input"],
            [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": '{"chunk":{"id":1}}'}]}],
        )

    def test_chat_parses_streaming_completed_event(self) -> None:
        client = ResponsesClient("https://chatgpt.com/backend-api/codex", 5)
        fake_session = _FakeSession()
        fake_session.response = _FakeResponse(
            lines=[
                'event: response.in_progress',
                'data: {"type":"response.in_progress"}',
                "",
                'event: response.output_text.delta',
                'data: {"type":"response.output_text.delta","delta":"partial"}',
                "",
                'event: response.completed',
                'data: {"type":"response.completed","response":{"output_text":"final","usage":{"input_tokens":17,"output_tokens":5}}}',
                "",
            ]
        )
        client.session = fake_session

        body = client.chat("gpt-5.3-codex-spark", [{"role": "user", "content": "hi"}], temperature=0.0, num_ctx=8192)

        self.assertEqual(body["message"]["content"], "final")
        self.assertEqual(body["prompt_eval_count"], 17)
        self.assertEqual(body["eval_count"], 5)

    def test_chat_parses_streaming_completed_event_from_bytes(self) -> None:
        client = ResponsesClient("https://chatgpt.com/backend-api/codex", 5)
        fake_session = _FakeSession()
        fake_session.response = _FakeResponse(
            lines=[
                b"event: response.output_text.delta",
                b'data: {"type":"response.output_text.delta","delta":"partial"}',
                b"",
                b"event: response.completed",
                b'data: {"type":"response.completed","response":{"output_text":"final","usage":{"input_tokens":17,"output_tokens":5}}}',
                b"",
            ]
        )
        client.session = fake_session

        body = client.chat("gpt-5.3-codex-spark", [{"role": "user", "content": "hi"}], temperature=0.0, num_ctx=8192)

        self.assertEqual(body["message"]["content"], "final")
        self.assertEqual(body["prompt_eval_count"], 17)
        self.assertEqual(body["eval_count"], 5)

    def test_chat_raises_clear_error_for_http_error_body(self) -> None:
        client = ResponsesClient("https://chatgpt.com/backend-api/codex", 5)
        fake_session = _FakeSession()
        fake_session.response = _FakeResponse(text='{"error":"unsupported field"}', status_code=400)
        client.session = fake_session

        with self.assertRaisesRegex(requests.HTTPError, 'responses client upstream returned HTTP 400: \\{"error":"unsupported field"\\}'):
            client.chat("gpt-5.3-codex-spark", [{"role": "user", "content": "hi"}], temperature=0.0, num_ctx=8192)

    def test_chat_raises_clear_error_for_non_json_response(self) -> None:
        class _BrokenResponse(_FakeResponse):
            def json(self):
                raise ValueError("bad json")

        client = ResponsesClient("https://chatgpt.com/backend-api/codex", 5)
        fake_session = _FakeSession()
        fake_session.response = _BrokenResponse(text="not json")
        client.session = fake_session

        with self.assertRaisesRegex(ValueError, "responses client returned non-JSON output: not json"):
            client.chat("gpt-5.3-codex-spark", [{"role": "user", "content": "hi"}], temperature=0.0, num_ctx=8192)
