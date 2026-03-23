from __future__ import annotations

import json
import unittest

from app.clients.ollama_client import OllamaClient


class _FakeResponse:
    def __init__(self, payload=None, lines=None):
        self.payload = payload or {}
        self.lines = lines or []
        self.raise_called = False

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
        self.mounts = []
        self.calls = []
        self.response = None

    def mount(self, prefix, adapter):
        self.mounts.append((prefix, adapter))

    def post(self, url, json=None, timeout=None, stream=False):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "stream": stream})
        return self.response


class TestOllamaClient(unittest.TestCase):
    def test_chat_builds_expected_payload(self) -> None:
        client = OllamaClient("http://127.0.0.1:11434/", (1, 2), pool_connections=3, pool_maxsize=4)
        fake_session = _FakeSession()
        fake_session.response = _FakeResponse(payload={"message": {"content": "ok"}})
        client.session = fake_session

        body = client.chat(
            "model",
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            num_ctx=4096,
            max_tokens=256,
            system="sys",
            response_format="json",
            tools=[{"type": "function"}],
        )

        self.assertEqual(body["message"]["content"], "ok")
        payload = fake_session.calls[0]["json"]
        self.assertEqual(fake_session.calls[0]["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["tools"], [{"type": "function"}])
        self.assertEqual(payload["options"]["num_predict"], 256)
        self.assertFalse(payload["stream"])

    def test_chat_without_optional_flags_keeps_minimal_payload(self) -> None:
        client = OllamaClient("http://127.0.0.1:11434", 5)
        fake_session = _FakeSession()
        fake_session.response = _FakeResponse(payload={"done": True})
        client.session = fake_session

        client.chat("model", [{"role": "user", "content": "hi"}], temperature=0.0, num_ctx=1024)
        payload = fake_session.calls[0]["json"]
        self.assertNotIn("format", payload)
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])

    def test_chat_stream_yields_json_lines_and_skips_blanks(self) -> None:
        client = OllamaClient("http://127.0.0.1:11434", 5)
        fake_session = _FakeSession()
        fake_session.response = _FakeResponse(lines=["", json.dumps({"message": {"content": "a"}}), json.dumps({"done": True})])
        client.session = fake_session

        items = list(
            client.chat_stream(
                "model",
                [{"role": "user", "content": "hi"}],
                temperature=0.2,
                num_ctx=4096,
                system=None,
                tools=None,
            )
        )

        self.assertEqual(items, [{"message": {"content": "a"}}, {"done": True}])
        self.assertTrue(fake_session.calls[0]["stream"])

    def test_chat_stream_applies_optional_system_json_and_tools(self) -> None:
        client = OllamaClient("http://127.0.0.1:11434", 5)
        fake_session = _FakeSession()
        fake_session.response = _FakeResponse(lines=[json.dumps({"done": True})])
        client.session = fake_session

        list(
            client.chat_stream(
                "model",
                [{"role": "user", "content": "hi"}],
                temperature=0.2,
                num_ctx=4096,
                max_tokens=64,
                system="sys",
                response_format="json",
                tools=[{"type": "function"}],
            )
        )

        payload = fake_session.calls[0]["json"]
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["tools"], [{"type": "function"}])
        self.assertEqual(payload["options"]["num_predict"], 64)
