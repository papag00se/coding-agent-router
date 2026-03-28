from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.clients import ollama_client
from app.clients.ollama_client import OllamaClient


class _DummyJSONResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class _DummyStreamResponse:
    def __init__(self, recorder: "_Recorder", *, block_after_first_line: bool) -> None:
        self.recorder = recorder
        self.block_after_first_line = block_after_first_line

    def __enter__(self) -> "_DummyStreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = True):
        del decode_unicode
        self.recorder.stream_started.set()
        yield json.dumps({"message": {"content": "chunk-1"}})
        if self.block_after_first_line:
            self.recorder.release.wait(timeout=2)
            yield json.dumps({"message": {"content": "chunk-2"}})


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.first_entered = threading.Event()
        self.stream_started = threading.Event()
        self.release = threading.Event()


class _BlockingChatSession:
    def __init__(self, recorder: _Recorder, name: str, *, block_first: bool) -> None:
        self.recorder = recorder
        self.name = name
        self.block_first = block_first

    def post(self, *args, **kwargs):
        del args, kwargs
        self.recorder.calls.append(self.name)
        if len(self.recorder.calls) == 1:
            self.recorder.first_entered.set()
            if self.block_first:
                self.recorder.release.wait(timeout=2)
        payload = {"message": {"content": self.name}}
        return _DummyJSONResponse(payload)


class _BlockingStreamSession:
    def __init__(self, recorder: _Recorder, name: str, *, block_first: bool) -> None:
        self.recorder = recorder
        self.name = name
        self.block_first = block_first

    def post(self, *args, **kwargs):
        del args, kwargs
        self.recorder.calls.append(self.name)
        if len(self.recorder.calls) == 1:
            self.recorder.first_entered.set()
        return _DummyStreamResponse(self.recorder, block_after_first_line=self.block_first)


class TestOllamaClientQueueing(unittest.TestCase):
    def test_chat_requests_to_same_base_url_queue_across_clients(self) -> None:
        recorder = _Recorder()
        results: dict[str, dict[str, object]] = {}
        errors: list[BaseException] = []

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(ollama_client, "_OLLAMA_LOCK_DIR", Path(tmpdir)):
            client_one = OllamaClient("http://ollama.local:11434", timeout_seconds=5)
            client_one.session = _BlockingChatSession(recorder, "one", block_first=True)
            client_two = OllamaClient("http://ollama.local:11434", timeout_seconds=5)
            client_two.session = _BlockingChatSession(recorder, "two", block_first=False)

            def run_chat(client: OllamaClient, key: str) -> None:
                try:
                    results[key] = client.chat(
                        "model",
                        [{"role": "user", "content": "hello"}],
                        temperature=0.0,
                        num_ctx=8,
                    )
                except BaseException as exc:  # pragma: no cover - surfaced in assertion
                    errors.append(exc)

            first = threading.Thread(target=run_chat, args=(client_one, "one"))
            second = threading.Thread(target=run_chat, args=(client_two, "two"))

            first.start()
            self.assertTrue(recorder.first_entered.wait(timeout=2))
            second.start()
            time.sleep(0.1)

            self.assertEqual(recorder.calls, ["one"])

            recorder.release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertFalse(errors)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(recorder.calls, ["one", "two"])
        self.assertEqual(results["one"]["message"]["content"], "one")
        self.assertEqual(results["two"]["message"]["content"], "two")

    def test_chat_stream_requests_hold_lock_until_stream_finishes(self) -> None:
        recorder = _Recorder()
        results: dict[str, list[dict[str, object]]] = {}
        errors: list[BaseException] = []

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(ollama_client, "_OLLAMA_LOCK_DIR", Path(tmpdir)):
            client_one = OllamaClient("http://ollama.local:11434", timeout_seconds=5)
            client_one.session = _BlockingStreamSession(recorder, "one", block_first=True)
            client_two = OllamaClient("http://ollama.local:11434", timeout_seconds=5)
            client_two.session = _BlockingStreamSession(recorder, "two", block_first=False)

            def run_stream(client: OllamaClient, key: str) -> None:
                try:
                    results[key] = list(
                        client.chat_stream(
                            "model",
                            [{"role": "user", "content": "hello"}],
                            temperature=0.0,
                            num_ctx=8,
                        )
                    )
                except BaseException as exc:  # pragma: no cover - surfaced in assertion
                    errors.append(exc)

            first = threading.Thread(target=run_stream, args=(client_one, "one"))
            second = threading.Thread(target=run_stream, args=(client_two, "two"))

            first.start()
            self.assertTrue(recorder.first_entered.wait(timeout=2))
            self.assertTrue(recorder.stream_started.wait(timeout=2))
            second.start()
            time.sleep(0.1)

            self.assertEqual(recorder.calls, ["one"])

            recorder.release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertFalse(errors)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(recorder.calls, ["one", "two"])
        self.assertEqual(results["one"][0]["message"]["content"], "chunk-1")
        self.assertEqual(results["two"][0]["message"]["content"], "chunk-1")
