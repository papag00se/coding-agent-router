from __future__ import annotations

import asyncio
import tempfile
import unittest

from app import app_server


class _FakeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_json(self):
        if not self.messages:
            raise RuntimeError("done")
        value = self.messages.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def send_json(self, payload):
        self.sent.append(payload)


class TestAppServerHelpers(unittest.TestCase):
    def test_flatten_response_render_and_text_helpers(self) -> None:
        self.assertEqual(
            app_server._flatten_user_input(
                [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "url": "https://example.test/cat.png"},
                    {"type": "localImage", "path": "/tmp/cat.png"},
                    {"type": "skill", "name": "tooling", "path": "/skills/tooling"},
                    {"type": "mention", "name": "repo", "path": "app://repo"},
                    "ignored",
                ]
            ),
            "hello\nhttps://example.test/cat.png\n/tmp/cat.png\ntooling /skills/tooling\nrepo app://repo",
        )
        self.assertEqual(
            app_server._response_text({"content": [{"type": "text", "text": "a"}, {"type": "tool_use", "name": "x"}, {"type": "text", "text": "b"}]}),
            "a\nb",
        )
        rendered = app_server._render_compacted_flow(
            {
                "durable_memory": [{"name": "TASK_STATE.md", "content": "# Task State\n- item\n"}],
                "structured_handoff": {"stable_task_definition": "rename"},
                "recent_raw_turns": [{"role": "user", "content": "old"}],
                "current_request": "fallback",
            },
            current_request="current",
        )
        self.assertIn("TASK_STATE.md", rendered)
        self.assertIn('"stable_task_definition": "rename"', rendered)
        self.assertTrue(rendered.endswith("current"))

    def test_get_thread_state_missing_items_for_compaction_and_latest_user_text(self) -> None:
        bridge = app_server.CodexAppServerBridge(service=object(), state_dir=tempfile.mkdtemp())
        self.assertIsNone(bridge._get_thread_state("missing"))

        state = app_server.ThreadState(
            thread={"preview": "fallback"},
            approval_policy="never",
            cwd="/tmp",
            model="router",
            model_provider="router",
            sandbox={"type": "dangerFullAccess"},
            history_items=[{"role": "user", "content": "latest"}],
        )
        self.assertEqual(bridge._items_for_compaction(state), [{"role": "user", "content": "latest"}])
        self.assertEqual(bridge._latest_user_text(state.history_items, "fallback"), "latest")
        self.assertEqual(bridge._latest_user_text([{"role": "assistant", "content": "x"}], "fallback"), "fallback")

        state.compacted_flow = {
            "current_request": "remember",
            "durable_memory": [],
            "structured_handoff": {},
            "recent_raw_turns": [],
        }
        items = bridge._items_for_compaction(state)
        self.assertEqual(items[0]["role"], "assistant")
        self.assertIn("Durable memory:", items[0]["content"])

    def test_handle_websocket_ignores_noise_and_reports_unsupported_methods(self) -> None:
        bridge = app_server.CodexAppServerBridge(service=object(), state_dir=tempfile.mkdtemp())
        ws = _FakeWebSocket(
            [
                "not-a-dict",
                {"method": "missing-id"},
                {"id": 1, "method": "initialized"},
                {"id": 2, "method": "unknown", "params": {}},
            ]
        )
        asyncio.run(bridge.handle_websocket(ws))
        self.assertTrue(ws.accepted)
        self.assertEqual(ws.sent[0]["error"]["code"], -32601)

    def test_unknown_thread_errors_for_turn_and_compact(self) -> None:
        bridge = app_server.CodexAppServerBridge(service=object(), state_dir=tempfile.mkdtemp())
        ws = _FakeWebSocket([])
        asyncio.run(bridge._handle_turn_start(ws, 1, {"threadId": "missing"}))
        asyncio.run(bridge._handle_thread_compact_start(ws, 2, {"threadId": "missing"}))
        self.assertEqual(ws.sent[0]["error"]["code"], -32602)
        self.assertEqual(ws.sent[1]["error"]["code"], -32602)
