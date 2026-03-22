from __future__ import annotations

import unittest
from unittest.mock import patch

from app import models, router
from app.tool_adapter import anthropic_messages_to_ollama, normalize_ollama_tools, recover_ollama_message


class TestDevstralToolAdapter(unittest.TestCase):
    def test_normalize_ollama_tools_converts_anthropic_schema(self) -> None:
        tools = normalize_ollama_tools(
            [
                {
                    "name": "run_shell",
                    "description": "Run a shell command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ]
        )

        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "run_shell")
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["cmd"])

    def test_anthropic_messages_to_ollama_converts_tool_use_and_results(self) -> None:
        messages = anthropic_messages_to_ollama(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will inspect that."},
                        {"type": "tool_use", "id": "toolu_1", "name": "run_shell", "input": {"cmd": "pwd"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": {"stdout": "/repo"}},
                    ],
                },
            ]
        )

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "run_shell")
        self.assertEqual(messages[1]["role"], "tool")
        self.assertIn("/repo", messages[1]["content"])

    def test_recover_ollama_message_parses_tool_call_json_from_content(self) -> None:
        message = recover_ollama_message(
            {
                "content": '{"tool_calls":[{"name":"run_shell","arguments":{"cmd":"pwd"},"id":"call_1"}],"content":"Checked the repo."}'
            }
        )

        self.assertEqual(message["tool_calls"][0]["function"]["name"], "run_shell")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"]["cmd"], "pwd")
        self.assertEqual(message["content"], "Checked the repo.")

    def test_invoke_from_anthropic_uses_adapter_only_for_devstral_coder(self) -> None:
        class DummyCoderClient:
            def __init__(self) -> None:
                self.messages = None
                self.tools = None

            def chat(self, *_args, **kwargs):
                self.messages = _args[1]
                self.tools = kwargs.get("tools")
                return {
                    "message": {
                        "content": '{"tool_calls":[{"name":"run_shell","arguments":{"cmd":"pwd"},"id":"call_1"}],"content":""}'
                    }
                }

        service = router.RoutingService.__new__(router.RoutingService)
        service.router_client = None
        service.coder_client = DummyCoderClient()
        service.reasoner_client = None
        service.codex_client = None
        service.route = lambda *_args, **_kwargs: models.RouteDecision(route="local_coder", confidence=1.0, reason="", metrics_payload={})

        req = models.AnthropicMessagesRequest(
            messages=[models.Message(role="user", content=[{"type": "text", "text": "Inspect the repo"}])],
            tools=[
                {
                    "name": "run_shell",
                    "description": "Run a shell command",
                    "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                }
            ],
        )

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.coder_model = "devstral:24b"
            mock_settings.coder_temperature = 0.1
            mock_settings.coder_num_ctx = 16384
            response = service.invoke_from_anthropic(req)

        self.assertEqual(service.coder_client.tools[0]["function"]["name"], "run_shell")
        self.assertEqual(service.coder_client.messages[0]["role"], "user")
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["content"][0]["type"], "tool_use")

    def test_invoke_from_anthropic_keeps_plain_text_response_for_non_devstral_coder(self) -> None:
        class DummyCoderClient:
            def __init__(self) -> None:
                self.messages = None
                self.tools = None

            def chat(self, *_args, **kwargs):
                self.messages = _args[1]
                self.tools = kwargs.get("tools")
                return {"message": {"content": "plain text answer"}}

        service = router.RoutingService.__new__(router.RoutingService)
        service.router_client = None
        service.coder_client = DummyCoderClient()
        service.reasoner_client = None
        service.codex_client = None
        service.route = lambda *_args, **_kwargs: models.RouteDecision(route="local_coder", confidence=1.0, reason="", metrics_payload={})

        req = models.AnthropicMessagesRequest(
            messages=[models.Message(role="user", content=[{"type": "text", "text": "Inspect the repo"}])],
            tools=[
                {
                    "name": "run_shell",
                    "description": "Run a shell command",
                    "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                }
            ],
        )

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.coder_model = "qwen3-coder:30b"
            mock_settings.coder_temperature = 0.1
            mock_settings.coder_num_ctx = 16384
            response = service.invoke_from_anthropic(req)

        self.assertEqual(service.coder_client.tools[0]["function"]["name"], "run_shell")
        self.assertEqual(service.coder_client.messages[0]["role"], "user")
        self.assertEqual(service.coder_client.messages[0]["content"], "Inspect the repo")
        self.assertEqual(response["stop_reason"], "end_turn")
        self.assertEqual(response["content"], [{"type": "text", "text": "plain text answer"}])
