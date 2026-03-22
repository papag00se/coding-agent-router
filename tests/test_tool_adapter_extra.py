from __future__ import annotations

import unittest

from app.tool_adapter import (
    _looks_like_partial_tool_block,
    _parse_json_blob,
    _stringify,
    anthropic_messages_to_ollama,
    normalize_ollama_tools,
    ollama_message_to_anthropic_content,
    recover_ollama_message,
    recover_stream_ollama_message,
)


class TestToolAdapterExtra(unittest.TestCase):
    def test_normalize_tools_handles_passthrough_and_invalid_entries(self) -> None:
        tools = normalize_ollama_tools(
            [
                {"type": "function", "function": {"name": "already", "parameters": {"type": "object"}}},
                {"name": "plain"},
                {"description": "missing name"},
                "ignore",
            ]
        )
        self.assertEqual([tool["function"]["name"] for tool in tools], ["already", "plain"])
        self.assertIsNone(normalize_ollama_tools(None))

    def test_anthropic_messages_to_ollama_stringifies_unknown_blocks(self) -> None:
        messages = anthropic_messages_to_ollama(
            [
                {"role": "user", "content": {"path": "README.md"}},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Checking"},
                        {"type": "tool_use", "id": "toolu_1", "name": "run_shell", "input": {"cmd": "pwd"}},
                        {"type": "unknown", "value": 1},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "only text"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_2", "name": "set_flag", "input": {}}]},
                {"role": "user", "content": [1]},
            ]
        )
        self.assertEqual(messages[0]["content"], '{"path": "README.md"}')
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "run_shell")
        self.assertIn('"type": "unknown"', messages[1]["content"])
        self.assertEqual(messages[2]["content"], "only text")
        self.assertEqual(messages[3]["tool_calls"][0]["function"]["name"], "set_flag")
        self.assertEqual(messages[4]["content"], "1")

    def test_recover_helpers_return_original_when_not_applicable(self) -> None:
        with_tool_calls = {"content": "ignored", "tool_calls": [{"function": {"name": "x", "arguments": {}}}]}
        self.assertIs(recover_ollama_message(with_tool_calls), with_tool_calls)
        self.assertIs(recover_stream_ollama_message(with_tool_calls), with_tool_calls)
        self.assertEqual(recover_ollama_message({"content": 3}), {"content": 3})
        self.assertEqual(recover_stream_ollama_message({"content": 3}), {"content": 3})
        self.assertEqual(
            recover_stream_ollama_message({"content": '{"tool_calls":[],"content":"x"}'})["content"],
            '{"tool_calls":[],"content":"x"}',
        )

    def test_anthropic_content_conversion_coerces_arguments_and_defaults(self) -> None:
        content = ollama_message_to_anthropic_content(
            {
                "content": "plain",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "run_shell", "arguments": '{"cmd":"pwd"}'}},
                    {"id": "call_2", "name": "set_flag", "arguments": 7},
                ],
            }
        )
        self.assertEqual(content[0], {"type": "text", "text": "plain"})
        self.assertEqual(content[1]["input"], {"cmd": "pwd"})
        self.assertEqual(content[2]["input"], {"value": 7})
        self.assertEqual(ollama_message_to_anthropic_content({"content": ""}), [{"type": "text", "text": ""}])

    def test_private_helpers_cover_fences_partial_blocks_and_stringify(self) -> None:
        self.assertEqual(_parse_json_blob("not json"), None)
        self.assertEqual(_parse_json_blob("```\n{\"a\":1}\n```"), {"a": 1})
        self.assertEqual(_parse_json_blob("```\n{\"a\":1}"), None)
        self.assertTrue(_looks_like_partial_tool_block('[USER]\n{"type":"tool_use","name":"run_shell"'))
        self.assertFalse(_looks_like_partial_tool_block("plain"))
        self.assertEqual(_stringify(None), "")
        self.assertEqual(_stringify("hi"), "hi")
        self.assertEqual(_stringify({"a": 1}), '{"a": 1}')
