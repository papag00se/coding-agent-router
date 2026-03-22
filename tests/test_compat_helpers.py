from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import compat


def _event_names(payload: str) -> list[str]:
    names: list[str] = []
    for block in payload.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("event: "):
                names.append(line.removeprefix("event: "))
    return names


class TestCompatHelpers(unittest.TestCase):
    def test_text_blocks_tool_input_and_anthropic_tools_handle_edge_cases(self) -> None:
        self.assertEqual(compat._text_blocks("hello"), [{"type": "text", "text": "hello"}])
        self.assertEqual(
            compat._text_blocks([{"type": "input_text", "input_text": "hi"}]),
            [{"type": "text", "text": "hi"}],
        )
        self.assertEqual(
            compat._text_blocks([{"type": "image", "url": "https://example.test/cat.png"}]),
            [{"type": "text", "text": '{"type": "image", "url": "https://example.test/cat.png"}'}],
        )
        self.assertEqual(compat._tool_input({"path": "a.txt"}), {"path": "a.txt"})
        self.assertEqual(compat._tool_input('{"path":"a.txt"}'), {"path": "a.txt"})
        self.assertEqual(compat._tool_input("oops"), {"input": "oops"})
        self.assertEqual(compat._tool_input(7), {})

        tools = compat._anthropic_tools(
            [
                "ignore",
                {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
                {"name": "plain_tool"},
                {"type": "function", "function": {"description": "missing name"}},
            ]
        )
        self.assertEqual([tool["name"] for tool in tools], ["read_file", "plain_tool"])

    def test_anthropic_messages_converts_system_tool_and_assistant_calls(self) -> None:
        system, messages = compat._anthropic_messages(
            [
                {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
                {"role": "tool", "tool_call_id": "call_1", "content": {"stdout": "ok"}},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Checking."}],
                    "tool_calls": [
                        {"id": "call_2", "function": {"name": "run_shell", "arguments": '{"cmd":"pwd"}'}},
                        {"id": "ignored", "function": {"description": "missing name"}},
                    ],
                },
                {"role": "user", "content": "Continue"},
            ]
        )

        self.assertEqual(system, "Be concise.")
        self.assertEqual(messages[0]["content"][0]["type"], "tool_result")
        self.assertEqual(messages[1]["content"][0]["type"], "text")
        self.assertEqual(messages[1]["content"][1]["type"], "tool_use")
        self.assertEqual(messages[1]["content"][1]["input"], {"cmd": "pwd"})
        self.assertEqual(messages[2], {"role": "user", "content": "Continue"})

    def test_request_converters_cover_openai_ollama_and_responses_shapes(self) -> None:
        openai_req = compat.anthropic_request_from_openai_chat(
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Say ok"},
                ],
                "max_completion_tokens": 12,
            }
        )
        self.assertEqual(openai_req.system, "Be concise.")
        self.assertEqual(openai_req.max_tokens, 12)

        ollama_req = compat.anthropic_request_from_ollama_chat(
            {
                "messages": [{"role": "user", "content": "Say ok"}],
                "options": {"num_predict": 9},
            }
        )
        self.assertEqual(ollama_req.max_tokens, 9)

        responses_req = compat.anthropic_request_from_responses(
            {
                "model": "gpt-5.4",
                "instructions": "Top level instructions",
                "metadata": {"source": "test"},
                "tools": [
                    {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
                    {"type": "custom", "name": "apply_patch"},
                ],
                "input": [
                    {"type": "function_call_output", "call_id": "call_out", "output": {"stdout": "done"}},
                    {"type": "function_call", "call_id": "call_in", "name": "search_text", "arguments": '{"pattern":"foo"}'},
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Developer note"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Looking."},
                            {"type": "function_call", "call_id": "call_nested", "name": "run_shell", "arguments": '{"command":"pwd"}'},
                        ],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Continue"},
                            {"type": "function_call_output", "call_id": "call_nested", "output": "ok"},
                        ],
                    },
                ],
            }
        )
        self.assertIn("Top level instructions", responses_req.system)
        self.assertIn("Developer note", responses_req.system)
        self.assertEqual(responses_req.messages[0].content[0]["type"], "tool_result")
        self.assertEqual(responses_req.messages[1].content[0]["type"], "tool_use")
        self.assertEqual(responses_req.messages[2].content[1]["type"], "tool_use")
        self.assertEqual(responses_req.messages[3].content[1]["type"], "tool_result")
        self.assertEqual([tool["name"] for tool in responses_req.tools], ["run_shell", "write_patch"])
        self.assertIn("_tool_aliases", responses_req.metadata)

    def test_model_cache_route_banner_and_created_at_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "models_cache.json"
            cache_path.write_text(json.dumps({"models": [{"slug": "gpt-5.4", "owned_by": "cached-owner"}]}), encoding="utf-8")
            with patch.object(compat, "_CODEX_MODELS_CACHE_PATH", cache_path):
                body = compat.openai_models_response()
            self.assertEqual(body["data"][0]["id"], "gpt-5.4")
            self.assertEqual(body["data"][0]["owned_by"], "cached-owner")

            bad_path = Path(tmpdir) / "bad.json"
            bad_path.write_text("{not-json", encoding="utf-8")
            with patch.object(compat, "_CODEX_MODELS_CACHE_PATH", bad_path):
                self.assertIsNone(compat._codex_model_entry("gpt-5.4"))
            with patch.object(compat, "_CODEX_MODELS_CACHE_PATH", cache_path):
                self.assertIsNone(compat._codex_model_entry("missing"))
            with patch.object(compat, "_CODEX_MODELS_CACHE_PATH", Path(tmpdir) / "absent.json"):
                self.assertEqual(compat.openai_models_response()["data"][0]["id"], "gpt-5.4")

        self.assertEqual(compat._route_banner({"route_decision": {"route": "local_coder"}, "model": "qwen"}), "[router] local_coder -> qwen")
        self.assertEqual(compat._route_banner({"model": "qwen"}), "")
        self.assertEqual(compat._response_created_at({"created_at": 55}), 55)
        with patch("app.compat.time.time", return_value=77):
            self.assertEqual(compat._response_created_at({"created_at": "not-a-date"}), 77)

    def test_responses_converter_covers_non_dict_and_string_message_paths(self) -> None:
        req = compat.anthropic_request_from_responses(
            {
                "input": [
                    "ignore",
                    {"type": "other"},
                    {"type": "message", "role": "system", "content": []},
                    {"type": "message", "role": "user", "content": "plain text"},
                    {"type": "message", "role": "assistant", "content": ["ignore", {"type": "input_text", "text": ""}]},
                ]
            }
        )
        self.assertEqual(req.messages[0].content, "plain text")
        self.assertEqual(req.messages[1].content, "")

    def test_response_renderers_cover_tool_only_shapes(self) -> None:
        response = {
            "id": "resp_1",
            "model": "local_coder",
            "content": [{"type": "tool_use", "id": "call_1", "name": "run_shell", "input": {"cmd": "pwd"}}],
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "request_metadata": {
                "_tool_aliases": {
                    "run_shell": {"original_name": "exec_command", "mode": "exec_command"},
                }
            },
        }
        rendered = compat.responses_response(response, {"model": "gpt-5.4"})
        self.assertEqual(rendered["output_text"], "")
        self.assertEqual([item["type"] for item in rendered["output"]], ["function_call"])
        self.assertTrue(rendered["parallel_tool_calls"])
        self.assertEqual(json.loads(rendered["output"][0]["arguments"]), {"cmd": ""})

        chat = compat.openai_chat_response(response)
        self.assertEqual(chat["choices"][0]["finish_reason"], "tool_calls")
        self.assertIsNone(chat["choices"][0]["message"]["content"])

        streamed = "".join(compat.iter_openai_chat_response(response))
        self.assertIn('"finish_reason": "tool_calls"', streamed)
        self.assertIn("[DONE]", streamed)

        ollama = compat.ollama_chat_response(response)
        self.assertEqual(ollama["done_reason"], "tool_calls")
        self.assertEqual(ollama["message"]["tool_calls"][0]["function"]["name"], "run_shell")
        self.assertTrue(next(compat.iter_ollama_chat_response(response)).endswith("\n"))

    def test_iter_responses_response_emits_expected_event_sequence(self) -> None:
        response = {
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "README.md"}},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

        payload = "".join(compat.iter_responses_response(response, {"model": "gpt-5.4"}))
        names = _event_names(payload)
        self.assertIn("response.created", names)
        self.assertIn("response.output_text.delta", names)
        self.assertIn("response.function_call_arguments.done", names)
        self.assertEqual(names[-1], "response.completed")

    def test_iter_responses_progress_covers_tool_streaming_finalization_and_errors(self) -> None:
        payload = "".join(
            compat.iter_responses_progress(
                [
                    {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "run_shell", "arguments": {"command": "pwd"}},
                            }
                        ],
                    },
                    {
                        "type": "final",
                        "response": {
                            "model": "local_coder",
                            "content": [
                                {"type": "tool_use", "id": "call_1", "name": "run_shell", "input": {"command": "pwd"}}
                            ],
                            "route_decision": {"route": "local_coder"},
                            "request_metadata": {
                                "_tool_aliases": {
                                    "run_shell": {"original_name": "exec_command", "mode": "exec_command"},
                                }
                            },
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    },
                ],
                {
                    "model": "gpt-5.4",
                    "tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}],
                },
            )
        )
        names = _event_names(payload)
        self.assertIn("response.function_call_arguments.done", names)
        self.assertEqual(names[-1], "response.completed")
        self.assertIn("exec_command", payload)

        payload_with_text = "".join(
            compat.iter_responses_progress(
                [
                    {"type": "text_delta", "delta": "Partial answer"},
                    {
                        "type": "final",
                        "response": {
                            "model": "local_reasoner",
                            "content": [{"type": "text", "text": "Partial answer"}],
                            "route_decision": {"route": "local_reasoner"},
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    },
                ],
                {"model": "gpt-5.4"},
            )
        )
        self.assertIn("Partial answer", payload_with_text)
        self.assertIn("[router] local_reasoner -> local_reasoner", payload_with_text)

        with self.assertRaisesRegex(RuntimeError, "unexpected event type"):
            list(compat.iter_responses_progress([{"type": "nope"}], {"model": "gpt-5.4"}))

    def test_iter_responses_progress_covers_empty_generator_and_final_output_paths(self) -> None:
        empty_payload = "".join(compat.iter_responses_progress([], {"model": "gpt-5.4"}))
        self.assertEqual(_event_names(empty_payload), ["response.created", "response.in_progress"])

        final_without_text = "".join(
            compat.iter_responses_progress(
                [
                    {
                        "type": "final",
                        "response": {
                            "model": "local_reasoner",
                            "content": [{"type": "text", "text": "done"}],
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    }
                ],
                {"model": "gpt-5.4"},
            )
        )
        self.assertIn("response.output_item.done", final_without_text)
        self.assertIn('"text": "done"', final_without_text)

        final_with_text_and_tool = "".join(
            compat.iter_responses_progress(
                [
                    {"type": "text_delta", "delta": "prefix"},
                    {
                        "type": "final",
                        "response": {
                            "model": "local_coder",
                            "content": [
                                {"type": "text", "text": "prefix"},
                                {"type": "tool_use", "id": "call_2", "name": "read_file", "input": {"path": "README.md"}},
                            ],
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    },
                ],
                {"model": "gpt-5.4"},
            )
        )
        self.assertIn("response.function_call_arguments.done", final_with_text_and_tool)
        self.assertIn('"call_id": "call_2"', final_with_text_and_tool)
