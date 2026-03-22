from __future__ import annotations

import json
import unittest

from app.tool_surface import map_stream_tool_call_to_original, map_tool_call_to_original, translate_responses_tools


class TestToolSurfaceExtra(unittest.TestCase):
    def test_translate_responses_tools_maps_all_supported_aliases(self) -> None:
        tools, aliases = translate_responses_tools(
            [
                {"type": "function", "name": "rg"},
                {"type": "function", "name": "exec_command"},
                {"type": "function", "name": "write_stdin"},
                {"type": "function", "name": "update_plan"},
                {"type": "function", "name": "view_image"},
                {"type": "custom", "name": "apply_patch"},
            ]
        )
        self.assertEqual(
            [tool["name"] for tool in tools],
            ["search_text", "run_shell", "send_terminal_input", "set_plan", "view_image", "write_patch"],
        )
        self.assertEqual(aliases["run_shell"]["original_name"], "exec_command")

    def test_map_tool_call_to_original_handles_unknown_and_non_dict_arguments(self) -> None:
        untouched = {"id": "x", "function": {"name": "no_alias", "arguments": {}}}
        self.assertIs(map_tool_call_to_original(untouched, {}), untouched)

        mapped = map_tool_call_to_original(
            {"id": "1", "function": {"name": "run_shell", "arguments": '{"command":"pwd","cwd":"/tmp","wait_ms":5}'}},
            {"run_shell": {"original_name": "exec_command", "mode": "exec_command"}},
        )
        self.assertEqual(json.loads(mapped["function"]["arguments"]), {"cmd": "pwd", "workdir": "/tmp", "yield_time_ms": 5})

        mapped_raw = map_tool_call_to_original(
            {"id": "2", "function": {"name": "write_patch", "arguments": "not-json"}},
            {"write_patch": {"original_name": "apply_patch", "mode": "patch"}},
        )
        self.assertEqual(json.loads(mapped_raw["function"]["arguments"]), {"input": ""})

        mapped_value = map_tool_call_to_original(
            {"id": "3", "function": {"name": "send_terminal_input", "arguments": 7}},
            {"send_terminal_input": {"original_name": "write_stdin", "mode": "write_stdin"}},
        )
        self.assertEqual(json.loads(mapped_value["function"]["arguments"]), {"session_id": None})

    def test_map_stream_tool_call_to_original_covers_modes(self) -> None:
        mapped = map_stream_tool_call_to_original(
            {"id": "call_1", "function": {"name": "send_terminal_input", "arguments": {"session_id": 9, "text": "y", "wait_ms": 7}}},
            {"send_terminal_input": {"original_name": "write_stdin", "mode": "write_stdin"}},
        )
        self.assertEqual(mapped["function"]["arguments"], {"session_id": 9, "chars": "y", "yield_time_ms": 7})

        mapped_plan = map_stream_tool_call_to_original(
            {"id": "call_2", "function": {"name": "set_plan", "arguments": {"explanation": "why", "steps": [{"step": "do", "status": "pending"}]}}},
            {"set_plan": {"original_name": "update_plan", "mode": "update_plan"}},
        )
        self.assertEqual(
            mapped_plan["function"]["arguments"],
            {"explanation": "why", "plan": [{"step": "do", "status": "pending"}]},
        )
