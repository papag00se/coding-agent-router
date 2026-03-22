from __future__ import annotations

import unittest

from app.tool_adapter import recover_ollama_message, recover_stream_ollama_message


class TestToolAdapter(unittest.TestCase):
    def test_recover_embedded_tool_use_blocks_from_text(self):
        message = {
            "role": "assistant",
            "content": (
                "I'll search for it.\n\n"
                '[USER]\n'
                '{"type":"tool_use","id":"call_1k9m2p3q","name":"rg","input":{"pattern":"Local Agent Router Starter"}}\n\n'
                '[USER]\n'
                '{"type":"tool_result","tool_use_id":"call_1k9m2p3q","content":"No matches found"}'
            ),
        }

        recovered = recover_ollama_message(message)

        self.assertEqual(recovered["content"], "I'll search for it.")
        self.assertEqual(recovered["tool_calls"][0]["id"], "call_1k9m2p3q")
        self.assertEqual(recovered["tool_calls"][0]["function"]["name"], "rg")
        self.assertEqual(
            recovered["tool_calls"][0]["function"]["arguments"],
            {"pattern": "Local Agent Router Starter"},
        )

    def test_stream_recovery_holds_back_partial_embedded_tool_use(self):
        message = {
            "role": "assistant",
            "content": (
                "I'll update the file.\n\n"
                '{"type":"tool_use","id":"call_partial","name":"run_shell","input":{"command":"sed -i'
            ),
        }

        recovered = recover_stream_ollama_message(message)

        self.assertEqual(recovered["content"], "I'll update the file.")
        self.assertNotIn("tool_calls", recovered)
