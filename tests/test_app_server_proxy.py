from __future__ import annotations

import io
import json
import unittest

from app.app_server_proxy import AppServerProxy


class _DropInitializeProxy(AppServerProxy):
    def rewrite_client_message(self, message):
        if message.get("method") == "drop":
            return None
        return message


class TestAppServerProxy(unittest.TestCase):
    def test_rewrite_line_preserves_json_rpc_messages(self):
        proxy = AppServerProxy(["codex", "app-server"])
        line = '{"id":1,"method":"initialize","params":{"clientInfo":{"name":"probe"}}}\n'

        rewritten = proxy._rewrite_line(line, proxy.rewrite_client_message)

        self.assertEqual(json.loads(rewritten), json.loads(line))

    def test_rewrite_line_can_drop_messages(self):
        proxy = _DropInitializeProxy(["codex", "app-server"])
        line = '{"id":1,"method":"drop","params":{}}\n'

        rewritten = proxy._rewrite_line(line, proxy.rewrite_client_message)

        self.assertIsNone(rewritten)

    def test_rewrite_line_leaves_non_json_output_untouched(self):
        proxy = AppServerProxy(["codex", "app-server"])
        line = "plain stderr text\n"

        rewritten = proxy._rewrite_line(line, proxy.rewrite_server_message)

        self.assertEqual(rewritten, line)

    def test_pump_forwards_all_rewritten_lines(self):
        proxy = AppServerProxy(["codex", "app-server"], stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO())
        reader = io.StringIO('{"id":1,"method":"initialize","params":{}}\nplain text\n')
        writer = io.StringIO()

        proxy._pump(reader, writer, proxy.rewrite_client_message)

        self.assertEqual(writer.getvalue(), '{"id": 1, "method": "initialize", "params": {}}\nplain text\n')
