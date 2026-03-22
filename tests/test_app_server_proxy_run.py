from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from app.app_server_proxy import AppServerProxy


class _ClosableStringIO(io.StringIO):
    pass


class _CloseErrorWriter:
    def write(self, _value):
        return None

    def flush(self):
        return None

    def close(self):
        raise RuntimeError("boom")


class _FakeProc:
    def __init__(self):
        self.stdin = _ClosableStringIO()
        self.stdout = io.StringIO('{"id":2,"result":{"ok":true}}\n')
        self.stderr = io.StringIO('{"level":"warn"}\n')

    def wait(self):
        return 7


class TestAppServerProxyRun(unittest.TestCase):
    def test_rewrite_line_preserves_blank_lines(self) -> None:
        proxy = AppServerProxy(["codex", "app-server"])
        self.assertEqual(proxy._rewrite_line("\n", proxy.rewrite_client_message), "\n")

    def test_pump_swallows_close_errors(self) -> None:
        proxy = AppServerProxy(["codex", "app-server"])
        proxy._pump(io.StringIO("plain\n"), _CloseErrorWriter(), lambda payload: payload, close_writer=True)

    def test_run_pumps_between_parent_and_child_streams(self) -> None:
        stdin = io.StringIO('{"id":1,"method":"initialize","params":{}}\n')
        stdout = io.StringIO()
        stderr = io.StringIO()
        proxy = AppServerProxy(["codex", "app-server"], stdin=stdin, stdout=stdout, stderr=stderr)

        with patch("app.app_server_proxy.subprocess.Popen", return_value=_FakeProc()):
            rc = proxy.run()

        self.assertEqual(rc, 7)
        self.assertIn('"id": 2', stdout.getvalue())
        self.assertIn('"level": "warn"', stderr.getvalue())
