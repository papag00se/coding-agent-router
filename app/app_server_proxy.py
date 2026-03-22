from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any, Callable, Optional, TextIO


JsonTransform = Callable[[dict[str, Any]], Optional[dict[str, Any]]]


class AppServerProxy:
    def __init__(
        self,
        child_cmd: list[str],
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self.child_cmd = child_cmd
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr

    def rewrite_client_message(self, message: dict[str, Any]) -> Optional[dict[str, Any]]:
        return message

    def rewrite_server_message(self, message: dict[str, Any]) -> Optional[dict[str, Any]]:
        return message

    def _rewrite_line(self, line: str, transform: JsonTransform) -> Optional[str]:
        stripped = line.strip()
        if not stripped:
            return line
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return line
        rewritten = transform(payload)
        if rewritten is None:
            return None
        return json.dumps(rewritten, ensure_ascii=False) + "\n"

    def _pump(self, reader: TextIO, writer: TextIO, transform: JsonTransform, *, close_writer: bool = False) -> None:
        for line in reader:
            rewritten = self._rewrite_line(line, transform)
            if rewritten is None:
                continue
            writer.write(rewritten)
            writer.flush()
        if close_writer:
            try:
                writer.close()
            except Exception:
                pass

    def run(self) -> int:
        proc = subprocess.Popen(
            self.child_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        client_thread = threading.Thread(
            target=self._pump,
            args=(self.stdin, proc.stdin, self.rewrite_client_message),
            kwargs={"close_writer": True},
            daemon=True,
        )
        server_thread = threading.Thread(
            target=self._pump,
            args=(proc.stdout, self.stdout, self.rewrite_server_message),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._pump,
            args=(proc.stderr, self.stderr, lambda payload: payload),
            daemon=True,
        )
        client_thread.start()
        server_thread.start()
        stderr_thread.start()
        client_thread.join()
        server_thread.join()
        stderr_thread.join()
        return proc.wait()
