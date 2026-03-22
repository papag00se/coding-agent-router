#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.app_server_proxy import AppServerProxy


def _stock_codex_cmd() -> str:
    return os.environ.get("CODEX_STOCK_CMD") or os.environ.get("CODEX_CMD") or "codex"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    stock = _stock_codex_cmd()
    if args[:1] == ["app-server"]:
        proxy = AppServerProxy([stock, *args])
        return proxy.run()
    os.execvp(stock, [stock, *args])


if __name__ == "__main__":
    raise SystemExit(main())
