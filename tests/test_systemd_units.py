from __future__ import annotations

import unittest

from app.config import settings
from scripts.render_systemd_units import render_unit


class TestSystemdUnits(unittest.TestCase):
    def test_full_router_unit_uses_main_app_and_port_8080(self):
        unit = render_unit(
            name="Coding Agent Router",
            module="app.main",
            port=8080,
            app_server_state_dir="/tmp/app_server_full",
        )
        self.assertIn("Description=Coding Agent Router", unit)
        self.assertIn("ExecStart=/home/jesse/src/coding-agent-router/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080", unit)
        self.assertIn("Environment=APP_SERVER_STATE_DIR=/tmp/app_server_full", unit)
        self.assertIn("Environment=CODEX_CMD=/home/jesse/.nvm/versions/node/v22.13.1/bin/codex", unit)

    def test_compaction_unit_uses_compaction_app_and_port_8081(self):
        unit = render_unit(
            name="Coding Agent Router Compaction Service",
            module="app.compaction_main",
            port=8081,
            app_server_state_dir="/tmp/app_server_compaction",
        )
        self.assertIn("Description=Coding Agent Router Compaction Service", unit)
        self.assertIn("ExecStart=/home/jesse/src/coding-agent-router/.venv/bin/uvicorn app.compaction_main:app --host 127.0.0.1 --port 8081", unit)
        self.assertIn("Environment=APP_SERVER_STATE_DIR=/tmp/app_server_compaction", unit)
        self.assertIn("Environment=REASONER_MODEL=qwen3.5-9b:iq4_xs", unit)


if __name__ == "__main__":
    unittest.main()
