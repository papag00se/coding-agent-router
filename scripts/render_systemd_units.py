from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deploy" / "systemd"
USER = "jesse"
GROUP = "jesse"
WORKDIR = str(ROOT)
UVICORN = str(ROOT / ".venv" / "bin" / "uvicorn")
CODEX = str(Path.home() / ".nvm" / "versions" / "node" / "v22.13.1" / "bin" / "codex")
COMMON_PATH = f"{Path(CODEX).parent}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def render_unit(*, name: str, module: str, port: int, app_server_state_dir: str) -> str:
    return f"""[Unit]
Description={name}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={USER}
Group={GROUP}
WorkingDirectory={WORKDIR}
Environment=PATH={COMMON_PATH}
Environment=PYTHONUNBUFFERED=1
Environment=HOST=127.0.0.1
Environment=PORT={port}
Environment=ROUTER_OLLAMA_BASE_URL=http://127.0.0.1:21435
Environment=ROUTER_MODEL=qwen3:8b
Environment=CODER_OLLAMA_BASE_URL=http://127.0.0.1:21435
Environment=CODER_MODEL=devstral2:24b-iq4_xs
Environment=REASONER_OLLAMA_BASE_URL=http://127.0.0.1:21435
Environment=REASONER_MODEL=qwen3.5-9b:iq4_xs
Environment=COMPACTOR_OLLAMA_BASE_URL=http://127.0.0.1:21435
Environment=COMPACTOR_MODEL=qwen3.5-9b:iq4_xs
Environment=ENABLE_CODEX_CLI=true
Environment=CODEX_CMD={CODEX}
Environment=CODEX_WORKDIR={WORKDIR}
Environment=APP_SERVER_STATE_DIR={app_server_state_dir}
ExecStart={UVICORN} {module}:app --host 127.0.0.1 --port {port}
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
"""


def write_units() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "coding-agent-router.service").write_text(
        render_unit(
            name="Coding Agent Router",
            module="app.main",
            port=8080,
            app_server_state_dir=f"{WORKDIR}/state/app_server_full",
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "coding-agent-router-compaction.service").write_text(
        render_unit(
            name="Coding Agent Router Compaction Service",
            module="app.compaction_main",
            port=8081,
            app_server_state_dir=f"{WORKDIR}/state/app_server_compaction",
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_units()
