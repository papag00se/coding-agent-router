#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "[run_server] python not found. Install Python 3 and rerun."
    exit 1
fi

if ! "$PYTHON_BIN" -m uvicorn --version >/dev/null 2>&1; then
    echo "[run_server] uvicorn not installed; installing requirements..."
    if ! "$PYTHON_BIN" -m pip install -r requirements.txt; then
        echo "[run_server] install on current interpreter failed; retrying in .venv ..."
        if ! command -v python3 >/dev/null 2>&1; then
            echo "[run_server] python3 not available to create .venv."
            echo "[run_server] install uvicorn manually in your environment or create a virtualenv first."
            exit 1
        fi
        python3 -m venv .venv
        if [ -x ".venv/bin/python3" ]; then
            PYTHON_BIN=".venv/bin/python3"
        elif [ -x ".venv/bin/python" ]; then
            PYTHON_BIN=".venv/bin/python"
        else
            echo "[run_server] created .venv but could not find python executable inside."
            exit 1
        fi
        "$PYTHON_BIN" -m pip install -r requirements.txt
    fi
fi

"$PYTHON_BIN" -m uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}" --reload
