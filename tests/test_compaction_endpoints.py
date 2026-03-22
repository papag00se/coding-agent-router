from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main


class DummyCompactionService:
    def __init__(self) -> None:
        self.calls = []

    def compact_session(self, session_id, items, *, current_request, repo_context=None, refresh_if_needed=False):
        self.calls.append(
            {
                "session_id": session_id,
                "items": items,
                "current_request": current_request,
                "repo_context": repo_context,
                "refresh_if_needed": refresh_if_needed,
            }
        )
        return {
            "stable_task_definition": "rename service",
            "repo_state": {"branch": "main"},
            "key_decisions": ["keep change minimal"],
            "unresolved_work": ["update app title"],
            "latest_plan": ["search", "edit", "test"],
            "failures_to_avoid": ["repeat tool leak"],
            "recent_raw_turns": items[-1:],
            "current_request": current_request,
        }


class TestCompactionEndpoints(unittest.TestCase):
    def test_internal_compact_returns_handoff(self):
        service = DummyCompactionService()
        with patch.object(main, "service", service):
            client = TestClient(main.app)
            response = client.post(
                "/internal/compact",
                json={
                    "session_id": "session-1",
                    "items": [{"role": "user", "content": "rename it"}],
                    "current_request": "finish rename",
                    "repo_context": {"repo": "coding-agent-router"},
                    "refresh_if_needed": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["stable_task_definition"], "rename service")
        self.assertEqual(body["current_request"], "finish rename")
        self.assertEqual(service.calls[0]["session_id"], "session-1")
        self.assertTrue(service.calls[0]["refresh_if_needed"])

    def test_internal_compact_can_log_before_and_after_payloads(self):
        service = DummyCompactionService()
        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transport.jsonl"
            with (
                patch.object(main, "service", service),
                patch.object(main, "_TRANSPORT_LOG_PATH", log_path),
                patch("app.compaction_transport.settings", SimpleNamespace(log_compaction_payloads=True)),
            ):
                client = TestClient(main.app)
                response = client.post(
                    "/internal/compact",
                    json={
                        "session_id": "session-2",
                        "items": [{"role": "user", "content": "rename it"}],
                        "current_request": "finish rename",
                        "repo_context": {"repo": "coding-agent-router"},
                        "refresh_if_needed": False,
                    },
                )

                events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[0]["event"], "internal_compact_start")
        self.assertEqual(events[0]["before_payload"]["session_id"], "session-2")
        self.assertEqual(events[1]["event"], "internal_compact_completed")
        self.assertEqual(events[1]["after_payload"]["stable_task_definition"], "rename service")
