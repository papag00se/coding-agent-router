from __future__ import annotations

import unittest
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
