from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app import config, router
from app import models
from app.compaction.models import CodexHandoffFlow


class TestRoutingPolicy(unittest.TestCase):
    def test_route_decision_rejects_unknown_backend(self):
        with self.assertRaises(ValidationError):
            models.RouteDecision(route="legacy_cloud_backend", confidence=0.0, reason="", routing_digest={})

    def test_route_decision_only_allows_expected_backends(self):
        schema = models.RouteDecision.model_json_schema()
        self.assertEqual(schema["properties"]["route"]["enum"], ["local_coder", "local_reasoner", "codex_cli"])

    def test_default_cloud_backend_is_codex_cli(self):
        self.assertEqual(config.settings.default_cloud_backend, "codex_cli")

    def test_fallback_prefers_local_backend_before_codex(self):
        service = router.RoutingService.__new__(router.RoutingService)
        route_name = service._fallback_route(["local_coder", "local_reasoner", "codex_cli"])
        self.assertEqual(route_name, "local_coder")

    def test_router_system_can_load_from_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = Path(tmpdir) / "router_system.md"
            custom_path.write_text("custom router system", encoding="utf-8")
            self.assertEqual(router._load_router_system(custom_path), "custom router system")

    def test_route_payload_includes_user_prompt_trajectory_and_numeric_metrics(self):
        class DummyClient:
            def __init__(self) -> None:
                self.prompt_payload = None
                self.system = "sentinel"

            def chat(self, *_args, **kwargs):
                self.prompt_payload = json.loads(_args[1][0]["content"])
                self.system = kwargs.get("system")
                return {
                    "message": {
                        "content": '{"route": "local_coder", "confidence": 0.93, "reason": "deterministic test"}'
                    }
                }

        service = router.RoutingService.__new__(router.RoutingService)
        client = DummyClient()
        service.router_client = client
        service.coder_client = None
        service.reasoner_client = None
        service.codex_client = object()

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.enable_local_coder = True
            mock_settings.enable_local_reasoner = True
            mock_settings.enable_codex_cli_backend = True
            mock_settings.router_model = "router-model"
            mock_settings.router_temperature = 0.0
            mock_settings.router_num_ctx = 8192
            mock_settings.coder_num_ctx = 16384
            mock_settings.reasoner_num_ctx = 12288
            decision = service.route(
                "You are a tester",
                "Build the login endpoint in auth.py",
                {
                    "router_user_prompt": "Build the login endpoint in auth.py",
                    "router_trajectory": [{"role": "assistant", "content": "Looked at auth.py"}],
                },
            )

        self.assertEqual(decision.route, "local_coder")
        self.assertIn("Choose exactly one route", client.prompt_payload["task"])
        self.assertEqual(client.prompt_payload["user_prompt"], "Build the login endpoint in auth.py")
        self.assertEqual(client.prompt_payload["trajectory"], [{"role": "assistant", "content": "Looked at auth.py"}])
        self.assertIn("router_request_tokens", client.prompt_payload["metrics"])
        self.assertIn("Return JSON only with keys: route, confidence, reason.", client.system)

    def test_router_skips_llm_when_router_payload_exceeds_context(self):
        class DummyClient:
            def chat(self, *_args, **_kwargs):
                raise AssertionError("router model should not be called")

        service = router.RoutingService.__new__(router.RoutingService)
        service.router_client = DummyClient()
        service.coder_client = None
        service.reasoner_client = None
        service.codex_client = object()

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.enable_local_coder = True
            mock_settings.enable_local_reasoner = True
            mock_settings.enable_codex_cli_backend = True
            mock_settings.router_num_ctx = 10
            mock_settings.coder_num_ctx = 16384
            mock_settings.reasoner_num_ctx = 12288
            decision = service.route("", "small prompt")

        self.assertEqual(decision.route, "codex_cli")
        self.assertEqual(decision.reason, "router request exceeds router context window")

    def test_router_removes_reasoner_when_request_exceeds_reasoner_context(self):
        class DummyClient:
            def __init__(self) -> None:
                self.prompt_payload = None

            def chat(self, *_args, **_kwargs):
                self.prompt_payload = json.loads(_args[1][0]["content"])
                return {
                    "message": {
                        "content": '{"route": "local_reasoner", "confidence": 0.5, "reason": "try reasoner"}'
                    }
                }

        service = router.RoutingService.__new__(router.RoutingService)
        client = DummyClient()
        service.router_client = client
        service.coder_client = None
        service.reasoner_client = None
        service.codex_client = object()

        long_prompt = "x" * 50000
        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.enable_local_coder = True
            mock_settings.enable_local_reasoner = True
            mock_settings.enable_codex_cli_backend = True
            mock_settings.router_model = "router-model"
            mock_settings.router_temperature = 0.0
            mock_settings.router_num_ctx = 50000
            mock_settings.coder_num_ctx = 16384
            mock_settings.reasoner_num_ctx = 12288
            decision = service.route("", long_prompt)

        self.assertEqual(decision.route, "local_coder")
        self.assertNotIn("local_reasoner", client.prompt_payload["available_routes"])

    def test_codex_cli_prompt_uses_compaction_handoff_when_session_present(self):
        class DummyCodexClient:
            def __init__(self) -> None:
                self.prompt = None
                self.workdir = None

            def exec_prompt(self, prompt: str, *, workdir: str | None = None):
                self.prompt = prompt
                self.workdir = workdir
                return {"stdout": "ok", "stderr": ""}

        class DummyCompactionService:
            def build_codex_handoff_flow(self, session_id: str, *, current_request: str = ""):
                return CodexHandoffFlow(
                    durable_memory=[
                        {"name": "TASK_STATE.md", "content": "# Task State\n- rename service\n"},
                        {"name": "DECISIONS.md", "content": "# Decisions\n- keep change minimal\n"},
                        {"name": "FAILURES_TO_AVOID.md", "content": "# Failures To Avoid\n- tool leak\n"},
                        {"name": "NEXT_STEPS.md", "content": "# Next Steps\n- update app title\n"},
                        {"name": "SESSION_HANDOFF.md", "content": "# Session Handoff\n- continue work\n"},
                    ],
                    structured_handoff={"stable_task_definition": "rename service"},
                    recent_raw_turns=[{"role": "user", "content": "old turn"}],
                    current_request=current_request,
                )

        service = router.RoutingService.__new__(router.RoutingService)
        service.codex_client = DummyCodexClient()
        service.compaction_service = DummyCompactionService()

        backend_model, output_text, thinking, raw = service._dispatch(
            "codex_cli",
            "Be concise.",
            "full flattened prompt",
            metadata={"session_id": "session-1", "router_user_prompt": "Rename the remaining files."},
        )

        self.assertEqual(backend_model, "codex-cli")
        self.assertEqual(output_text, "ok")
        self.assertIn("TASK_STATE.md", service.codex_client.prompt)
        self.assertIn("Rename the remaining files.", service.codex_client.prompt)
        self.assertNotIn("full flattened prompt", service.codex_client.prompt)
        self.assertEqual(service.codex_client.workdir, ".")

    def test_codex_cli_uses_metadata_cwd_when_present(self):
        class DummyCodexClient:
            def __init__(self) -> None:
                self.workdir = None

            def exec_prompt(self, prompt: str, *, workdir: str | None = None):
                self.workdir = workdir
                return {"stdout": "ok", "stderr": ""}

        service = router.RoutingService.__new__(router.RoutingService)
        service.codex_client = DummyCodexClient()
        service.compaction_service = None

        backend_model, output_text, thinking, raw = service._dispatch(
            "codex_cli",
            "",
            "Reply with exactly OK.",
            metadata={"cwd": "/tmp/demo-repo"},
        )

        self.assertEqual(backend_model, "codex-cli")
        self.assertEqual(output_text, "ok")
        self.assertEqual(service.codex_client.workdir, "/tmp/demo-repo")
