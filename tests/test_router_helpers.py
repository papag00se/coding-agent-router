from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from app import models, router
from app.compaction.models import CodexHandoffFlow, SessionHandoff


class TestRouterHelpers(unittest.TestCase):
    def test_flatten_content_prompt_digest_and_workdir_helpers(self) -> None:
        self.assertEqual(router.flatten_content("hi"), "hi")
        self.assertEqual(
            router.flatten_content([{"type": "text", "text": "hello"}, {"type": "tool_use", "name": "x"}, 7]),
            'hello\n{"type": "tool_use", "name": "x"}\n7',
        )
        self.assertEqual(router.flatten_content({"x": 1}), "{'x': 1}")

        req = models.AnthropicMessagesRequest.model_validate(
            {
                "system": [{"type": "text", "text": "Be concise."}],
                "messages": [
                    {"role": "assistant", "content": "Earlier"},
                    {"role": "user", "content": [{"type": "text", "text": "Latest"}]},
                ],
            }
        )
        prompt = router.anthropic_messages_to_prompt(req)
        self.assertEqual(prompt["system"], "Be concise.")
        self.assertEqual(prompt["user_prompt"], "Latest")
        self.assertEqual(prompt["trajectory"], [{"role": "assistant", "content": "Earlier"}])

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.reasoner_num_ctx = 100
            mock_settings.coder_num_ctx = 200
            mock_settings.router_num_ctx = 50
            digest = router.build_routing_digest("sys", "prompt", metadata=None, history=[{"role": "user", "content": "hi"}])
            self.assertEqual(digest["trajectory"], [{"role": "user", "content": "hi"}])
            self.assertEqual(digest["metrics"]["reasoner_context_limit"], 100)

            mock_settings.codex_workdir = "/default"
            self.assertEqual(router._codex_workdir(None), "/default")
            self.assertEqual(router._codex_workdir({"cwd": "/cwd"}), "/cwd")
            self.assertEqual(router._codex_workdir({"workdir": "/work"}), "/work")
            self.assertEqual(router._codex_workdir({"project_path": "/project"}), "/project")
            self.assertEqual(router._codex_workdir({"repo_path": "/repo"}), "/repo")
            self.assertEqual(router._codex_workdir({"repo_context": {"cwd": "/nested"}}), "/nested")
            self.assertEqual(router._codex_workdir({"repo_context": {"cwd": 7}}), "/default")

    def test_route_covers_no_backends_single_backend_and_parse_fallback(self) -> None:
        service = router.RoutingService.__new__(router.RoutingService)
        service.router_client = type("DummyRouter", (), {"chat": lambda *_args, **_kwargs: {"message": {"content": "not json"}}})()
        service.codex_client = None

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.enable_local_coder = False
            mock_settings.enable_local_reasoner = False
            mock_settings.enable_codex_cli = False
            mock_settings.reasoner_num_ctx = 100
            mock_settings.coder_num_ctx = 100
            mock_settings.router_num_ctx = 100
            decision = service.route("", "short")
        self.assertEqual(decision.reason, "no configured backends")

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.reasoner_num_ctx = 100
            mock_settings.coder_num_ctx = 100
            mock_settings.router_num_ctx = 100
            preferred = service.route("", "short", preferred_backend="codex_cli")
        self.assertEqual(preferred.reason, "preferred backend override")

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.enable_local_coder = True
            mock_settings.enable_local_reasoner = False
            mock_settings.enable_codex_cli = False
            mock_settings.reasoner_num_ctx = 100
            mock_settings.coder_num_ctx = 100
            mock_settings.router_num_ctx = 5000
            decision = service.route("", "short")
        self.assertEqual(decision.reason, "only one route fits context limits")
        self.assertEqual(decision.route, "local_coder")

        service.codex_client = object()
        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.enable_local_coder = True
            mock_settings.enable_local_reasoner = True
            mock_settings.enable_codex_cli = True
            mock_settings.router_model = "router-model"
            mock_settings.router_temperature = 0.0
            mock_settings.reasoner_num_ctx = 1000
            mock_settings.coder_num_ctx = 1000
            mock_settings.router_num_ctx = 5000
            decision = service.route("", "short")
        self.assertEqual(decision.reason, "router JSON parse fallback")
        self.assertEqual(decision.route, "local_coder")

    def test_route_covers_invalid_router_choice_and_local_context_exceeded(self) -> None:
        class DummyRouterClient:
            def chat(self, *_args, **_kwargs):
                return {"message": {"content": json.dumps({"route": "unknown", "confidence": 0.4, "reason": "bad"})}}

        service = router.RoutingService.__new__(router.RoutingService)
        service.router_client = DummyRouterClient()
        service.codex_client = object()

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.enable_local_coder = True
            mock_settings.enable_local_reasoner = True
            mock_settings.enable_codex_cli = True
            mock_settings.router_model = "router-model"
            mock_settings.router_temperature = 0.0
            mock_settings.reasoner_num_ctx = 1000
            mock_settings.coder_num_ctx = 1000
            mock_settings.router_num_ctx = 5000
            decision = service.route("", "short")
        self.assertEqual(decision.route, "local_coder")

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.enable_local_coder = True
            mock_settings.enable_local_reasoner = True
            mock_settings.enable_codex_cli = False
            mock_settings.router_model = "router-model"
            mock_settings.router_temperature = 0.0
            mock_settings.reasoner_num_ctx = 10
            mock_settings.coder_num_ctx = 10
            mock_settings.router_num_ctx = 5000
            decision = service.route("", "x" * 1000)
        self.assertEqual(decision.reason, "local context windows exceeded")
        self.assertEqual(decision.route, "local_coder")

    def test_compaction_wrappers_and_inline_compaction_paths(self) -> None:
        class DummyCompactionService:
            def compact_transcript(self, *args, **kwargs):
                return SessionHandoff(stable_task_definition="compact")

            def refresh_if_needed(self, *args, **kwargs):
                return SessionHandoff(stable_task_definition="refresh")

            def load_latest_handoff(self, session_id):
                if session_id == "found":
                    return SessionHandoff(stable_task_definition="loaded")
                return None

            def build_codex_handoff_flow(self, session_id, *, current_request=None):
                if session_id == "found":
                    return CodexHandoffFlow(current_request=current_request or "")
                return None

        class DummyCompactorClient:
            def chat(self, *args, **kwargs):
                return {"message": {"content": "summary"}, "prompt_eval_count": 5, "eval_count": 2}

            def chat_stream(self, *args, **kwargs):
                yield {"message": {"content": "sum"}, "prompt_eval_count": 5}
                yield {"message": {"content": "mary"}, "eval_count": 2}

        service = router.RoutingService.__new__(router.RoutingService)
        service.compaction_service = DummyCompactionService()
        service.compactor_client = DummyCompactorClient()

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.compactor_model = "compact-model"
            mock_settings.compactor_temperature = 0.0
            mock_settings.compactor_num_ctx = 1000
            mock_settings.inline_compact_sentinel = "<<<LOCAL_COMPACT>>>"

            compacted = service.compact_session("s1", [{"role": "user", "content": "hi"}], current_request="do it")
            refreshed = service.compact_session("s1", [{"role": "user", "content": "hi"}], current_request="do it", refresh_if_needed=True)
            self.assertEqual(compacted["stable_task_definition"], "compact")
            self.assertEqual(refreshed["stable_task_definition"], "refresh")
            self.assertEqual(service.load_compaction_handoff("missing"), None)
            self.assertEqual(service.load_compaction_handoff("found")["stable_task_definition"], "loaded")
            self.assertEqual(service.build_compaction_handoff_flow("missing"), None)
            self.assertEqual(service.build_compaction_handoff_flow("found", current_request="next")["current_request"], "next")

            req = models.AnthropicMessagesRequest.model_validate(
                {
                    "system": "<<<LOCAL_COMPACT>>> compact this",
                    "messages": [{"role": "user", "content": "<<<LOCAL_COMPACT>>> payload"}],
                    "metadata": {"source": "test"},
                }
            )
            inline = service.invoke_inline_compact_from_anthropic(req)
            self.assertEqual(inline["content"][0]["text"], "summary")
            self.assertEqual(inline["usage"]["input_tokens"], 5)

            streamed = list(service.stream_inline_compact_from_anthropic(req))
            self.assertEqual([event["type"] for event in streamed], ["text_delta", "text_delta", "final"])
            self.assertEqual(streamed[-1]["response"]["content"][0]["text"], "summary")

    def test_stream_from_anthropic_covers_reasoner_and_codex_routes(self) -> None:
        class DummyReasonerClient:
            def chat_stream(self, *args, **kwargs):
                yield {"message": {"content": "hello"}, "prompt_eval_count": 1, "eval_count": 1}

        service = router.RoutingService.__new__(router.RoutingService)
        service.reasoner_client = DummyReasonerClient()
        service.route = lambda *_args, **_kwargs: models.RouteDecision(route="local_reasoner", confidence=1.0, reason="test", metrics_payload={})

        req = models.AnthropicMessagesRequest.model_validate({"messages": [{"role": "user", "content": "Say hi"}]})
        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.coder_model = "coder"
            mock_settings.reasoner_model = "reasoner"
            mock_settings.reasoner_temperature = 0.0
            mock_settings.reasoner_num_ctx = 1000
            events = list(service.stream_from_anthropic(req))
        self.assertEqual(events[0], {"type": "text_delta", "delta": "hello"})
        self.assertEqual(events[-1]["response"]["model"], "reasoner")

        service.route = lambda *_args, **_kwargs: models.RouteDecision(route="codex_cli", confidence=1.0, reason="test", metrics_payload={})
        service.invoke_from_anthropic = lambda _req: {"model": "codex-cli", "content": [{"type": "text", "text": "done"}]}
        events = list(service.stream_from_anthropic(req))
        self.assertEqual(events, [{"type": "final", "response": {"model": "codex-cli", "content": [{"type": "text", "text": "done"}]}}])

    def test_stream_route_dispatch_and_normalizers_cover_remaining_branches(self) -> None:
        class DummyStreamClient:
            def chat_stream(self, *args, **kwargs):
                yield {
                    "message": {
                        "content": '{"tool_calls":[{"name":"run_shell","arguments":"not-json"}],"content":"prefix"}'
                    },
                    "prompt_eval_count": 1,
                }
                yield {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "run_shell", "arguments": {"cmd": "pwd"}}},
                            {"function": {"name": "run_shell", "arguments": {"cmd": "pwd"}}},
                            {"name": "", "arguments": {}},
                        ]
                    },
                    "eval_count": 1,
                }

        service = router.RoutingService.__new__(router.RoutingService)
        decision = models.RouteDecision(route="local_coder", confidence=1.0, reason="test", metrics_payload={})
        events = list(
            service._stream_ollama_route(
                decision,
                DummyStreamClient(),
                "coder",
                0.0,
                1000,
                "sys",
                [{"role": "user", "content": "prompt"}],
                tools=[],
                structured=True,
                devstral=False,
                request_metadata={},
            )
        )
        self.assertIn("tool_calls", {event["type"] for event in events})
        self.assertEqual(events[-1]["response"]["stop_reason"], "tool_use")

        self.assertEqual(router._normalize_stream_tool_calls(None), [])
        self.assertEqual(
            router._normalize_stream_tool_calls([{"function": {"name": "run_shell", "arguments": "oops"}}])[0]["function"]["arguments"],
            {"raw": "oops"},
        )
        self.assertEqual(
            router._normalize_stream_tool_calls([{"name": "tool", "arguments": 7}])[0]["function"]["arguments"],
            {"value": 7},
        )

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.inline_compact_sentinel = "<<<LOCAL_COMPACT>>>"
            self.assertEqual(router._strip_compaction_sentinel("<<<LOCAL_COMPACT>>> hello"), "hello")
            self.assertEqual(router._strip_compaction_sentinel("plain"), "plain")

        self.assertEqual(router._normalize_stream_tool_calls(["ignore"]), [])
        self.assertEqual(router._normalize_stream_tool_calls([{"function": "bad"}]), [])

        self.assertEqual(
            router.RoutingService.__new__(router.RoutingService)._fallback_route(["local_reasoner"]),
            "local_reasoner",
        )
        self.assertEqual(
            router.RoutingService.__new__(router.RoutingService)._fallback_route(["codex_cli"]),
            "codex_cli",
        )
        self.assertEqual(
            router.RoutingService.__new__(router.RoutingService)._fallback_route([]),
            "local_reasoner",
        )

    def test_dispatch_covers_reasoner_codex_and_unknown_route(self) -> None:
        class DummyReasonerClient:
            def chat(self, *args, **kwargs):
                return {"message": {"content": "reasoned", "thinking": "trace"}}

        service = router.RoutingService.__new__(router.RoutingService)
        service.reasoner_client = DummyReasonerClient()
        service.codex_client = None
        service.compaction_service = type("Compaction", (), {"build_codex_handoff_flow": lambda *_args, **_kwargs: None})()

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.reasoner_model = "reasoner"
            mock_settings.reasoner_temperature = 0.0
            mock_settings.reasoner_num_ctx = 1000
            backend_model, output, thinking, _raw = service._dispatch("local_reasoner", "", "prompt")
        self.assertEqual((backend_model, output, thinking), ("reasoner", "reasoned", "trace"))

        with self.assertRaisesRegex(RuntimeError, "Codex CLI backend is not configured"):
            service._dispatch("codex_cli", "", "prompt")

        with self.assertRaisesRegex(RuntimeError, "Unknown route"):
            service._dispatch("mystery", "", "prompt")

    def test_invoke_build_codex_prompt_and_stream_inline_empty_delta_branches(self) -> None:
        service = router.RoutingService.__new__(router.RoutingService)
        service.route = lambda *_args, **_kwargs: models.RouteDecision(route="local_reasoner", confidence=1.0, reason="ok", metrics_payload={})
        service._dispatch = lambda *_args, **_kwargs: ("reasoner", "answer", "thinking", {"usage": {"input_tokens": 1, "output_tokens": 1}})
        req = models.InvokeRequest(prompt="hello")
        result = service.invoke(req)
        self.assertEqual(result.backend_model, "reasoner")

        service.compaction_service = type("Compaction", (), {"build_codex_handoff_flow": lambda *_args, **_kwargs: None})()
        self.assertEqual(service._build_codex_cli_prompt("sys", "prompt", {"session_id": "s1"}), "sys\n\nprompt")

        class SilentCompactor:
            def chat_stream(self, *args, **kwargs):
                yield {"message": {"content": ""}, "prompt_eval_count": 1}
                yield {"message": {"content": "done"}, "eval_count": 1}

        service.compactor_client = SilentCompactor()
        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.compactor_model = "compact-model"
            mock_settings.compactor_temperature = 0.0
            mock_settings.compactor_num_ctx = 1000
            mock_settings.inline_compact_sentinel = "<<<LOCAL_COMPACT>>>"
            events = list(
                service.stream_inline_compact_from_anthropic(
                    models.AnthropicMessagesRequest.model_validate({"messages": [{"role": "user", "content": "<<<LOCAL_COMPACT>>> hi"}]})
                )
            )
        self.assertEqual([event["type"] for event in events], ["text_delta", "final"])

    def test_stream_route_handles_non_prefix_recovery(self) -> None:
        class DummyStreamClient:
            def chat_stream(self, *args, **kwargs):
                yield {"message": {"content": "chunk-1"}, "prompt_eval_count": 1}
                yield {"message": {"content": "chunk-2"}, "eval_count": 1}

        service = router.RoutingService.__new__(router.RoutingService)
        decision = models.RouteDecision(route="local_coder", confidence=1.0, reason="test", metrics_payload={})

        def fake_recover(message):
            if message.get("content") == "chunk-1chunk-2":
                return {"content": "mismatch", "tool_calls": []}
            return {"content": "", "tool_calls": []}

        with patch.object(router, "recover_stream_ollama_message", side_effect=fake_recover):
            events = list(
                service._stream_ollama_route(
                    decision,
                    DummyStreamClient(),
                    "coder",
                    0.0,
                    1000,
                    "sys",
                    [{"role": "user", "content": "prompt"}],
                    tools=[],
                    structured=True,
                    devstral=False,
                    request_metadata={},
                )
            )
        self.assertEqual(events[-1]["response"]["content"], [{"type": "text", "text": "chunk-1chunk-2"}])
