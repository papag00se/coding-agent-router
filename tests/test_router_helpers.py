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
            def __init__(self):
                self.compact_calls = []

            def compact_transcript(self, *args, **kwargs):
                self.compact_calls.append((args, kwargs))
                items = args[1] if len(args) > 1 else kwargs["items"]
                return SessionHandoff(
                    stable_task_definition="fix hydration mismatch",
                    key_decisions=["prefer the shared compacted flow renderer"],
                    unresolved_work=["rerun the browser test"],
                    current_request=kwargs["current_request"],
                    recent_raw_turns=items[-1:],
                )

            def refresh_if_needed(self, *args, **kwargs):
                return SessionHandoff(stable_task_definition="refresh")

            def load_latest_handoff(self, session_id):
                if session_id == "found":
                    return SessionHandoff(stable_task_definition="loaded")
                return None

            def build_codex_handoff_flow(self, session_id, *, current_request=None):
                if session_id in {"found", "thread-1"}:
                    return CodexHandoffFlow(
                        durable_memory=[
                            {"name": "TASK_STATE.md", "content": "# Task State\n- fix hydration mismatch\n"},
                            {"name": "NEXT_STEPS.md", "content": "# Next Steps\n- rerun the browser test\n"},
                        ],
                        structured_handoff={"stable_task_definition": "fix hydration mismatch"},
                        recent_raw_turns=[{"role": "assistant", "content": "Investigated header rendering."}],
                        current_request=current_request or "",
                    )
                return None

        service = router.RoutingService.__new__(router.RoutingService)
        service.compaction_service = DummyCompactionService()

        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.compactor_model = "compact-model"
            mock_settings.compactor_temperature = 0.0
            mock_settings.compactor_num_ctx = 1000
            mock_settings.inline_compact_sentinel = "<<<LOCAL_COMPACT>>>"

            compacted = service.compact_session("s1", [{"role": "user", "content": "hi"}], current_request="do it")
            refreshed = service.compact_session("s1", [{"role": "user", "content": "hi"}], current_request="do it", refresh_if_needed=True)
            self.assertEqual(compacted["stable_task_definition"], "fix hydration mismatch")
            self.assertEqual(refreshed["stable_task_definition"], "refresh")
            self.assertEqual(service.load_compaction_handoff("missing"), None)
            self.assertEqual(service.load_compaction_handoff("found")["stable_task_definition"], "loaded")
            self.assertEqual(service.build_compaction_handoff_flow("missing"), None)
            self.assertEqual(service.build_compaction_handoff_flow("found", current_request="next")["current_request"], "next")

            req = models.AnthropicMessagesRequest.model_validate(
                {
                    "system": "<<<LOCAL_COMPACT>>> compact this",
                    "messages": [
                        {"role": "assistant", "content": "Investigated header rendering."},
                        {"role": "user", "content": "Fix the hydration mismatch in the header."},
                        {"role": "user", "content": "<<<LOCAL_COMPACT>>> Summarize the thread for continuation."},
                    ],
                    "metadata": {"source": "test", "session_id": "thread-1", "cwd": "/tmp/project"},
                }
            )
            inline = service.invoke_inline_compact_from_anthropic(req)
            self.assertIn("# Task State", inline["content"][0]["text"])
            self.assertIn("Fix the hydration mismatch in the header.", inline["content"][0]["text"])
            self.assertNotIn("Summarize the thread for continuation.", inline["content"][0]["text"])
            self.assertNotIn("Structured handoff:", inline["content"][0]["text"])
            self.assertNotIn("Recent raw turns:", inline["content"][0]["text"])
            self.assertIn("Structured handoff:", inline["raw_backend"]["machine_compacted_flow"])
            self.assertEqual(inline["raw_backend"]["current_request"], "Fix the hydration mismatch in the header.")
            self.assertGreater(inline["usage"]["input_tokens"], 0)
            self.assertEqual(service.compaction_service.compact_calls[-1][0][0], "thread-1")
            self.assertEqual(
                service.compaction_service.compact_calls[-1][0][1],
                [
                    {"role": "assistant", "content": "Investigated header rendering."},
                    {"role": "user", "content": "Fix the hydration mismatch in the header."},
                ],
            )

            streamed = list(service.stream_inline_compact_from_anthropic(req))
            self.assertEqual([event["type"] for event in streamed], ["text_delta", "final"])
            self.assertIn("# Task State", streamed[-1]["response"]["content"][0]["text"])

    def test_inline_compaction_inputs_preserve_structured_tool_output_turns(self) -> None:
        with patch.object(router, "settings", create=True) as mock_settings:
            mock_settings.inline_compact_sentinel = "<<<LOCAL_COMPACT>>>"
            req = models.AnthropicMessagesRequest.model_validate(
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Inspect the failing tool output."}]},
                        {
                            "role": "assistant",
                            "content": [{"type": "tool_use", "id": "call_1", "name": "run_shell", "input": {"cmd": "pwd"}}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "stdout"}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "<<<LOCAL_COMPACT>>> Summarize the thread."}],
                        },
                    ]
                }
            )

            items, current_request = router._inline_compaction_inputs(req)

        self.assertEqual(current_request, "Inspect the failing tool output.")
        self.assertEqual(items[1]["content"][0]["type"], "tool_use")
        self.assertEqual(items[2]["content"][0]["type"], "tool_result")
        self.assertNotIn("Summarize the thread.", str(items))

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
                256,
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

        class InlineCompactionService:
            def compact_transcript(self, session_id, items, *, current_request, repo_context=None, progress_callback=None):
                return SessionHandoff(current_request=current_request, recent_raw_turns=items[-1:])

            def build_codex_handoff_flow(self, session_id, *, current_request=None):
                return CodexHandoffFlow(
                    durable_memory=[{"name": "TASK_STATE.md", "content": "# Task State\n- done\n"}],
                    structured_handoff={},
                    recent_raw_turns=[],
                    current_request=current_request or "",
                )

        service.compaction_service = InlineCompactionService()
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
                    256,
                    "sys",
                    [{"role": "user", "content": "prompt"}],
                    tools=[],
                    structured=True,
                    devstral=False,
                    request_metadata={},
                )
            )
        self.assertEqual(events[-1]["response"]["content"], [{"type": "text", "text": "chunk-1chunk-2"}])
