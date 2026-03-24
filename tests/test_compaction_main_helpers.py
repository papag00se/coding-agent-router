from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import Mock, patch

from app import compaction_main


class _DummyRequest:
    def __init__(self, headers):
        self.headers = headers


async def _collect_streaming_response(response) -> bytes:
    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    return body


class TestCompactionMainHelpers(unittest.TestCase):
    def test_sentinel_and_message_text_helpers_cover_edge_cases(self) -> None:
        with patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.inline_compact_sentinel = "<<<LOCAL_COMPACT>>>"
            self.assertTrue(compaction_main._contains_sentinel("<<<LOCAL_COMPACT>>>"))
            self.assertTrue(compaction_main._contains_sentinel(["x", {"a": "<<<LOCAL_COMPACT>>>"}]))
            self.assertFalse(compaction_main._contains_sentinel(7))
            self.assertEqual(list(compaction_main._iter_message_text("plain")), ["plain"])
            self.assertEqual(
                list(
                    compaction_main._iter_message_text(
                        [{"type": "input_text", "text": "a"}, {"type": "other", "text": "b"}, "ignore"]
                    )
                ),
                ["a"],
            )
            self.assertTrue(compaction_main._current_turn_contains_sentinel("<<<LOCAL_COMPACT>>> in string"))
            self.assertFalse(compaction_main._current_turn_contains_sentinel({"bad": "shape"}))
            self.assertTrue(
                compaction_main._current_turn_contains_sentinel(
                    [
                        "ignore",
                        {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "old"}]},
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<<<LOCAL_COMPACT>>> now"}]},
                    ]
                )
            )
            self.assertFalse(
                compaction_main._current_turn_contains_sentinel(
                    [
                        {"type": "event"},
                        {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "old"}]},
                    ]
                )
            )
            self.assertTrue(compaction_main._is_inline_compaction({"instructions": "<<<LOCAL_COMPACT>>> summarize"}))
            self.assertFalse(compaction_main._is_inline_compaction({"input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "continue"}]}]}))
            self.assertEqual(list(compaction_main._iter_message_text([{"type": "text", "text": ""}])), [])
            self.assertEqual(list(compaction_main._iter_message_text(7)), [])

            mock_settings.inline_compact_sentinel = ""
            self.assertFalse(compaction_main._contains_sentinel("<<<LOCAL_COMPACT>>>"))

    def test_proxy_helpers_cover_header_url_auth_and_response_modes(self) -> None:
        with patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.openai_passthrough_base_url = "https://example.test/base/"
            mock_settings.ollama_connect_timeout_seconds = 1
            mock_settings.codex_timeout_seconds = 2
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 0.2

            request = _DummyRequest(
                {
                    "host": "localhost",
                    "content-length": "10",
                    "connection": "keep-alive",
                    "authorization": "Bearer token",
                    "x-custom": "y",
                }
            )
            self.assertEqual(compaction_main._proxy_headers(request), {"authorization": "Bearer token", "x-custom": "y"})
            self.assertEqual(compaction_main._proxy_url("/responses"), "https://example.test/base/responses")
            self.assertEqual(compaction_main._auth_header_kind({"authorization": "Bearer token"}), "bearer")
            self.assertEqual(compaction_main._auth_header_kind({"Authorization": "Basic abc"}), "other")
            self.assertEqual(compaction_main._auth_header_kind({}), "none")

            non_stream = Mock()
            non_stream.status_code = 500
            non_stream.content = b'{"error":"bad"}'
            non_stream.headers = {"content-type": "application/json"}
            non_stream.close = Mock()

            with patch.object(compaction_main._UPSTREAM, "post", return_value=non_stream):
                response = compaction_main._proxy_response(request, "/responses", {"stream": False})
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.body, b'{"error":"bad"}')
            non_stream.close.assert_called_once()

            stream = Mock()
            stream.status_code = 200
            stream.headers = {"content-type": "text/event-stream"}
            stream.iter_content = Mock(return_value=iter([b"a", b"", b"b"]))
            stream.close = Mock()
            with patch.object(compaction_main._UPSTREAM, "post", return_value=stream):
                response = compaction_main._proxy_response(request, "/responses", {"stream": True})
                body = asyncio.run(_collect_streaming_response(response))
            self.assertEqual(body, b"ab")
            stream.close.assert_called_once()

    def test_spark_rewrite_classifier_covers_qualifying_shapes(self) -> None:
        file_read_payload = {
            "model": "gpt-5.4",
            "input": [
                {"type": "reasoning", "encrypted_content": "skip"},
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Command: /bin/bash -lc \"sed -n '1,220p' app/router.py\"\nOutput:\nfrom app import router\n",
                },
            ],
        }
        search_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                }
            ],
        }
        test_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_3",
                    "output": "Command: /bin/bash -lc '.venv/bin/python -m unittest tests.test_router_helpers'\nOutput:\nOK\n",
                }
            ],
        }
        polling_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_4",
                    "output": "Command: /bin/bash -lc 'tail -n 40 -f state/compaction_transport.jsonl'\nProcess running with session ID 123\nOutput:\n",
                }
            ],
        }
        self.assertEqual(compaction_main._qualifying_spark_category(file_read_payload), "file_read")
        self.assertEqual(compaction_main._qualifying_spark_category(search_payload), "search_inventory")
        self.assertEqual(compaction_main._qualifying_spark_category(test_payload), "targeted_test")
        self.assertEqual(compaction_main._qualifying_spark_category(polling_payload), "polling")

    def test_spark_rewrite_respects_rate_and_context_limit(self) -> None:
        payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                }
            ],
        }
        with patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_spark(payload)
        self.assertEqual(rewritten["model"], "gpt-5.3-codex-spark")
        self.assertTrue(rewrite["applied"])
        huge_payload = {
            "model": "gpt-5.4",
            "input": [
                {"type": "function_call_output", "call_id": "call_9", "output": "Command: /bin/bash -lc 'rg --files .'\nOutput:\n" + ("a" * 500_000)}
            ],
        }
        with patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_spark(huge_payload)
        self.assertEqual(rewritten["model"], "gpt-5.4")
        self.assertFalse(rewrite["applied"])

    def test_spark_rewrite_uses_tokenizer_based_request_estimate(self) -> None:
        payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_7",
                    "output": "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                }
            ],
        }
        with (
            patch.object(compaction_main, "settings", create=True) as mock_settings,
            patch.object(compaction_main, "estimate_openai_tokens", return_value=99_999) as estimate_openai_tokens,
        ):
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_spark(payload)
        estimate_openai_tokens.assert_called_once_with(payload, model="gpt-5.4")
        self.assertEqual(rewritten["model"], "gpt-5.3-codex-spark")
        self.assertEqual(rewrite["request_tokens"], 99_999)

    def test_direct_endpoint_helpers_cover_health_and_simple_wrappers(self) -> None:
        class DummyService:
            def invoke(self, req):
                return type("Resp", (), {"model_dump": lambda self: {"prompt": req.prompt}})()

            def compact_session(self, *args, **kwargs):
                return {"ok": True}

        with patch.object(compaction_main, "service", DummyService()), patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.router_model = "router"
            mock_settings.coder_model = "coder"
            mock_settings.reasoner_model = "reasoner"
            mock_settings.enable_codex_cli = True
            self.assertTrue(compaction_main.health()["ok"])
            self.assertEqual(compaction_main.ollama_version()["version"], "0.0.0-router")
            self.assertEqual(compaction_main.ollama_tags()["models"][0]["name"], "router")
            self.assertEqual(compaction_main.openai_models()["object"], "list")
            self.assertEqual(compaction_main.invoke(type("Req", (), {"prompt": "hello"})()).body, b'{"prompt":"hello"}')
            compact_response = compaction_main.compact(
                type(
                    "Compact",
                    (),
                    {
                        "session_id": "s1",
                        "items": [],
                        "current_request": "go",
                        "repo_context": {},
                        "refresh_if_needed": False,
                    },
                )()
            )
            self.assertEqual(compact_response.body, b'{"ok":true}')
