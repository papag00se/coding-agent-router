from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app import compaction_main
from app.transport_metrics import clear_transport_metrics_caches


class _DummyRequest:
    def __init__(self, headers):
        self.headers = headers


async def _collect_streaming_response(response) -> bytes:
    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    return body


class TestCompactionMainHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._transport_log_path = Path(self._tmpdir.name) / "transport.jsonl"
        self._spark_quota_path = Path(self._tmpdir.name) / "spark_quota.json"
        self._transport_log_patcher = patch.object(compaction_main, "_TRANSPORT_LOG_PATH", self._transport_log_path)
        self._transport_log_patcher.start()
        self._spark_quota_state = compaction_main.SparkQuotaState(self._spark_quota_path)
        self._spark_quota_patcher = patch.object(compaction_main, "_SPARK_QUOTA_STATE", self._spark_quota_state)
        self._spark_quota_patcher.start()
        clear_transport_metrics_caches()

    def tearDown(self) -> None:
        clear_transport_metrics_caches()
        self._spark_quota_patcher.stop()
        self._transport_log_patcher.stop()
        self._tmpdir.cleanup()

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

    def test_spark_rewrite_classifier_covers_extended_bounded_categories(self) -> None:
        test_fix_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_test",
                    "output": "Command: /bin/bash -lc '.venv/bin/python -m pytest tests/test_router.py'\nFAILED tests/test_router.py::test_handles_error\nTraceback (most recent call last):\n  File \"tests/test_router.py\", line 12, in test_handles_error\nAssertionError\n",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_read_after_test",
                    "output": "Command: /bin/bash -lc \"sed -n '1,220p' app/router.py\"\nOutput:\nfrom app import router\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix tests/test_router.py."}],
                },
            ],
        }
        diff_followup_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_diff",
                    "output": "Command: /bin/bash -lc 'git diff -- app/router.py tests/test_router.py'\ndiff --git a/app/router.py b/app/router.py\n--- a/app/router.py\n+++ b/app/router.py\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Tweak the review comment cleanup in app/router.py and tests/test_router.py."}],
                },
            ],
        }
        stacktrace_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_trace",
                    "output": "Command: /bin/bash -lc '.venv/bin/python -m pytest tests/test_router.py'\nTraceback (most recent call last):\n  File \"/repo/app/router.py\", line 22, in route\nValueError: bad route\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Debug the failing stacktrace in /repo/app/router.py."}],
                },
            ],
        }
        localized_edit_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_read",
                    "output": "Command: /bin/bash -lc \"sed -n '1,220p' app/router.py\"\nOutput:\nfrom app import router\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Update app/router.py to adjust route selection."}],
                },
            ],
        }
        small_refactor_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_rg",
                    "output": "Command: /bin/bash -lc 'rg -n \"functionName\" app/router.py tests/test_router.py'\napp/router.py:10:def functionName():\ntests/test_router.py:8:functionName()\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Rename functionName in app/router.py and tests/test_router.py."}],
                },
            ],
        }
        synthesis_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_findings",
                    "output": "Command: /bin/bash -lc 'rg -n \"spark\" app/router.py docs/spec/routing.md'\napp/router.py:20:spark\ndocs/spec/routing.md:5:spark\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Summarize the findings in app/router.py and docs/spec/routing.md with next steps."}],
                },
            ],
        }
        self.assertEqual(compaction_main._qualifying_spark_category(test_fix_payload), "test_fix_loop")
        self.assertEqual(compaction_main._qualifying_spark_category(diff_followup_payload), "diff_followup")
        self.assertEqual(compaction_main._qualifying_spark_category(stacktrace_payload), "stacktrace_triage")
        self.assertEqual(compaction_main._qualifying_spark_category(localized_edit_payload), "localized_edit")
        self.assertEqual(compaction_main._qualifying_spark_category(small_refactor_payload), "small_refactor")
        self.assertEqual(compaction_main._qualifying_spark_category(synthesis_payload), "simple_synthesis")

    def test_spark_quota_state_blocks_until_reset_and_then_reopens(self) -> None:
        response = Mock()
        response.status_code = 429
        response.headers = {"Retry-After": "120"}
        response.content = json.dumps({"error": {"message": "rate limited"}}).encode("utf-8")

        snapshot = self._spark_quota_state.mark_exhausted(response, model="gpt-5.3-codex-spark", now=100.0)

        self.assertTrue(snapshot["blocked"])
        self.assertEqual(snapshot["last_error_status"], 429)
        self.assertEqual(snapshot["last_error_model"], "gpt-5.3-codex-spark")
        self.assertEqual(snapshot["last_error_reason"], "retry-after")
        self.assertEqual(snapshot["blocked_until"], 220.0)
        self.assertTrue(self._spark_quota_state.is_blocked(now=150.0))
        self.assertFalse(self._spark_quota_state.is_blocked(now=221.0))
        self.assertFalse(self._spark_quota_state.snapshot(now=221.0)["blocked"])

    def test_spark_quota_state_blocks_without_http_response(self) -> None:
        snapshot = self._spark_quota_state.mark_exhausted(
            None,
            model="gpt-5.3-codex-spark",
            now=100.0,
            error="spark down",
        )

        self.assertTrue(snapshot["blocked"])
        self.assertIsNone(snapshot["last_error_status"])
        self.assertEqual(snapshot["last_error_reason"], "spark down")
        self.assertEqual(snapshot["last_error_preview"], "spark down")

    def test_mini_rewrite_classifier_covers_bounded_non_spark_shapes(self) -> None:
        investigation_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_search",
                    "output": "Command: /bin/bash -lc 'rg -n \"RouterDecision\" app/router.py tests/test_router.py docs/spec/routing.md app/models.py'\napp/router.py:154:class RoutingService\ntests/test_router.py:18:RouteDecision\ndocs/spec/routing.md:47:return JSON containing route, confidence, and reason.\napp/models.py:27:class RouteDecision\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Investigate the likely root cause across app/router.py, tests/test_router.py, docs/spec/routing.md, and app/models.py."}],
                },
            ],
        }
        analysis_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_diff",
                    "output": "Command: /bin/bash -lc 'git diff -- app/router.py tests/test_router.py docs/spec/routing.md app/models.py'\ndiff --git a/app/router.py b/app/router.py\n--- a/app/router.py\n+++ b/app/router.py\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Summarize the changes and recommend next steps across app/router.py, tests/test_router.py, docs/spec/routing.md, and app/models.py."}],
                },
            ],
        }
        self.assertEqual(compaction_main._qualifying_mini_category(investigation_payload), "bounded_investigation")
        self.assertEqual(compaction_main._qualifying_mini_category(analysis_payload), "cross_file_analysis")

    def test_passthrough_model_rewrite_respects_spark_rate_and_sends_oversize_to_mini(self) -> None:
        spark_payload = {
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
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_model(spark_payload)
        self.assertEqual(rewritten["model"], "gpt-5.3-codex-spark")
        self.assertTrue(rewrite["applied"])
        self.assertEqual(rewrite["rewrite_family"], "spark")
        self.assertTrue(rewrite["spark"]["applied"])
        investigation_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_search",
                    "output": "Command: /bin/bash -lc 'rg -n \"RouteDecision\" app/router.py tests/test_router.py docs/spec/routing.md app/models.py'\napp/router.py:154:class RoutingService\ntests/test_router.py:18:RouteDecision\ndocs/spec/routing.md:47:return JSON containing route, confidence, and reason.\napp/models.py:27:class RouteDecision\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Investigate the likely root cause across app/router.py, tests/test_router.py, docs/spec/routing.md, and app/models.py."}],
                },
            ],
        }
        with patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_model(investigation_payload)
        self.assertEqual(rewritten["model"], "gpt-5.4-mini")
        self.assertEqual(rewrite["rewrite_family"], "mini")
        self.assertTrue(rewrite["spark"]["eligible"])
        self.assertFalse(rewrite["spark"]["applied"])
        huge_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_9",
                    "output": "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                }
            ],
        }
        with (
            patch.object(compaction_main, "settings", create=True) as mock_settings,
            patch.object(compaction_main, "estimate_openai_tokens", return_value=114_689),
        ):
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_model(huge_payload)
        self.assertEqual(rewritten["model"], "gpt-5.4-mini")
        self.assertTrue(rewrite["applied"])
        self.assertEqual(rewrite["rewrite_family"], "mini")
        self.assertEqual(rewrite["mini"]["category"], "oversize_for_spark")

    def test_passthrough_model_rewrite_normalizes_gpt54_only_fields_for_spark(self) -> None:
        payload = {
            "model": "gpt-5.4",
            "reasoning": {"effort": "none", "summary": "auto"},
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Reading app/router.py"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Command: /bin/bash -lc \"sed -n '1,220p' app/router.py\"\nOutput:\nfrom app import router\n",
                },
            ],
        }
        with patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_model(payload)
        self.assertEqual(rewritten["model"], "gpt-5.3-codex-spark")
        self.assertEqual(rewritten["reasoning"]["effort"], "low")
        self.assertEqual(rewritten["reasoning"]["summary"], "auto")
        self.assertNotIn("phase", rewritten["input"][0])
        self.assertEqual(payload["reasoning"]["effort"], "none")
        self.assertEqual(payload["input"][0]["phase"], "commentary")
        self.assertEqual(rewrite["rewrite_family"], "spark")

    def test_passthrough_model_rewrite_blocks_spark_for_image_payloads_but_allows_mini(self) -> None:
        spark_image_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_spark_image",
                    "output": "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Read this screenshot and tell me what it shows."},
                        {"type": "input_image", "image_url": "https://example.test/screenshot.png"},
                    ],
                },
            ],
        }
        mini_image_payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_mini_image",
                    "output": "Command: /bin/bash -lc 'rg -n \"RouterDecision\" app/router.py tests/test_router.py docs/spec/routing.md app/models.py'\napp/router.py:154:class RoutingService\ntests/test_router.py:18:RouteDecision\ndocs/spec/routing.md:47:return JSON containing route, confidence, and reason.\napp/models.py:27:class RouteDecision\n",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Investigate the likely root cause across app/router.py, tests/test_router.py, docs/spec/routing.md, and app/models.py."},
                        {"type": "input_image", "image_url": "https://example.test/chart.png"},
                    ],
                },
            ],
        }
        with patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            spark_rewritten, spark_rewrite = compaction_main._rewrite_passthrough_payload_for_model(spark_image_payload)
            mini_rewritten, mini_rewrite = compaction_main._rewrite_passthrough_payload_for_model(mini_image_payload)
        self.assertIs(spark_rewritten, spark_image_payload)
        self.assertEqual(spark_rewritten["model"], "gpt-5.4")
        self.assertFalse(spark_rewrite["applied"])
        self.assertTrue(spark_rewrite["image_inputs_present"])
        self.assertFalse(spark_rewrite["spark"]["eligible"])
        self.assertEqual(mini_rewritten["model"], "gpt-5.4-mini")
        self.assertTrue(mini_rewrite["applied"])
        self.assertEqual(mini_rewrite["rewrite_family"], "mini")
        self.assertTrue(mini_rewrite["image_inputs_present"])

    def test_passthrough_model_rewrite_rescues_direct_spark_image_requests_to_mini(self) -> None:
        payload = {
            "model": "gpt-5.3-codex-spark",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Read this screenshot and explain it."},
                        {"type": "input_image", "image_url": "https://example.test/screenshot.png"},
                    ],
                }
            ],
        }
        with patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_model(payload)
        self.assertEqual(rewritten["model"], "gpt-5.4-mini")
        self.assertEqual(rewrite["rewrite_family"], "mini")
        self.assertTrue(rewrite["mini"]["image_spark_fallback_applied"])
        self.assertTrue(rewrite["image_inputs_present"])

    def test_passthrough_model_rewrite_uses_tokenizer_based_request_estimate(self) -> None:
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
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_model(payload)
        estimate_openai_tokens.assert_called_once_with(payload, model="gpt-5.4")
        self.assertEqual(rewritten["model"], "gpt-5.3-codex-spark")
        self.assertEqual(rewrite["request_tokens"], 99_999)

    def test_passthrough_model_rewrite_falls_back_to_mini_when_spark_is_blocked(self) -> None:
        response = Mock()
        response.status_code = 429
        response.headers = {"Retry-After": "3600"}
        response.content = b'{"error":{"message":"rate limited"}}'
        self._spark_quota_state.mark_exhausted(response, model="gpt-5.3-codex-spark", now=time.time())

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
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_model(payload)
        self.assertEqual(rewritten["model"], "gpt-5.4-mini")
        self.assertTrue(rewrite["spark_targeted"])
        self.assertTrue(rewrite["spark_quota_blocked"])
        self.assertFalse(rewrite["spark"]["applied"])
        self.assertTrue(rewrite["mini"]["applied"])
        self.assertEqual(rewrite["mini"]["category"], "spark_quota_fallback")

    def test_passthrough_model_rewrite_allows_requests_at_spark_cap(self) -> None:
        payload = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_8",
                    "output": "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                }
            ],
        }
        with (
            patch.object(compaction_main, "settings", create=True) as mock_settings,
            patch.object(compaction_main, "estimate_openai_tokens", return_value=114_688),
        ):
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_model(payload)
        self.assertEqual(rewritten["model"], "gpt-5.3-codex-spark")
        self.assertTrue(rewrite["spark"]["eligible"])
        self.assertTrue(rewrite["spark"]["applied"])

    def test_passthrough_model_rewrite_does_not_rewrite_missing_model(self) -> None:
        payload = {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_8",
                    "output": "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                }
            ],
        }
        with patch.object(compaction_main, "settings", create=True) as mock_settings:
            mock_settings.codex_spark_model = "gpt-5.3-codex-spark"
            mock_settings.codex_spark_qualified_rate = 1.0
            mock_settings.codex_mini_model = "gpt-5.4-mini"
            rewritten, rewrite = compaction_main._rewrite_passthrough_payload_for_model(payload)
        self.assertEqual(rewritten, payload)
        self.assertFalse(rewrite["applied"])

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
