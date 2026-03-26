from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.compaction_transport import record_transport_event
from app.task_metrics import estimate_openai_tokens
from app.transport_metrics import (
    clear_transport_metrics_caches,
    metrics_state_path_for_log,
    transport_metrics_snapshot,
)


class TestTransportMetrics(unittest.TestCase):
    def setUp(self) -> None:
        clear_transport_metrics_caches()

    def tearDown(self) -> None:
        clear_transport_metrics_caches()

    def test_bootstraps_metrics_from_existing_transport_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transport.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event": "responses_passthrough",
                                "path": "/responses",
                                "status": 200,
                                "upstream_model": "gpt-5.3-codex-spark",
                                "request_tokens": 1000,
                                "spark_eligible": True,
                                "spark_rewrite_applied": True,
                                "spark_category": "file_read",
                                "mini_eligible": False,
                                "mini_rewrite_applied": False,
                            }
                        ),
                        json.dumps(
                            {
                                "event": "inline_compaction_detected",
                                "path": "/v1/responses",
                                "request_key": "inline_job_1",
                                "request_tokens": 400,
                            }
                        ),
                        json.dumps(
                            {
                                "event": "inline_compaction_job_started",
                                "path": "/v1/responses",
                                "request_key": "inline_job_1",
                                "local_model": "qwen3.5-9b:iq4_xs",
                            }
                        ),
                        json.dumps(
                            {
                                "event": "inline_compaction_spark_fallback",
                                "path": "/v1/responses",
                                "request_key": "inline_job_1",
                                "upstream_model": "gpt-5.3-codex-spark",
                            }
                        ),
                        json.dumps(
                            {
                                "event": "inline_compaction_completed",
                                "path": "/v1/responses",
                                "request_key": "inline_job_1",
                                "after_payload": {
                                    "model": "gpt-5.3-codex-spark",
                                    "raw_backend": {"mode": "spark_chunked_durable_compaction"},
                                },
                            }
                        ),
                        json.dumps({"event": "unsupported_transport", "path": "/v1/messages", "transport": "v1_messages"}),
                        json.dumps({"event": "internal_compact_start", "path": "/internal/compact"}),
                        json.dumps({"event": "internal_compact_completed", "path": "/internal/compact", "request_tokens": 250}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = transport_metrics_snapshot(log_path)

            self.assertTrue(snapshot["source"]["bootstrapped_from_log"])
            self.assertEqual(snapshot["source"]["bootstrapped_event_count"], 8)
            self.assertEqual(snapshot["events_total"], 8)
            self.assertEqual(snapshot["responses"]["spark"]["expected"], 1)
            self.assertEqual(snapshot["responses"]["spark"]["successful"], 1)
            self.assertEqual(snapshot["compaction"]["detected"], 1)
            self.assertEqual(snapshot["compaction"]["started_by_model"]["qwen3.5-9b:iq4_xs"], 1)
            self.assertEqual(snapshot["compaction"]["spark_fallbacks_by_model"]["gpt-5.3-codex-spark"], 1)
            self.assertEqual(snapshot["compaction"]["completed_by_model"]["gpt-5.3-codex-spark"], 1)
            self.assertEqual(snapshot["paths"]["/v1/responses"]["passthrough_total"], 1)
            self.assertEqual(snapshot["paths"]["/v1/responses"]["inline_completed"], 1)
            self.assertEqual(snapshot["unsupported"]["by_transport"]["v1_messages"], 1)
            self.assertEqual(snapshot["internal_compact"]["completed"], 1)
            self.assertEqual(snapshot["responses"]["estimated_token_savings"]["spark"], 700)
            self.assertEqual(snapshot["compaction"]["estimated_token_savings"]["spark"], 280)
            self.assertEqual(snapshot["internal_compact"]["estimated_token_savings"]["local"], 250)
            self.assertEqual(snapshot["estimated_token_savings"]["total"], 1230)
            self.assertTrue(metrics_state_path_for_log(log_path).exists())

    def test_record_transport_event_updates_metrics_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transport.jsonl"
            record_transport_event(
                log_path,
                "responses_passthrough",
                path="/responses",
                status=500,
                upstream_model="gpt-5.4-mini",
                request_tokens=330,
                spark_eligible=False,
                spark_rewrite_applied=False,
                mini_eligible=True,
                mini_rewrite_applied=True,
                mini_category="bounded_investigation",
            )

            snapshot = transport_metrics_snapshot(log_path)

            self.assertEqual(snapshot["events_total"], 1)
            self.assertEqual(snapshot["responses"]["failed"], 1)
            self.assertEqual(snapshot["responses"]["mini"]["eligible"], 1)
            self.assertEqual(snapshot["responses"]["mini"]["expected"], 1)
            self.assertEqual(snapshot["responses"]["mini"]["failed"], 1)
            self.assertEqual(snapshot["paths"]["/v1/responses"]["passthrough_failed"], 1)
            self.assertEqual(snapshot["responses"]["by_upstream_model"]["gpt-5.4-mini"]["failed"], 1)
            self.assertEqual(snapshot["responses"]["estimated_token_savings"]["mini"], 0)

    def test_bootstrap_rebuilds_from_log_when_metrics_state_version_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transport.jsonl"
            log_path.write_text(
                json.dumps(
                    {
                        "event": "responses_passthrough",
                        "path": "/responses",
                        "status": 200,
                        "upstream_model": "gpt-5.4-mini",
                        "request_tokens": 330,
                        "spark_eligible": False,
                        "spark_rewrite_applied": False,
                        "mini_eligible": True,
                        "mini_rewrite_applied": True,
                        "mini_category": "bounded_investigation",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metrics_path = metrics_state_path_for_log(log_path)
            metrics_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "events_total": 0,
                        "responses": {
                            "mini": {"expected": 0},
                            "estimated_token_savings": {"mini": 0},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = transport_metrics_snapshot(log_path)

            self.assertEqual(snapshot["version"], 3)
            self.assertEqual(snapshot["events_total"], 1)
            self.assertEqual(snapshot["responses"]["mini"]["expected"], 1)
            self.assertEqual(snapshot["responses"]["estimated_token_savings"]["mini"], 230)

    def test_bootstrap_recovers_historical_local_compaction_savings_from_detected_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transport.jsonl"
            before_payload = {
                "model": "gpt-5.4",
                "instructions": "<<<LOCAL_COMPACT>>> summarize",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "A" * 4000},
                        ],
                    }
                ],
            }
            expected_tokens = estimate_openai_tokens(before_payload, model="gpt-5.4")
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event": "inline_compaction_detected",
                                "path": "/v1/responses",
                                "request_key": "inline_job_1",
                                "before_payload": before_payload,
                            }
                        ),
                        json.dumps(
                            {
                                "event": "inline_compaction_completed",
                                "path": "/v1/responses",
                                "stream": True,
                                "after_payload": {
                                    "model": "qwen3.5-9b:iq4_xs",
                                    "raw_backend": {"created_at": "2026-03-19T00:00:00Z"},
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = transport_metrics_snapshot(log_path)

            self.assertEqual(snapshot["compaction"]["estimated_token_savings"]["local"], expected_tokens)
            self.assertEqual(snapshot["estimated_token_savings"]["local"], expected_tokens)

    def test_bootstrap_uses_default_local_compaction_savings_when_unpaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transport.jsonl"
            log_path.write_text(
                json.dumps(
                    {
                        "event": "inline_compaction_completed",
                        "path": "/v1/responses",
                        "stream": True,
                        "after_payload": {
                            "model": "qwen3.5-9b:iq4_xs",
                            "raw_backend": {"created_at": "2026-03-19T00:00:00Z"},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = transport_metrics_snapshot(log_path)

            self.assertEqual(snapshot["compaction"]["estimated_token_savings"]["local"], 250000)
            self.assertEqual(snapshot["estimated_token_savings"]["local"], 250000)
