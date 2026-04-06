from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from fastapi.testclient import TestClient

from app import compaction_main
from app.compaction.models import CodexHandoffFlow, SessionHandoff
from app.transport_metrics import clear_transport_metrics_caches


class DummyCompactionService:
    def __init__(self) -> None:
        self.compact_calls = []

    def compact_session(self, *args, **kwargs):
        self.compact_calls.append((args, kwargs))
        return {'ok': True}


class DummyInlineService(DummyCompactionService):
    def __init__(self) -> None:
        super().__init__()
        self.inline_requests = []

    def invoke_inline_compact_from_anthropic(self, req, *, progress_callback=None):
        self.inline_requests.append(req.model_dump())
        return {
            'id': 'msg_router_local',
            'type': 'message',
            'role': 'assistant',
            'model': 'qwen3.5-9b:iq4_xs',
            'content': [{'type': 'text', 'text': 'compact summary'}],
            'usage': {'input_tokens': 5, 'output_tokens': 3},
            'raw_backend': {'created_at': '2026-03-19T00:00:00Z'},
        }

    def stream_inline_compact_from_anthropic(self, req, *, progress_callback=None):
        self.inline_requests.append(req.model_dump())
        if progress_callback is not None:
            progress_callback('normalize_completed', compactable_count=1, preserved_tail_count=0)
        yield {'type': 'text_delta', 'delta': 'compact '}
        yield {'type': 'text_delta', 'delta': 'summary'}
        yield {
            'type': 'final',
            'response': {
                'id': 'msg_router_local',
                'type': 'message',
                'role': 'assistant',
                'model': 'qwen3.5-9b:iq4_xs',
                'content': [{'type': 'text', 'text': 'compact summary'}],
                'usage': {'input_tokens': 5, 'output_tokens': 3},
                'raw_backend': {'created_at': '2026-03-19T00:00:00Z'},
            },
        }


class TestCompactionMain(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._transport_log_path = Path(self._tmpdir.name) / 'transport.jsonl'
        self._spark_quota_path = Path(self._tmpdir.name) / 'spark_quota.json'
        self._transport_log_patcher = patch.object(compaction_main, '_TRANSPORT_LOG_PATH', self._transport_log_path)
        self._transport_log_patcher.start()
        self._spark_quota_state = compaction_main.SparkQuotaState(self._spark_quota_path)
        self._spark_quota_patcher = patch.object(compaction_main, '_SPARK_QUOTA_STATE', self._spark_quota_state)
        self._spark_quota_patcher.start()
        clear_transport_metrics_caches()

    def tearDown(self) -> None:
        clear_transport_metrics_caches()
        self._spark_quota_patcher.stop()
        self._transport_log_patcher.stop()
        self._tmpdir.cleanup()

    def test_responses_detects_inline_compaction_and_uses_local_service(self):
        service = DummyInlineService()
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main, 'inline_compaction_jobs', compaction_main.InlineCompactionJobManager(service)),
        ):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['output_text'], 'compact summary')
        self.assertEqual(service.inline_requests[0]['system'], '<<<LOCAL_COMPACT>>> Summarize the thread.')

    def test_responses_strip_codex_bootstrap_instructions_before_inline_compaction(self):
        service = DummyInlineService()
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main, 'inline_compaction_jobs', compaction_main.InlineCompactionJobManager(service)),
        ):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'instructions': 'You are Codex, a coding agent based on GPT-5.',
                    'input': [
                        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': '<<<LOCAL_COMPACT>>> summarize'}]},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(service.inline_requests[0]['system'])

    def test_responses_can_log_inline_compaction_payloads(self):
        service = DummyInlineService()
        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(compaction_main, 'service', service),
                patch.object(compaction_main, 'inline_compaction_jobs', compaction_main.InlineCompactionJobManager(service)),
                patch.object(compaction_main, '_TRANSPORT_LOG_PATH', Path(tmpdir) / 'transport.jsonl'),
                patch('app.compaction_transport.settings', SimpleNamespace(log_compaction_payloads=True)),
            ):
                client = TestClient(compaction_main.app)
                response = client.post(
                    '/v1/responses',
                    json={
                        'model': 'gpt-5.4',
                        'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
                        'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
                    },
                )
                events = [json.loads(line) for line in (Path(tmpdir) / 'transport.jsonl').read_text(encoding='utf-8').splitlines()]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[0]['event'], 'inline_compaction_detected')
        self.assertEqual(events[0]['before_payload']['model'], 'gpt-5.4')
        self.assertEqual(events[1]['event'], 'inline_compaction_job_started')
        self.assertEqual(events[-1]['event'], 'inline_compaction_completed')
        self.assertEqual(events[-1]['after_payload']['content'][0]['text'], 'compact summary')

    def test_responses_passthroughs_non_compaction_requests(self):
        service = DummyInlineService()
        upstream = Mock()
        upstream.status_code = 200
        upstream.content = json.dumps({'object': 'response', 'output_text': 'upstream'}).encode('utf-8')
        upstream.headers = {'content-type': 'application/json'}
        upstream.close = Mock()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_qualified_rate = 0.0
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main._UPSTREAM, 'post', return_value=upstream) as post,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Check the repo'}]}],
                },
                headers={'Authorization': 'Bearer test-token'},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['output_text'], 'upstream')
        self.assertEqual(service.inline_requests, [])
        self.assertEqual(post.call_args.args[0], 'https://chatgpt.com/backend-api/codex/responses')
        self.assertEqual(post.call_args.kwargs['headers']['authorization'], 'Bearer test-token')
        self.assertEqual(post.call_args.kwargs['json']['model'], 'gpt-5.4')

    def test_responses_passthrough_rewrites_qualifying_requests_to_spark(self):
        service = DummyInlineService()
        upstream = Mock()
        upstream.status_code = 200
        upstream.content = json.dumps({'object': 'response', 'output_text': 'upstream'}).encode('utf-8')
        upstream.headers = {'content-type': 'application/json'}
        upstream.close = Mock()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        settings.codex_spark_qualified_rate = 1.0
        payload = {
            'model': 'gpt-5.4',
            'input': [
                {
                    'type': 'function_call_output',
                    'call_id': 'call_1',
                    'output': "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                }
            ],
        }
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main._UPSTREAM, 'post', return_value=upstream) as post,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post('/v1/responses', json=payload, headers={'Authorization': 'Bearer test-token'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_args.kwargs['json']['model'], 'gpt-5.3-codex-spark')

    def test_responses_passthrough_retries_spark_quota_failures_on_mini(self):
        service = DummyInlineService()
        spark_upstream = Mock()
        spark_upstream.status_code = 429
        spark_upstream.content = json.dumps({'error': {'message': 'rate limited'}}).encode('utf-8')
        spark_upstream.headers = {'content-type': 'application/json', 'Retry-After': '120'}
        spark_upstream.close = Mock()
        mini_upstream = Mock()
        mini_upstream.status_code = 200
        mini_upstream.content = json.dumps({'object': 'response', 'output_text': 'mini fallback'}).encode('utf-8')
        mini_upstream.headers = {'content-type': 'application/json'}
        mini_upstream.close = Mock()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        settings.codex_spark_qualified_rate = 1.0
        settings.codex_mini_model = 'gpt-5.4-mini'
        payload = {
            'model': 'gpt-5.4',
            'input': [
                {
                    'type': 'function_call_output',
                    'call_id': 'call_1',
                    'output': "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                }
            ],
        }
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main._UPSTREAM, 'post', side_effect=[spark_upstream, mini_upstream]) as post,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post('/v1/responses', json=payload, headers={'Authorization': 'Bearer test-token'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['output_text'], 'mini fallback')
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs['json']['model'], 'gpt-5.3-codex-spark')
        self.assertEqual(post.call_args_list[1].kwargs['json']['model'], 'gpt-5.4-mini')
        metrics = compaction_main.transport_metrics_snapshot(self._transport_log_path)
        self.assertEqual(metrics['responses']['spark']['blocked_by_quota'], 1)
        self.assertEqual(metrics['responses']['mini']['expected'], 1)
        self.assertEqual(metrics['responses']['mini']['successful'], 1)

    def test_responses_passthrough_retries_spark_service_unavailable_failures_on_mini(self):
        service = DummyInlineService()
        spark_upstream = Mock()
        spark_upstream.status_code = 503
        spark_upstream.content = json.dumps({'error': {'message': 'service unavailable'}}).encode('utf-8')
        spark_upstream.headers = {'content-type': 'application/json'}
        spark_upstream.close = Mock()
        mini_upstream = Mock()
        mini_upstream.status_code = 200
        mini_upstream.content = json.dumps({'object': 'response', 'output_text': 'mini fallback'}).encode('utf-8')
        mini_upstream.headers = {'content-type': 'application/json'}
        mini_upstream.close = Mock()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        settings.codex_spark_qualified_rate = 1.0
        settings.codex_mini_model = 'gpt-5.4-mini'
        payload = {
            'model': 'gpt-5.4',
            'input': [
                {
                    'type': 'function_call_output',
                    'call_id': 'call_1',
                    'output': "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                }
            ],
        }
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main._UPSTREAM, 'post', side_effect=[spark_upstream, mini_upstream]) as post,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post('/v1/responses', json=payload, headers={'Authorization': 'Bearer test-token'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['output_text'], 'mini fallback')
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs['json']['model'], 'gpt-5.3-codex-spark')
        self.assertEqual(post.call_args_list[1].kwargs['json']['model'], 'gpt-5.4-mini')
        snapshot = self._spark_quota_state.snapshot()
        self.assertTrue(snapshot['blocked'])
        self.assertEqual(snapshot['last_error_status'], 503)

    def test_internal_metrics_reports_passthrough_and_inline_counts(self):
        service = DummyInlineService()
        upstream = Mock()
        upstream.status_code = 200
        upstream.content = json.dumps({'object': 'response', 'output_text': 'upstream'}).encode('utf-8')
        upstream.headers = {'content-type': 'application/json'}
        upstream.close = Mock()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        settings.codex_spark_qualified_rate = 1.0
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'transport.jsonl'
            with (
                patch.object(compaction_main, 'service', service),
                patch.object(compaction_main, 'inline_compaction_jobs', compaction_main.InlineCompactionJobManager(service)),
                patch.object(compaction_main, '_TRANSPORT_LOG_PATH', log_path),
                patch.object(compaction_main._UPSTREAM, 'post', return_value=upstream),
                patch.object(compaction_main, 'settings', settings),
            ):
                client = TestClient(compaction_main.app)
                passthrough = client.post(
                    '/v1/responses',
                    json={
                        'model': 'gpt-5.4',
                        'input': [
                            {
                                'type': 'function_call_output',
                                'call_id': 'call_1',
                                'output': "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                            }
                        ],
                    },
                    headers={'Authorization': 'Bearer test-token'},
                )
                inline = client.post(
                    '/v1/responses',
                    json={
                        'model': 'gpt-5.4',
                        'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
                        'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
                    },
                )
                metrics = client.get('/internal/metrics')

        self.assertEqual(passthrough.status_code, 200)
        self.assertEqual(inline.status_code, 200)
        self.assertEqual(metrics.status_code, 200)
        body = metrics.json()
        self.assertEqual(body['responses']['spark']['expected'], 1)
        self.assertEqual(body['responses']['spark']['successful'], 1)
        self.assertGreater(body['responses']['estimated_token_savings']['spark'], 0)
        self.assertEqual(body['compaction']['detected'], 1)
        self.assertEqual(body['compaction']['completed'], 1)
        self.assertEqual(body['compaction']['completed_by_model']['qwen3.5-9b:iq4_xs'], 1)
        self.assertGreater(body['compaction']['estimated_token_savings']['local'], 0)
        self.assertEqual(body['paths']['/v1/responses']['passthrough_total'], 1)
        self.assertEqual(body['paths']['/v1/responses']['inline_detected'], 1)
        self.assertEqual(body['paths']['/v1/responses']['inline_completed'], 1)

    def test_responses_passthrough_rewrites_bounded_investigation_requests_to_mini(self):
        service = DummyInlineService()
        upstream = Mock()
        upstream.status_code = 200
        upstream.content = json.dumps({'object': 'response', 'output_text': 'upstream'}).encode('utf-8')
        upstream.headers = {'content-type': 'application/json'}
        upstream.close = Mock()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        settings.codex_spark_qualified_rate = 1.0
        settings.codex_mini_model = 'gpt-5.4-mini'
        payload = {
            'model': 'gpt-5.4',
            'input': [
                {
                    'type': 'function_call_output',
                    'call_id': 'call_search',
                    'output': "Command: /bin/bash -lc 'rg -n \"RouteDecision\" app/router.py tests/test_router.py docs/spec/routing.md app/models.py'\napp/router.py:154:class RoutingService\ntests/test_router.py:18:RouteDecision\ndocs/spec/routing.md:47:return JSON containing route, confidence, and reason.\napp/models.py:27:class RouteDecision\n",
                },
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'Investigate the likely root cause across app/router.py, tests/test_router.py, docs/spec/routing.md, and app/models.py.'}],
                },
            ],
        }
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main._UPSTREAM, 'post', return_value=upstream) as post,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post('/v1/responses', json=payload, headers={'Authorization': 'Bearer test-token'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_args.kwargs['json']['model'], 'gpt-5.4-mini')

    def test_responses_passthrough_keeps_image_requests_on_original_model(self):
        service = DummyInlineService()
        upstream = Mock()
        upstream.status_code = 200
        upstream.content = json.dumps({'object': 'response', 'output_text': 'upstream'}).encode('utf-8')
        upstream.headers = {'content-type': 'application/json'}
        upstream.close = Mock()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        settings.codex_spark_qualified_rate = 1.0
        settings.codex_mini_model = 'gpt-5.4-mini'
        payload = {
            'model': 'gpt-5.4',
            'input': [
                {
                    'type': 'function_call_output',
                    'call_id': 'call_1',
                    'output': "Command: /bin/bash -lc 'rg --files .'\nOutput:\n./app/router.py\n",
                },
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': 'Read this screenshot and tell me what it shows.'},
                        {'type': 'input_image', 'image_url': 'https://example.test/screenshot.png'},
                    ],
                },
            ],
        }
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main._UPSTREAM, 'post', return_value=upstream) as post,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post('/v1/responses', json=payload, headers={'Authorization': 'Bearer test-token'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['output_text'], 'upstream')
        self.assertEqual(service.inline_requests, [])
        self.assertEqual(post.call_args.args[0], 'https://chatgpt.com/backend-api/codex/responses')
        self.assertEqual(post.call_args.kwargs['json']['model'], 'gpt-5.4')
        self.assertEqual(post.call_args.kwargs['json']['input'][1]['content'][1]['type'], 'input_image')

    def test_responses_passthrough_rewrites_direct_spark_image_requests_to_mini(self):
        service = DummyInlineService()
        upstream = Mock()
        upstream.status_code = 200
        upstream.content = json.dumps({'object': 'response', 'output_text': 'upstream'}).encode('utf-8')
        upstream.headers = {'content-type': 'application/json'}
        upstream.close = Mock()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        settings.codex_spark_qualified_rate = 1.0
        settings.codex_mini_model = 'gpt-5.4-mini'
        payload = {
            'model': 'gpt-5.3-codex-spark',
            'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': 'Read this screenshot and explain it.'},
                        {'type': 'input_image', 'image_url': 'https://example.test/screenshot.png'},
                    ],
                },
            ],
        }
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main._UPSTREAM, 'post', return_value=upstream) as post,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post('/v1/responses', json=payload, headers={'Authorization': 'Bearer test-token'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['output_text'], 'upstream')
        self.assertEqual(post.call_args.kwargs['json']['model'], 'gpt-5.4-mini')
        self.assertEqual(post.call_args.kwargs['json']['input'][0]['content'][1]['type'], 'input_image')

    def test_responses_stream_detects_inline_compaction_and_uses_local_service(self):
        service = DummyInlineService()
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main, 'inline_compaction_jobs', compaction_main.InlineCompactionJobManager(service)),
        ):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
                    'stream': True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('event: response.in_progress', response.text)
        self.assertIn('event: response.output_text.delta', response.text)
        self.assertIn('compact summary', response.text)

    def test_responses_inline_compaction_falls_back_to_spark_on_local_failure(self):
        class FailingInlineService(DummyCompactionService):
            def stream_inline_compact_from_anthropic(self, req, *, progress_callback=None):
                raise ValueError('bad handoff schema')

        class SparkChunkedFallbackService:
            def __init__(self) -> None:
                self.compact_calls = []

            def compact_transcript(self, session_id, items, *, current_request, repo_context=None, progress_callback=None):
                self.compact_calls.append(
                    {
                        'session_id': session_id,
                        'items': items,
                        'current_request': current_request,
                        'repo_context': repo_context,
                    }
                )
                return SessionHandoff(
                    stable_task_definition='spark compacted',
                    current_request=current_request,
                    recent_raw_turns=items[-1:],
                )

            def build_codex_handoff_flow(self, session_id, *, current_request=None):
                return CodexHandoffFlow(
                    durable_memory=[{'name': 'TASK_STATE.md', 'content': '# Task State\n- spark compacted\n'}],
                    structured_handoff={'stable_task_definition': 'spark compacted'},
                    recent_raw_turns=[{'role': 'user', 'content': 'Current thread'}],
                    current_request=current_request or '',
                )

        service = FailingInlineService()
        spark_service = SparkChunkedFallbackService()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(
                compaction_main,
                'inline_compaction_jobs',
                compaction_main.InlineCompactionJobManager(
                    service,
                    fallback_callback=compaction_main._stream_spark_inline_compaction_fallback,
                ),
            ),
            patch.object(compaction_main, '_build_spark_chunked_compaction_service', return_value=spark_service) as build_fallback,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
                },
                headers={'Authorization': 'Bearer test-token'},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('spark compacted', response.json()['output_text'])
        self.assertEqual(build_fallback.call_args.args[0]['authorization'], 'Bearer test-token')
        self.assertEqual(spark_service.compact_calls[0]['current_request'], 'Current thread')

    def test_responses_inline_compaction_falls_back_to_mini_when_spark_is_unavailable(self):
        class FailingInlineService(DummyCompactionService):
            def stream_inline_compact_from_anthropic(self, req, *, progress_callback=None):
                if progress_callback is not None:
                    progress_callback('normalize_completed', compactable_count=1, preserved_tail_count=0)
                raise ValueError('bad handoff schema')

        class SparkDownService:
            def compact_transcript(self, session_id, items, *, current_request, repo_context=None, progress_callback=None):
                raise requests.ConnectionError('spark down')

        class MiniChunkedService:
            def compact_transcript(self, session_id, items, *, current_request, repo_context=None, progress_callback=None):
                if progress_callback is not None:
                    progress_callback('render_completed', session_id=session_id)
                return SessionHandoff(
                    stable_task_definition='mini compacted',
                    current_request=current_request,
                    recent_raw_turns=items[-1:],
                )

            def build_codex_handoff_flow(self, session_id, *, current_request=None):
                return CodexHandoffFlow(
                    durable_memory=[{'name': 'TASK_STATE.md', 'content': '# Task State\n- mini compacted\n'}],
                    structured_handoff={'stable_task_definition': 'mini compacted'},
                    recent_raw_turns=[{'role': 'user', 'content': 'Current thread'}],
                    current_request=current_request or '',
                )

        service = FailingInlineService()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        settings.codex_mini_model = 'gpt-5.4-mini'
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(
                compaction_main,
                'inline_compaction_jobs',
                compaction_main.InlineCompactionJobManager(
                    service,
                    fallback_callback=compaction_main._stream_spark_inline_compaction_fallback,
                ),
            ),
            patch.object(compaction_main, '_build_spark_chunked_compaction_service', return_value=SparkDownService()) as build_spark,
            patch.object(compaction_main, '_build_mini_chunked_compaction_service', return_value=MiniChunkedService()) as build_mini,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
                },
                headers={'Authorization': 'Bearer test-token'},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('mini compacted', response.json()['output_text'])
        self.assertEqual(build_spark.call_count, 1)
        self.assertEqual(build_mini.call_count, 1)
        snapshot = self._spark_quota_state.snapshot()
        self.assertTrue(snapshot['blocked'])
        self.assertEqual(snapshot['last_error_reason'], 'spark down')

    def test_responses_stream_inline_compaction_falls_back_to_spark_on_local_failure(self):
        class FailingInlineService(DummyCompactionService):
            def stream_inline_compact_from_anthropic(self, req, *, progress_callback=None):
                if progress_callback is not None:
                    progress_callback('normalize_completed', compactable_count=1, preserved_tail_count=0)
                raise ValueError('bad handoff schema')

        class SparkChunkedFallbackService:
            def __init__(self) -> None:
                self.compact_calls = []

            def compact_transcript(self, session_id, items, *, current_request, repo_context=None, progress_callback=None):
                self.compact_calls.append(
                    {
                        'session_id': session_id,
                        'items': items,
                        'current_request': current_request,
                        'repo_context': repo_context,
                    }
                )
                if progress_callback is not None:
                    progress_callback('render_completed', session_id=session_id)
                return SessionHandoff(
                    stable_task_definition='spark compacted',
                    current_request=current_request,
                    recent_raw_turns=items[-1:],
                )

            def build_codex_handoff_flow(self, session_id, *, current_request=None):
                return CodexHandoffFlow(
                    durable_memory=[{'name': 'TASK_STATE.md', 'content': '# Task State\n- spark compacted\n'}],
                    structured_handoff={'stable_task_definition': 'spark compacted'},
                    recent_raw_turns=[{'role': 'user', 'content': 'Current thread'}],
                    current_request=current_request or '',
                )

        service = FailingInlineService()
        spark_service = SparkChunkedFallbackService()
        settings = SimpleNamespace(**compaction_main.settings.__dict__)
        settings.codex_spark_model = 'gpt-5.3-codex-spark'
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(
                compaction_main,
                'inline_compaction_jobs',
                compaction_main.InlineCompactionJobManager(
                    service,
                    fallback_callback=compaction_main._stream_spark_inline_compaction_fallback,
                ),
            ),
            patch.object(compaction_main, '_build_spark_chunked_compaction_service', return_value=spark_service) as build_fallback,
            patch.object(compaction_main, 'settings', settings),
        ):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
                    'stream': True,
                },
                headers={'Authorization': 'Bearer test-token'},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('spark compacted', response.text)
        self.assertNotIn('event: response.failed', response.text)
        self.assertEqual(build_fallback.call_args.args[0]['authorization'], 'Bearer test-token')
        self.assertEqual(spark_service.compact_calls[0]['current_request'], 'Current thread')

    def test_responses_reuses_inflight_inline_compaction_job(self):
        service = DummyInlineService()
        payload = {
            'model': 'gpt-5.4',
            'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
            'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
            'stream': True,
        }
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main, 'inline_compaction_jobs', compaction_main.InlineCompactionJobManager(service)),
        ):
            job1, created1 = compaction_main.inline_compaction_jobs.get_or_create(
                payload,
                compaction_main.anthropic_request_from_responses(compaction_main.sanitize_inline_compaction_payload(payload)),
            )
            job2, created2 = compaction_main.inline_compaction_jobs.get_or_create(
                payload,
                compaction_main.anthropic_request_from_responses(compaction_main.sanitize_inline_compaction_payload(payload)),
            )
            final = job2.wait()

        self.assertTrue(created1)
        self.assertFalse(created2)
        self.assertIs(job1, job2)
        self.assertEqual(final['content'][0]['text'], 'compact summary')
        self.assertEqual(len(service.inline_requests), 1)

    def test_responses_stream_emits_failure_event_instead_of_broken_stream(self):
        class FailingInlineService(DummyCompactionService):
            def stream_inline_compact_from_anthropic(self, req, *, progress_callback=None):
                if progress_callback is not None:
                    progress_callback('normalize_completed', compactable_count=1, preserved_tail_count=0)
                raise RuntimeError('boom')

        service = FailingInlineService()
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main, 'inline_compaction_jobs', compaction_main.InlineCompactionJobManager(service)),
        ):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
                    'stream': True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('event: response.failed', response.text)
        self.assertIn('boom', response.text)

    def test_failed_inline_compaction_job_is_not_reused(self):
        class FlakyInlineService(DummyCompactionService):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def stream_inline_compact_from_anthropic(self, req, *, progress_callback=None):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('boom')
                yield {
                    'type': 'final',
                    'response': {
                        'id': 'msg_router_local',
                        'type': 'message',
                        'role': 'assistant',
                        'model': 'qwen3.5-9b:iq4_xs',
                        'content': [{'type': 'text', 'text': 'recovered summary'}],
                        'usage': {'input_tokens': 5, 'output_tokens': 3},
                    },
                }

        service = FlakyInlineService()
        payload = {
            'model': 'gpt-5.4',
            'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
            'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Current thread'}]}],
            'stream': True,
        }
        with (
            patch.object(compaction_main, 'service', service),
            patch.object(compaction_main, 'inline_compaction_jobs', compaction_main.InlineCompactionJobManager(service)),
        ):
            request = compaction_main.anthropic_request_from_responses(compaction_main.sanitize_inline_compaction_payload(payload))
            job1, created1 = compaction_main.inline_compaction_jobs.get_or_create(payload, request)
            with self.assertRaisesRegex(RuntimeError, 'boom'):
                job1.wait()
            job2, created2 = compaction_main.inline_compaction_jobs.get_or_create(payload, request)
            final = job2.wait()

        self.assertTrue(created1)
        self.assertTrue(created2)
        self.assertIsNot(job1, job2)
        self.assertEqual(final['content'][0]['text'], 'recovered summary')

    def test_responses_ignores_sentinel_in_historical_tool_output(self):
        service = DummyInlineService()
        upstream = Mock()
        upstream.status_code = 200
        upstream.content = json.dumps({'object': 'response', 'output_text': 'upstream'}).encode('utf-8')
        upstream.headers = {'content-type': 'application/json'}
        upstream.close = Mock()
        with patch.object(compaction_main, 'service', service), patch.object(compaction_main._UPSTREAM, 'post', return_value=upstream):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'input': [
                        {
                            'type': 'function_call_output',
                            'call_id': 'call_1',
                            'output': 'INLINE_COMPACT_SENTINEL=<<<LOCAL_COMPACT>>>',
                        },
                        {
                            'type': 'message',
                            'role': 'user',
                            'content': [{'type': 'input_text', 'text': 'Keep going with the audit.'}],
                        },
                    ],
                },
                headers={'Authorization': 'Bearer test-token'},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['output_text'], 'upstream')
        self.assertEqual(service.inline_requests, [])

    def test_openai_chat_completions_is_explicitly_unsupported(self):
        service = DummyCompactionService()
        with patch.object(compaction_main, 'service', service):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/chat/completions',
                json={
                    'model': 'gpt-5.4',
                    'messages': [{'role': 'user', 'content': 'Check the repo'}],
                },
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn('Unsupported transport', response.json()['error'])
        self.assertEqual(service.compact_calls, [])

    def test_ollama_chat_is_explicitly_unsupported(self):
        service = DummyCompactionService()
        with patch.object(compaction_main, 'service', service):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/api/chat',
                json={
                    'model': 'gpt-5.4',
                    'messages': [{'role': 'user', 'content': 'Check the repo'}],
                },
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn('Unsupported transport', response.json()['error'])
        self.assertEqual(service.compact_calls, [])

    def test_anthropic_messages_proxied_upstream(self):
        service = DummyCompactionService()
        with patch.object(compaction_main, 'service', service), \
             patch.object(compaction_main, '_UPSTREAM') as mock_upstream:
            mock_resp = unittest.mock.MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b'{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}'
            mock_resp.headers = {'content-type': 'application/json'}
            mock_upstream.post.return_value = mock_resp
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/messages',
                json={
                    'model': 'claude-sonnet-4-6-20250514',
                    'messages': [{'role': 'user', 'content': 'Check the repo'}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.compact_calls, [])
        mock_upstream.post.assert_called_once()
        call_url = mock_upstream.post.call_args[0][0]
        self.assertIn('/v1/messages', call_url)

    def test_anthropic_messages_sentinel_triggers_compaction(self):
        """Requests with the compaction sentinel in user message should be handled locally."""
        service = DummyCompactionService()
        service.compact_result = {
            'task_state': 'working', 'decisions': [], 'failures': [],
            'next_steps': [], 'structured_handoff': {},
        }
        service.handoff_flow_result = {
            'durable_memory': [{'name': 'TASK_STATE.md', 'content': 'test'}],
            'structured_handoff': {},
        }
        with patch.object(compaction_main, 'service', service), \
             patch.object(compaction_main, 'anthropic_inline_compaction_jobs') as mock_jobs:
            mock_job = unittest.mock.MagicMock()
            mock_job.key = 'test_key'
            mock_job.wait.return_value = {
                'id': 'msg_router_local', 'type': 'message', 'role': 'assistant',
                'model': 'qwen3.5:9b',
                'content': [{'type': 'text', 'text': 'compacted summary'}],
                'stop_reason': 'end_turn', 'stop_sequence': None,
                'usage': {'input_tokens': 100, 'output_tokens': 50},
            }
            mock_jobs.get_or_create.return_value = (mock_job, True)
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/messages',
                json={
                    'model': 'claude-sonnet-4-6-20250514',
                    'max_tokens': 1024,
                    'messages': [
                        {'role': 'user', 'content': 'Hello'},
                        {'role': 'assistant', 'content': 'Hi there'},
                        {'role': 'user', 'content': '<<<LOCAL_COMPACT>>>'},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['type'], 'message')
        self.assertEqual(body['content'][0]['text'], 'compacted summary')
        mock_jobs.get_or_create.assert_called_once()

    def test_anthropic_messages_sentinel_in_system_triggers_compaction(self):
        """Sentinel in system prompt should also trigger compaction."""
        service = DummyCompactionService()
        with patch.object(compaction_main, 'service', service), \
             patch.object(compaction_main, 'anthropic_inline_compaction_jobs') as mock_jobs:
            mock_job = unittest.mock.MagicMock()
            mock_job.key = 'test_key'
            mock_job.wait.return_value = {
                'id': 'msg_router_local', 'type': 'message', 'role': 'assistant',
                'model': 'qwen3.5:9b',
                'content': [{'type': 'text', 'text': 'compacted'}],
                'stop_reason': 'end_turn', 'stop_sequence': None,
                'usage': {'input_tokens': 50, 'output_tokens': 20},
            }
            mock_jobs.get_or_create.return_value = (mock_job, True)
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/messages',
                json={
                    'model': 'claude-sonnet-4-6-20250514',
                    'max_tokens': 1024,
                    'system': 'You are helpful. <<<LOCAL_COMPACT>>>',
                    'messages': [{'role': 'user', 'content': 'Continue'}],
                },
            )

        self.assertEqual(response.status_code, 200)
        mock_jobs.get_or_create.assert_called_once()

    def test_anthropic_compaction_failure_falls_back_to_upstream(self):
        """If local compaction fails, should fall back to proxying upstream."""
        service = DummyCompactionService()
        with patch.object(compaction_main, 'service', service), \
             patch.object(compaction_main, 'anthropic_inline_compaction_jobs') as mock_jobs, \
             patch.object(compaction_main, '_UPSTREAM') as mock_upstream:
            mock_job = unittest.mock.MagicMock()
            mock_job.key = 'test_key'
            mock_job.wait.side_effect = RuntimeError('compaction failed')
            mock_jobs.get_or_create.return_value = (mock_job, True)
            mock_resp = unittest.mock.MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b'{"id":"msg_1","type":"message","content":[{"type":"text","text":"upstream"}]}'
            mock_resp.headers = {'content-type': 'application/json'}
            mock_upstream.post.return_value = mock_resp
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/messages',
                json={
                    'model': 'claude-sonnet-4-6-20250514',
                    'max_tokens': 1024,
                    'messages': [{'role': 'user', 'content': '<<<LOCAL_COMPACT>>>'}],
                },
            )

        self.assertEqual(response.status_code, 200)
        mock_upstream.post.assert_called_once()
