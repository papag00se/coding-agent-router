from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app import compaction_main


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

    def test_anthropic_messages_is_explicitly_unsupported(self):
        service = DummyCompactionService()
        with patch.object(compaction_main, 'service', service):
            client = TestClient(compaction_main.app)
            response = client.post(
                '/v1/messages',
                json={
                    'model': 'gpt-5.4',
                    'messages': [{'role': 'user', 'content': 'Check the repo'}],
                },
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn('Unsupported transport', response.json()['error'])
        self.assertEqual(service.compact_calls, [])
