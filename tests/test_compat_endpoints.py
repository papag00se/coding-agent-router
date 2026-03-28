from __future__ import annotations

import json
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import compat
from app import main


class DummyService:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.request = None
        self.delay_seconds = delay_seconds

    def invoke_from_anthropic(self, req):
        self.request = req.model_dump()
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return {
            'id': 'msg_router_local',
            'type': 'message',
            'role': 'assistant',
            'model': 'local_reasoner',
            'content': [{'type': 'text', 'text': 'ok'}],
            'usage': {'input_tokens': 7, 'output_tokens': 3},
            'route_decision': {'route': 'local_reasoner', 'confidence': 1.0, 'reason': 'test'},
            'raw_backend': {'created_at': '2026-03-19T00:00:00Z'},
        }


class DummyToolService(DummyService):
    def invoke_from_anthropic(self, req):
        self.request = req.model_dump()
        return {
            'id': 'msg_router_local',
            'type': 'message',
            'role': 'assistant',
            'model': 'local_coder',
            'content': [{'type': 'tool_use', 'id': 'call_1', 'name': 'read_file', 'input': {'path': 'a.txt'}}],
            'usage': {'input_tokens': 5, 'output_tokens': 2},
            'route_decision': {'route': 'local_coder', 'confidence': 1.0, 'reason': 'test'},
            'raw_backend': {'created_at': '2026-03-19T00:00:00Z'},
        }


class DummyStreamingService(DummyService):
    def stream_from_anthropic(self, req):
        self.request = req.model_dump()
        yield {'type': 'text_delta', 'delta': 'o'}
        yield {'type': 'text_delta', 'delta': 'k'}
        yield {
            'type': 'final',
            'response': {
                'id': 'msg_router_local',
                'type': 'message',
                'role': 'assistant',
                'model': 'local_reasoner',
                'content': [{'type': 'text', 'text': 'ok'}],
                'usage': {'input_tokens': 7, 'output_tokens': 3},
                'route_decision': {'route': 'local_reasoner', 'confidence': 1.0, 'reason': 'test'},
                'raw_backend': {'created_at': '2026-03-19T00:00:00Z'},
            },
        }


class DummyStreamingToolService(DummyService):
    def stream_from_anthropic(self, req):
        self.request = req.model_dump()
        yield {'type': 'text_delta', 'delta': 'Let me check.'}
        yield {
            'type': 'tool_calls',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {
                        'name': 'find_files',
                        'arguments': {'pattern': 'Makefile'},
                    },
                }
            ],
        }
        yield {
            'type': 'final',
            'response': {
                'id': 'msg_router_local',
                'type': 'message',
                'role': 'assistant',
                'model': 'local_coder',
                'content': [
                    {'type': 'text', 'text': 'Let me check.'},
                    {'type': 'tool_use', 'id': 'call_1', 'name': 'find_files', 'input': {'pattern': 'Makefile'}},
                ],
                'usage': {'input_tokens': 7, 'output_tokens': 3},
                'route_decision': {'route': 'local_coder', 'confidence': 1.0, 'reason': 'test'},
                'raw_backend': {'created_at': '2026-03-19T00:00:00Z'},
            },
        }


class DummyStreamingAliasToolService(DummyService):
    def stream_from_anthropic(self, req):
        self.request = req.model_dump()
        yield {'type': 'text_delta', 'delta': 'Checking.'}
        yield {
            'type': 'tool_calls',
            'tool_calls': [
                {
                    'id': 'call_exec',
                    'type': 'function',
                    'function': {
                        'name': 'run_shell',
                        'arguments': {'command': 'grep foo .'},
                    },
                }
            ],
        }
        yield {
            'type': 'final',
            'response': {
                'id': 'msg_router_local',
                'type': 'message',
                'role': 'assistant',
                'model': 'local_coder',
                'content': [
                    {'type': 'text', 'text': 'Checking.'},
                    {'type': 'tool_use', 'id': 'call_exec', 'name': 'run_shell', 'input': {'command': 'grep foo .'}},
                ],
                'usage': {'input_tokens': 7, 'output_tokens': 3},
                'route_decision': {'route': 'local_coder', 'confidence': 1.0, 'reason': 'test'},
                'request_metadata': {
                    '_tool_aliases': {
                        'run_shell': {'original_name': 'exec_command', 'mode': 'exec_command'},
                    }
                },
                'raw_backend': {'created_at': '2026-03-19T00:00:00Z'},
            },
        }


class TestCompatEndpoints(unittest.TestCase):
    def test_openai_chat_completions_maps_request_and_response(self):
        service = DummyService()
        with patch.object(main, 'service', service):
            client = TestClient(main.app)
            response = client.post(
                '/v1/chat/completions',
                json={
                    'model': 'router',
                    'messages': [
                        {'role': 'system', 'content': 'Be concise.'},
                        {'role': 'user', 'content': 'Say ok'},
                    ],
                    'max_tokens': 32,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.request['system'], 'Be concise.')
        self.assertEqual(service.request['messages'], [{'role': 'user', 'content': 'Say ok'}])
        body = response.json()
        self.assertEqual(body['choices'][0]['message']['content'], 'ok')
        self.assertEqual(body['usage']['total_tokens'], 10)

    def test_ollama_chat_maps_tool_calls(self):
        service = DummyToolService()
        with patch.object(main, 'service', service):
            client = TestClient(main.app)
            response = client.post(
                '/api/chat',
                json={
                    'model': 'router',
                    'messages': [{'role': 'user', 'content': 'Use a tool'}],
                    'tools': [
                        {
                            'type': 'function',
                            'function': {
                                'name': 'read_file',
                                'description': 'Read a file',
                                'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}},
                            },
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.request['tools'][0]['name'], 'read_file')
        body = response.json()
        self.assertEqual(body['message']['tool_calls'][0]['function']['name'], 'read_file')
        self.assertEqual(body['message']['tool_calls'][0]['function']['arguments']['path'], 'a.txt')

    def test_openai_and_ollama_discovery_endpoints_exist(self):
        client = TestClient(main.app)
        self.assertEqual(client.get('/v1/models').json()['data'][0]['id'], 'gpt-5.4')
        self.assertEqual(client.get('/api/version').json()['version'], '0.0.0-router')
        self.assertEqual(client.get('/api/tags').json()['models'][0]['name'], 'router')

    def test_health_reports_single_codex_cli_toggle(self):
        client = TestClient(main.app)
        body = client.get('/health').json()
        self.assertIn('codex_cli_enabled', body)
        self.assertNotIn('codex_cli_backend_enabled', body)

    def test_responses_maps_request_and_response(self):
        service = DummyService()
        with patch.object(main, 'service', service):
            client = TestClient(main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'router',
                    'instructions': 'Be concise.',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Say ok'}]}],
                    'max_output_tokens': 32,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.request['system'], 'Be concise.')
        self.assertEqual(service.request['messages'], [{'role': 'user', 'content': [{'type': 'text', 'text': 'Say ok'}]}])
        body = response.json()
        self.assertEqual(body['object'], 'response')
        self.assertEqual(body['model'], 'router')
        self.assertTrue(body['output_text'].startswith('[router]'))
        self.assertTrue(body['output_text'].endswith('ok'))
        self.assertEqual(body['usage']['total_tokens'], 10)

    def test_responses_stream_returns_completed_event(self):
        service = DummyStreamingService()
        with patch.object(main, 'service', service):
            client = TestClient(main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'router',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Say ok'}]}],
                    'stream': True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('event: response.created', response.text)
        self.assertIn('event: response.output_text.delta', response.text)
        self.assertIn('event: response.completed', response.text)

    def test_responses_stream_emits_function_call_after_text(self):
        service = DummyStreamingToolService()
        with patch.object(main, 'service', service):
            client = TestClient(main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Find Makefile'}]}],
                    'stream': True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('event: response.output_text.delta', response.text)
        self.assertIn('event: response.function_call_arguments.done', response.text)
        self.assertIn('"name": "find_files"', response.text)
        self.assertEqual(response.text.count('event: response.function_call_arguments.done'), 1)

    def test_responses_stream_maps_alias_tool_call_back_to_original(self):
        service = DummyStreamingAliasToolService()
        with patch.object(main, 'service', service):
            client = TestClient(main.app)
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'gpt-5.4',
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Search'}]}],
                    'tools': [
                        {
                            'type': 'function',
                            'name': 'exec_command',
                            'description': 'Run a shell command',
                            'parameters': {'type': 'object', 'properties': {'cmd': {'type': 'string'}}},
                        }
                    ],
                    'stream': True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"name": "exec_command"', response.text)
        self.assertIn('\\"cmd\\": \\"grep foo .\\"', response.text)
        self.assertNotIn('"name": "run_shell"', response.text)

    def test_openai_chat_stream_returns_sse_done(self):
        service = DummyService()
        with patch.object(main, 'service', service):
            client = TestClient(main.app)
            response = client.post(
                '/v1/chat/completions',
                json={'model': 'router', 'messages': [{'role': 'user', 'content': 'Say ok'}], 'stream': True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('data: [DONE]', response.text)

    def test_openai_chat_stream_emits_keepalive_before_done(self):
        service = DummyService(delay_seconds=0.05)
        with patch.object(main, 'service', service), patch.object(compat, '_SSE_KEEPALIVE_INTERVAL_SECONDS', 0.01):
            client = TestClient(main.app)
            response = client.post(
                '/v1/chat/completions',
                json={'model': 'router', 'messages': [{'role': 'user', 'content': 'Say ok'}], 'stream': True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(': keepalive', response.text)
        self.assertIn('data: [DONE]', response.text)

    def test_ollama_chat_stream_returns_ndjson(self):
        service = DummyService()
        with patch.object(main, 'service', service):
            client = TestClient(main.app)
            response = client.post(
                '/api/chat',
                json={'model': 'router', 'messages': [{'role': 'user', 'content': 'Say ok'}], 'stream': True},
            )

        self.assertEqual(response.status_code, 200)
        lines = [line for line in response.text.splitlines() if line.strip()]
        self.assertTrue(lines)
        self.assertTrue(json.loads(lines[-1])['done'])

    def test_ollama_chat_stream_emits_blank_keepalive_before_ndjson(self):
        service = DummyService(delay_seconds=0.05)
        with patch.object(main, 'service', service), patch.object(compat, '_SSE_KEEPALIVE_INTERVAL_SECONDS', 0.01):
            client = TestClient(main.app)
            response = client.post(
                '/api/chat',
                json={'model': 'router', 'messages': [{'role': 'user', 'content': 'Say ok'}], 'stream': True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.text.startswith('\n'))
        lines = [line for line in response.text.splitlines() if line.strip()]
        self.assertTrue(lines)
        self.assertTrue(json.loads(lines[-1])['done'])
