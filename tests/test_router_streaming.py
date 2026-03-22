from __future__ import annotations

import unittest

from app.models import AnthropicMessagesRequest, RouteDecision
from app.router import RoutingService


class _FakeCoderClient:
    def chat_stream(self, *args, **kwargs):
        yield {
            'message': {
                'content': (
                    'I will search for it.\n\n'
                    '{"type":"tool_use","id":"call_abc123","name":"run_shell","input":{"command":"find .'
                )
            }
        }
        yield {
            'message': {
                'content': ' -type f"}}'
            }
        }


class TestRouterStreaming(unittest.TestCase):
    def test_structured_coder_stream_recovers_embedded_tool_use(self):
        service = RoutingService()
        service.route = lambda *args, **kwargs: RouteDecision(route='local_coder', confidence=1.0, reason='test')
        service.coder_client = _FakeCoderClient()

        req = AnthropicMessagesRequest.model_validate(
            {
                'model': 'gpt-5.4',
                'messages': [{'role': 'user', 'content': 'Find it'}],
                'tools': [{'name': 'run_shell', 'description': 'Run shell', 'input_schema': {'type': 'object'}}],
            }
        )

        events = list(service.stream_from_anthropic(req))

        self.assertEqual(
            [event['type'] for event in events],
            ['text_delta', 'tool_calls', 'final'],
        )
        self.assertEqual(events[0]['delta'], 'I will search for it.')
        self.assertEqual(events[1]['tool_calls'][0]['function']['name'], 'run_shell')
        self.assertEqual(events[1]['tool_calls'][0]['function']['arguments'], {'command': 'find . -type f'})
        final_content = events[2]['response']['content']
        self.assertEqual(final_content[0]['text'], 'I will search for it.')
        self.assertEqual(final_content[1]['name'], 'run_shell')
