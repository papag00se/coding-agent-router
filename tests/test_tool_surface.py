from __future__ import annotations

import json
import unittest

from app.compat import anthropic_request_from_responses, responses_response
from app.tool_surface import translate_responses_tools


class TestToolSurface(unittest.TestCase):
    def test_translate_builtin_tools_to_small_surface(self):
        tools, aliases = translate_responses_tools(
            [
                {'type': 'function', 'name': 'rg', 'description': 'search', 'parameters': {'type': 'object'}},
                {'type': 'custom', 'name': 'apply_patch', 'description': 'patch'},
            ]
        )

        self.assertEqual([tool['name'] for tool in tools], ['search_text', 'write_patch'])
        self.assertEqual(aliases['search_text']['original_name'], 'rg')
        self.assertEqual(aliases['write_patch']['original_name'], 'apply_patch')

    def test_responses_request_uses_translated_tools(self):
        req = anthropic_request_from_responses(
            {
                'model': 'gpt-5.4',
                'input': 'hi',
                'tools': [
                    {'type': 'function', 'name': 'rg', 'description': 'search', 'parameters': {'type': 'object'}},
                    {'type': 'custom', 'name': 'apply_patch', 'description': 'patch'},
                ],
            }
        )

        self.assertEqual([tool['name'] for tool in req.tools], ['search_text', 'write_patch'])
        self.assertEqual(req.metadata['_tool_aliases']['write_patch']['original_name'], 'apply_patch')

    def test_responses_response_maps_aliases_back_to_original_names(self):
        body = responses_response(
            {
                'model': 'qwen3-coder',
                'content': [
                    {'type': 'tool_use', 'id': 'call_1', 'name': 'write_patch', 'input': {'patch': '*** Begin Patch\n*** End Patch\n'}},
                    {'type': 'tool_use', 'id': 'call_2', 'name': 'search_text', 'input': {'pattern': 'foo'}},
                ],
                'usage': {'input_tokens': 1, 'output_tokens': 1},
                'request_metadata': {
                    '_tool_aliases': {
                        'write_patch': {'original_name': 'apply_patch', 'mode': 'patch'},
                        'search_text': {'original_name': 'rg', 'mode': 'identity'},
                    }
                },
            },
            {'model': 'gpt-5.4'},
        )

        outputs = [item for item in body['output'] if item['type'] == 'function_call']
        self.assertEqual(outputs[0]['name'], 'apply_patch')
        self.assertEqual(json.loads(outputs[0]['arguments']), {'input': '*** Begin Patch\n*** End Patch\n'})
        self.assertEqual(outputs[1]['name'], 'rg')
        self.assertEqual(json.loads(outputs[1]['arguments']), {'pattern': 'foo'})
