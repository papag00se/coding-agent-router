from __future__ import annotations

import unittest

from app.compaction.normalize import normalize_transcript_for_compaction, sanitize_inline_compaction_payload


class TestCompactionNormalize(unittest.TestCase):
    def test_normalize_transcript_strips_attachments_and_encrypted_content(self) -> None:
        normalized = normalize_transcript_for_compaction(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': 'first'},
                        {'type': 'input_image', 'image_url': 'https://example.test/cat.png'},
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'tool_result', 'tool_use_id': 'call_1', 'content': 'done', 'encrypted_content': 'secret'},
                    ],
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': 'latest'},
                        {'type': 'input_image', 'image_url': 'https://example.test/dog.png'},
                    ],
                },
            ],
            max_item_tokens=50,
        )

        self.assertEqual(
            normalized.compactable_items,
            [
                {'role': 'user', 'content': [{'type': 'input_text', 'text': 'first'}]},
            ],
        )
        self.assertEqual(
            normalized.preserved_tail,
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': 'latest'},
                        {'type': 'input_image', 'image_url': 'https://example.test/dog.png'},
                    ],
                }
            ],
        )

    def test_normalize_transcript_drops_oversize_compactable_blocks_but_preserves_latest_raw_item(self) -> None:
        normalized = normalize_transcript_for_compaction(
            [
                {'role': 'assistant', 'content': 'older'},
                {'role': 'user', 'content': [{'type': 'input_text', 'text': 'x' * 500}]},
            ],
            max_item_tokens=30,
        )

        self.assertEqual(normalized.compactable_items, [{'role': 'assistant', 'content': 'older'}])
        self.assertEqual(normalized.preserved_tail, [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'x' * 500}]}])

    def test_normalize_transcript_drops_historical_tool_results_and_function_outputs(self) -> None:
        normalized = normalize_transcript_for_compaction(
            [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': 'call_1',
                            'content': 'historical tool output',
                        }
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'function_call_output',
                            'call_id': 'call_2',
                            'output': {'stdout': 'historical function output'},
                        }
                    ],
                },
                {'role': 'assistant', 'content': 'latest'},
            ],
            max_item_tokens=10_000,
        )

        self.assertEqual(normalized.compactable_items, [])

    def test_normalize_transcript_preserves_latest_raw_tool_result_turn(self) -> None:
        normalized = normalize_transcript_for_compaction(
            [
                {'role': 'assistant', 'content': 'older'},
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': 'call_1',
                            'content': '{"type":"input_image","image_url":"data:image/png;base64,AAAA"}',
                        }
                    ],
                },
            ],
            max_item_tokens=10_000,
        )

        self.assertEqual(normalized.compactable_items, [{'role': 'assistant', 'content': 'older'}])
        self.assertEqual(
            normalized.preserved_tail,
            [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': 'call_1',
                            'content': '{"type":"input_image","image_url":"data:image/png;base64,AAAA"}',
                        }
                    ],
                }
            ],
        )

    def test_sanitize_inline_compaction_payload_strips_only_codex_bootstrap_instructions(self) -> None:
        payload = {
            'instructions': '<<<LOCAL_COMPACT>>>\nYou are Codex, a coding agent based on GPT-5.',
            'input': [],
        }
        sanitized = sanitize_inline_compaction_payload(payload)
        self.assertNotIn('instructions', sanitized)

        preserved = sanitize_inline_compaction_payload(
            {
                'instructions': '<<<LOCAL_COMPACT>>> Summarize the thread.',
                'input': [],
            }
        )
        self.assertEqual(preserved['instructions'], '<<<LOCAL_COMPACT>>> Summarize the thread.')
