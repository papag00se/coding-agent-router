from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.app_server import CodexAppServerBridge


class DummyService:
    def __init__(self):
        self.requests = []

    def invoke_from_anthropic(self, req):
        self.requests.append(req)
        preferred_backend = (req.metadata or {}).get('preferred_backend')
        text = 'ok'
        if preferred_backend == 'codex_cli':
            text = 'codex'
        return {
            'id': 'msg_router_local',
            'type': 'message',
            'role': 'assistant',
            'model': 'local_reasoner',
            'content': [{'type': 'text', 'text': text}],
            'usage': {'input_tokens': 1, 'output_tokens': 1},
            'raw_backend': {},
        }

    def compact_session(self, session_id, items, *, current_request, repo_context, refresh_if_needed):
        return {
            'stable_task_definition': 'summary',
            'repo_state': {'cwd': repo_context.get('cwd')},
            'key_decisions': ['keep context local'],
            'unresolved_work': ['continue work'],
            'latest_plan': ['resume from compacted state'],
            'failures_to_avoid': ['lose history'],
            'recent_raw_turns': items[-2:],
            'current_request': current_request,
        }

    def build_compaction_handoff_flow(self, session_id, *, current_request=None):
        return {
            'durable_memory': [
                {'name': 'TASK_STATE.md', 'content': 'task'},
                {'name': 'DECISIONS.md', 'content': 'decisions'},
                {'name': 'FAILURES_TO_AVOID.md', 'content': 'failures'},
                {'name': 'NEXT_STEPS.md', 'content': 'next'},
                {'name': 'SESSION_HANDOFF.md', 'content': 'handoff'},
            ],
            'structured_handoff': {'stable_task_definition': 'summary'},
            'recent_raw_turns': [{'role': 'assistant', 'content': 'recent raw'}],
            'current_request': current_request or '',
        }


class TestAppServerBridge(unittest.TestCase):
    def test_initialize_thread_start_and_turn_start(self):
        service = DummyService()
        bridge = CodexAppServerBridge(service)
        with patch.object(main, 'app_server', bridge):
            client = TestClient(main.app)
            with client.websocket_connect('/app-server/ws') as ws:
                ws.send_json({'id': 1, 'method': 'initialize', 'params': {'clientInfo': {'name': 'vscode', 'version': '1.0'}}})
                self.assertEqual(ws.receive_json()['result']['userAgent'], 'coding-agent-router-app-server')

                ws.send_json({'id': 2, 'method': 'thread/start', 'params': {'cwd': '/tmp', 'model': 'router'}})
                response = ws.receive_json()
                self.assertEqual(response['id'], 2)
                thread = response['result']['thread']
                self.assertEqual(thread['source'], 'appServer')
                self.assertEqual(thread['status']['type'], 'idle')

                started = ws.receive_json()
                self.assertEqual(started['method'], 'thread/started')
                thread_id = started['params']['thread']['id']

                ws.send_json(
                    {
                        'id': 3,
                        'method': 'turn/start',
                        'params': {'threadId': thread_id, 'input': [{'type': 'text', 'text': 'Reply with exactly: ok'}]},
                    }
                )
                self.assertEqual(ws.receive_json()['method'], 'thread/status/changed')
                self.assertEqual(ws.receive_json()['method'], 'turn/started')
                self.assertEqual(ws.receive_json()['method'], 'item/started')
                completed_item = ws.receive_json()
                self.assertEqual(completed_item['method'], 'item/completed')
                self.assertEqual(completed_item['params']['item']['text'], 'ok')
                self.assertEqual(ws.receive_json()['method'], 'turn/completed')
                self.assertEqual(ws.receive_json()['method'], 'thread/status/changed')
                final_response = ws.receive_json()
                self.assertEqual(final_response['id'], 3)
                self.assertEqual(final_response['result']['turn']['status'], 'completed')

    def test_thread_compact_reduces_follow_up_context(self):
        service = DummyService()
        bridge = CodexAppServerBridge(service)
        with patch.object(main, 'app_server', bridge):
            client = TestClient(main.app)
            with client.websocket_connect('/app-server/ws') as ws:
                ws.send_json({'id': 1, 'method': 'initialize', 'params': {'clientInfo': {'name': 'vscode', 'version': '1.0'}}})
                ws.receive_json()

                ws.send_json({'id': 2, 'method': 'thread/start', 'params': {'cwd': '/tmp', 'model': 'router'}})
                thread = ws.receive_json()['result']['thread']
                thread_id = ws.receive_json()['params']['thread']['id']

                ws.send_json({'id': 3, 'method': 'turn/start', 'params': {'threadId': thread_id, 'input': [{'type': 'text', 'text': 'first request'}]}})
                for _ in range(7):
                    ws.receive_json()
                first_req = service.requests[-1]
                self.assertEqual(len(first_req.messages), 1)
                self.assertIsNone(first_req.system)

                ws.send_json({'id': 4, 'method': 'thread/compact/start', 'params': {'threadId': thread_id}})
                started = ws.receive_json()
                self.assertEqual(started['method'], 'item/started')
                self.assertEqual(started['params']['item']['type'], 'contextCompaction')
                completed = ws.receive_json()
                self.assertEqual(completed['method'], 'item/completed')
                compacted = ws.receive_json()
                self.assertEqual(compacted['method'], 'thread/compacted')
                compact_response = ws.receive_json()
                self.assertEqual(compact_response['id'], 4)
                self.assertEqual(compact_response['result']['threadId'], thread_id)

                ws.send_json({'id': 5, 'method': 'turn/start', 'params': {'threadId': thread_id, 'input': [{'type': 'text', 'text': 'second request'}]}})
                for _ in range(7):
                    ws.receive_json()
                second_req = service.requests[-1]
                self.assertGreaterEqual(len(second_req.messages), 1)
                self.assertEqual(second_req.messages[-1].content, 'second request')
                self.assertIn('Durable memory:', second_req.system)
                self.assertIn('Current request:\n\nsecond request', second_req.system)

    def test_compaction_only_mode_forces_codex_cli_and_persists_state(self):
        service = DummyService()
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = CodexAppServerBridge(service, mode='compaction_only', state_dir=tmpdir)
            with patch.object(main, 'app_server', bridge), patch('app.app_server._TRANSPORT_LOG_PATH', Path(tmpdir) / 'transport.jsonl'):
                client = TestClient(main.app)
                with client.websocket_connect('/app-server/ws') as ws:
                    ws.send_json({'id': 1, 'method': 'initialize', 'params': {'clientInfo': {'name': 'vscode', 'version': '1.0'}}})
                    ws.receive_json()

                    ws.send_json({'id': 2, 'method': 'thread/start', 'params': {'cwd': '/tmp', 'model': 'router'}})
                    ws.receive_json()
                    thread_id = ws.receive_json()['params']['thread']['id']

                    ws.send_json({'id': 3, 'method': 'turn/start', 'params': {'threadId': thread_id, 'input': [{'type': 'text', 'text': 'first request'}]}})
                    for _ in range(7):
                        ws.receive_json()
                    first_req = service.requests[-1]
                    self.assertEqual(first_req.metadata['preferred_backend'], 'codex_cli')
                    self.assertEqual(first_req.metadata['cwd'], '/tmp')
                    self.assertEqual(first_req.messages[-1].content, 'first request')

                    ws.send_json({'id': 4, 'method': 'thread/compact/start', 'params': {'threadId': thread_id}})
                    for _ in range(4):
                        ws.receive_json()
            events = [json.loads(line) for line in (Path(tmpdir) / 'transport.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(events[0]['event'], 'thread_compact_start')
            self.assertEqual(events[1]['event'], 'local_compaction_completed')

            reloaded = CodexAppServerBridge(service, mode='compaction_only', state_dir=tmpdir)
            state = reloaded._get_thread_state(thread_id)
            self.assertIsNotNone(state)
            self.assertIsNotNone(state.compacted_flow)
            self.assertGreaterEqual(len(state.history_items), 1)

            with patch.object(main, 'app_server', reloaded):
                client = TestClient(main.app)
                with client.websocket_connect('/app-server/ws') as ws:
                    ws.send_json({'id': 5, 'method': 'initialize', 'params': {'clientInfo': {'name': 'vscode', 'version': '1.0'}}})
                    ws.receive_json()
                    ws.send_json({'id': 6, 'method': 'turn/start', 'params': {'threadId': thread_id, 'input': [{'type': 'text', 'text': 'after compact'}]}})
                    for _ in range(7):
                        ws.receive_json()
                    followup_req = service.requests[-1]
                    self.assertEqual(followup_req.metadata['preferred_backend'], 'codex_cli')
                    self.assertEqual(followup_req.metadata['cwd'], '/tmp')
                    self.assertIn('Durable memory:', followup_req.system)
