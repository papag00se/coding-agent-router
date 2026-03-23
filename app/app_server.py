from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import WebSocket
from starlette.concurrency import run_in_threadpool

from .compaction import render_compacted_flow
from .compaction_transport import compaction_payload_fields, record_transport_event
from .config import settings
from .models import AnthropicMessagesRequest

_TRANSPORT_LOG_PATH = Path('state/compaction_transport.jsonl')


def _record_transport_event(event: str, **fields: Any) -> None:
    record_transport_event(_TRANSPORT_LOG_PATH, event, **fields)


def _now() -> int:
    return int(time.time())


def _jsonrpc_response(request_id: Any, result: Any) -> Dict[str, Any]:
    return {'id': request_id, 'result': result}


def _jsonrpc_error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {'id': request_id, 'error': {'code': code, 'message': message}}


def _notification(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {'method': method, 'params': params}


def _sandbox_policy() -> Dict[str, Any]:
    return {'type': 'dangerFullAccess'}


def _flatten_user_input(items: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get('type')
        if kind == 'text' and item.get('text'):
            parts.append(item['text'])
        elif kind == 'image' and item.get('url'):
            parts.append(item['url'])
        elif kind == 'localImage' and item.get('path'):
            parts.append(item['path'])
        elif kind in {'skill', 'mention'}:
            name = item.get('name', '')
            path = item.get('path', '')
            parts.append(f'{name} {path}'.strip())
    return '\n'.join(part for part in parts if part)


def _response_text(response: Dict[str, Any]) -> str:
    text_parts: List[str] = []
    for block in response.get('content') or []:
        if isinstance(block, dict) and block.get('type') == 'text' and block.get('text'):
            text_parts.append(block['text'])
    return '\n'.join(text_parts).strip()


def _render_compacted_flow(flow: Dict[str, Any], *, current_request: str) -> str:
    return render_compacted_flow(flow, current_request=current_request)


@dataclass
class ThreadState:
    thread: Dict[str, Any]
    approval_policy: Any
    cwd: str
    model: str
    model_provider: str
    sandbox: Dict[str, Any]
    base_instructions: Optional[str] = None
    developer_instructions: Optional[str] = None
    turns: List[Dict[str, Any]] = field(default_factory=list)
    history_items: List[Dict[str, Any]] = field(default_factory=list)
    compacted_flow: Optional[Dict[str, Any]] = None


class CodexAppServerBridge:
    def __init__(self, service: Any, *, mode: str = "full", state_dir: Optional[str] = None) -> None:
        self.service = service
        self.mode = mode
        self.threads: Dict[str, ThreadState] = {}
        self.state_dir = Path(state_dir or settings.app_server_state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)

    async def handle_websocket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        while True:
            try:
                message = await websocket.receive_json()
            except Exception:
                return
            if not isinstance(message, dict):
                continue
            if 'id' not in message:
                continue
            method = message.get('method')
            params = message.get('params') or {}
            if method == 'initialize':
                await websocket.send_json(_jsonrpc_response(message['id'], {'userAgent': 'coding-agent-router-app-server'}))
                continue
            if method == 'thread/start':
                await websocket.send_json(self._handle_thread_start(message['id'], params))
                thread_id = self._thread_id_from_response(message['id'])
                if thread_id:
                    await websocket.send_json(_notification('thread/started', {'thread': self.threads[thread_id].thread}))
                continue
            if method == 'turn/start':
                await self._handle_turn_start(websocket, message['id'], params)
                continue
            if method == 'thread/compact/start':
                await self._handle_thread_compact_start(websocket, message['id'], params)
                continue
            if method == 'initialized':
                continue
            await websocket.send_json(_jsonrpc_error(message['id'], -32601, f'Unsupported method: {method}'))

    def _thread_id_from_response(self, request_id: Any) -> Optional[str]:
        return getattr(self, f'_last_thread_id_{request_id}', None)

    def _get_thread_state(self, thread_id: str) -> Optional[ThreadState]:
        state = self.threads.get(thread_id)
        if state is not None:
            return state
        path = self.state_dir / f'{thread_id}.json'
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding='utf-8'))
        state = ThreadState(
            thread=payload['thread'],
            approval_policy=payload['approval_policy'],
            cwd=payload['cwd'],
            model=payload['model'],
            model_provider=payload['model_provider'],
            sandbox=payload['sandbox'],
            base_instructions=payload.get('base_instructions'),
            developer_instructions=payload.get('developer_instructions'),
            turns=payload.get('turns') or [],
            history_items=payload.get('history_items') or [],
            compacted_flow=payload.get('compacted_flow'),
        )
        self.threads[thread_id] = state
        return state

    def _handle_thread_start(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        created_at = _now()
        thread_id = f'thread_{uuid.uuid4().hex}'
        cwd = params.get('cwd') or os.getcwd()
        model = params.get('model') or 'router'
        model_provider = params.get('modelProvider') or 'router'
        approval_policy = params.get('approvalPolicy') or 'never'
        sandbox = _sandbox_policy()
        thread = {
            'agentNickname': None,
            'agentRole': None,
            'cliVersion': 'router-app-server',
            'createdAt': created_at,
            'cwd': cwd,
            'gitInfo': None,
            'id': thread_id,
            'modelProvider': model_provider,
            'name': None,
            'path': None,
            'preview': '',
            'source': 'appServer',
            'status': {'type': 'idle'},
            'turns': [],
            'updatedAt': created_at,
        }
        self.threads[thread_id] = ThreadState(
            thread=thread,
            approval_policy=approval_policy,
            cwd=cwd,
            model=model,
            model_provider=model_provider,
            sandbox=sandbox,
            base_instructions=params.get('baseInstructions'),
            developer_instructions=params.get('developerInstructions'),
        )
        self._save_thread_state(thread_id)
        setattr(self, f'_last_thread_id_{request_id}', thread_id)
        return _jsonrpc_response(
            request_id,
            {
                'approvalPolicy': approval_policy,
                'cwd': cwd,
                'model': model,
                'modelProvider': model_provider,
                'reasoningEffort': None,
                'sandbox': sandbox,
                'thread': thread,
            },
        )

    async def _handle_turn_start(self, websocket: WebSocket, request_id: Any, params: Dict[str, Any]) -> None:
        thread_id = params.get('threadId')
        state = self._get_thread_state(thread_id)
        if not state:
            await websocket.send_json(_jsonrpc_error(request_id, -32602, f'Unknown threadId: {thread_id}'))
            return

        turn_id = f'turn_{uuid.uuid4().hex}'
        turn = {'id': turn_id, 'items': [], 'status': 'inProgress', 'error': None}
        state.thread['status'] = {'type': 'active', 'activeFlags': []}
        state.thread['updatedAt'] = _now()
        state.turns.append(turn)

        await websocket.send_json(_notification('thread/status/changed', {'threadId': thread_id, 'status': state.thread['status']}))
        await websocket.send_json(_notification('turn/started', {'threadId': thread_id, 'turn': turn}))

        user_text = _flatten_user_input(params.get('input') or [])
        user_message = {'role': 'user', 'content': user_text}
        if not state.thread['preview']:
            state.thread['preview'] = user_text

        system_parts = [part for part in [state.base_instructions, state.developer_instructions] if part]
        if state.compacted_flow:
            system_parts.append(_render_compacted_flow(state.compacted_flow, current_request=user_text))
        preferred_backend = 'codex_cli' if self.mode == 'compaction_only' else None
        req = AnthropicMessagesRequest.model_validate(
            {
                'model': state.model,
                'max_tokens': 2048,
                'system': '\n\n'.join(system_parts) or None,
                'messages': [*state.history_items, user_message],
                'metadata': {
                    'compaction_session_id': thread_id,
                    'session_id': thread_id,
                    'preferred_backend': preferred_backend,
                    'cwd': state.cwd,
                },
            }
        )
        response = await run_in_threadpool(self.service.invoke_from_anthropic, req)
        text = _response_text(response)
        item = {'id': f'item_{uuid.uuid4().hex}', 'type': 'agentMessage', 'text': text, 'phase': 'final_answer'}
        assistant_message = {'role': 'assistant', 'content': text}

        await websocket.send_json(_notification('item/started', {'threadId': thread_id, 'turnId': turn_id, 'item': item}))
        await websocket.send_json(_notification('item/completed', {'threadId': thread_id, 'turnId': turn_id, 'item': item}))

        state.history_items.extend([user_message, assistant_message])
        turn['status'] = 'completed'
        state.thread['status'] = {'type': 'idle'}
        state.thread['updatedAt'] = _now()
        self._save_thread_state(thread_id)

        await websocket.send_json(_notification('turn/completed', {'threadId': thread_id, 'turn': turn}))
        await websocket.send_json(_notification('thread/status/changed', {'threadId': thread_id, 'status': state.thread['status']}))
        await websocket.send_json(_jsonrpc_response(request_id, {'turn': turn}))

    async def _handle_thread_compact_start(self, websocket: WebSocket, request_id: Any, params: Dict[str, Any]) -> None:
        thread_id = params.get('threadId')
        state = self._get_thread_state(thread_id)
        if not state:
            await websocket.send_json(_jsonrpc_error(request_id, -32602, f'Unknown threadId: {thread_id}'))
            return

        item = {'id': f'item_{uuid.uuid4().hex}', 'type': 'contextCompaction'}
        turn_id = f'turn_{uuid.uuid4().hex}'
        items = self._items_for_compaction(state)
        current_request = self._latest_user_text(state.history_items, state.thread['preview'])
        repo_context = {'cwd': state.cwd}
        _record_transport_event(
            'thread_compact_start',
            thread_id=thread_id,
            mode=self.mode,
            items=len(items),
            cwd=state.cwd,
            **compaction_payload_fields(
                before={
                    'session_id': thread_id,
                    'items': items,
                    'current_request': current_request,
                    'repo_context': repo_context,
                    'refresh_if_needed': False,
                }
            ),
        )

        await websocket.send_json(_notification('item/started', {'threadId': thread_id, 'turnId': turn_id, 'item': item}))
        handoff = await run_in_threadpool(
            self.service.compact_session,
            thread_id,
            items,
            current_request=current_request,
            repo_context=repo_context,
            refresh_if_needed=False,
        )
        state.compacted_flow = self.service.build_compaction_handoff_flow(thread_id, current_request=current_request)
        state.history_items = list((state.compacted_flow or {}).get('recent_raw_turns') or [])
        self._save_thread_state(thread_id)
        _record_transport_event(
            'local_compaction_completed',
            thread_id=thread_id,
            handoff=bool(state.compacted_flow),
            recent_raw_turns=len(state.history_items),
            **compaction_payload_fields(after=handoff),
        )

        await websocket.send_json(_notification('item/completed', {'threadId': thread_id, 'turnId': turn_id, 'item': item}))
        await websocket.send_json(_notification('thread/compacted', {'threadId': thread_id, 'turnId': turn_id}))
        await websocket.send_json(_jsonrpc_response(request_id, {'threadId': thread_id, 'handoff': handoff}))

    def _items_for_compaction(self, state: ThreadState) -> List[Dict[str, Any]]:
        if not state.compacted_flow:
            return list(state.history_items)
        return [
            {
                'role': 'assistant',
                'content': _render_compacted_flow(
                    state.compacted_flow,
                    current_request=state.compacted_flow.get('current_request', ''),
                ),
            },
            *state.history_items,
        ]

    def _latest_user_text(self, items: List[Dict[str, Any]], fallback: str = '') -> str:
        for item in reversed(items):
            if item.get('role') == 'user' and item.get('content'):
                return str(item['content'])
        return fallback

    def _save_thread_state(self, thread_id: str) -> None:
        state = self.threads[thread_id]
        payload = {
            'thread': state.thread,
            'approval_policy': state.approval_policy,
            'cwd': state.cwd,
            'model': state.model,
            'model_provider': state.model_provider,
            'sandbox': state.sandbox,
            'base_instructions': state.base_instructions,
            'developer_instructions': state.developer_instructions,
            'turns': state.turns,
            'history_items': state.history_items,
            'compacted_flow': state.compacted_flow,
        }
        (self.state_dir / f'{thread_id}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
