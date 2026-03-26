from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .clients.codex_client import CodexCLIClient
from .clients.ollama_client import OllamaClient
from .compaction import CompactionService, render_codex_support_prompt, render_compacted_flow, render_inline_compaction_summary
from .config import settings
from .models import AnthropicMessagesRequest, InvokeRequest, InvokeResponse, RouteDecision
from .prompt_loader import load_prompt
from .task_metrics import estimate_tokens, extract_task_metrics
from .tool_adapter import (
    anthropic_messages_to_ollama,
    is_devstral_model,
    normalize_ollama_tools,
    ollama_message_to_anthropic_content,
    recover_ollama_message,
    recover_stream_ollama_message,
)

logger = logging.getLogger(__name__)


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    parts.append(item.get('text', ''))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return '\n'.join(part for part in parts if part)
    return str(content)


def anthropic_messages_to_prompt(req: AnthropicMessagesRequest) -> Dict[str, Any]:
    system = flatten_content(req.system) if req.system else ''
    flattened_messages = [{'role': msg.role, 'content': flatten_content(msg.content)} for msg in req.messages]
    lines: List[str] = []
    for msg in flattened_messages:
        lines.append(f"[{msg['role'].upper()}]\n{msg['content']}")
    full_prompt = '\n\n'.join(lines)
    latest_user = next((msg['content'] for msg in reversed(flattened_messages) if msg['role'] == 'user'), '')
    return {
        'system': system,
        'prompt': full_prompt,
        'user_prompt': latest_user or full_prompt,
        'trajectory': flattened_messages[:-1] if flattened_messages else [],
    }


_ROUTER_SYSTEM_PATH = Path(__file__).resolve().parent / 'prompts' / 'router_system.md'
_ROUTER_TASK = load_prompt('router_task.md')


def _load_router_system(path: Path = _ROUTER_SYSTEM_PATH) -> str:
    return path.read_text(encoding='utf-8').strip()


def _router_payload_tokens(payload: Dict[str, Any]) -> int:
    return estimate_tokens(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))


def build_routing_digest(
    system: str,
    prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
    history: Optional[Any] = None,
) -> Dict[str, Any]:
    metadata_payload = metadata or {}
    user_prompt = str(metadata_payload.get('router_user_prompt') or prompt)
    trajectory = metadata_payload.get('router_trajectory')
    if trajectory is None:
        trajectory = history if history is not None else []

    metrics = extract_task_metrics(prompt=user_prompt, trajectory=trajectory, metadata=metadata_payload)
    metrics['backend_request_tokens'] = estimate_tokens(f'{system}\n\n{prompt}'.strip())
    metrics['reasoner_context_limit'] = settings.reasoner_num_ctx
    metrics['coder_context_limit'] = settings.coder_num_ctx
    metrics['router_context_limit'] = settings.router_num_ctx
    metrics['router_payload_tokens'] = 0
    metrics['router_request_tokens'] = 0

    return {
        'task': _ROUTER_TASK,
        'available_routes': [],
        'user_prompt': user_prompt,
        'trajectory': trajectory,
        'metrics': metrics,
    }


def _codex_workdir(metadata: Optional[Dict[str, Any]]) -> str:
    if not isinstance(metadata, dict):
        return settings.codex_workdir
    for key in ('cwd', 'workdir', 'project_path', 'repo_path'):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    repo_context = metadata.get('repo_context')
    if isinstance(repo_context, dict):
        value = repo_context.get('cwd')
        if isinstance(value, str) and value:
            return value
    return settings.codex_workdir


class RoutingService:
    def __init__(self) -> None:
        self.router_client = OllamaClient(
            settings.router_ollama_base_url,
            (settings.ollama_connect_timeout_seconds, settings.router_timeout_seconds),
            pool_connections=settings.ollama_pool_connections,
            pool_maxsize=settings.ollama_pool_maxsize,
        )
        self.coder_client = OllamaClient(
            settings.coder_ollama_base_url,
            (settings.ollama_connect_timeout_seconds, settings.coder_timeout_seconds),
            pool_connections=settings.ollama_pool_connections,
            pool_maxsize=settings.ollama_pool_maxsize,
        )
        self.reasoner_client = OllamaClient(
            settings.reasoner_ollama_base_url,
            (settings.ollama_connect_timeout_seconds, settings.reasoner_timeout_seconds),
            pool_connections=settings.ollama_pool_connections,
            pool_maxsize=settings.ollama_pool_maxsize,
        )
        self.compactor_client = OllamaClient(
            settings.compactor_ollama_base_url,
            (settings.ollama_connect_timeout_seconds, settings.compactor_timeout_seconds),
            pool_connections=settings.ollama_pool_connections,
            pool_maxsize=settings.ollama_pool_maxsize,
        )
        self.codex_client = (
            CodexCLIClient(
                settings.codex_cmd,
                settings.codex_workdir,
                settings.codex_timeout_seconds,
                model_provider=settings.codex_exec_model_provider,
                model=settings.codex_exec_model,
            )
            if settings.enable_codex_cli
            else None
        )
        self.compaction_service = CompactionService()

    def route(self, system: str, prompt: str, metadata: Optional[Dict[str, Any]] = None, preferred_backend: Optional[str] = None) -> RouteDecision:
        history = metadata.get('history') if isinstance(metadata, dict) else None if metadata else None
        metrics_payload = build_routing_digest(system, prompt, metadata, history)

        if preferred_backend:
            return RouteDecision(route=preferred_backend, confidence=1.0, reason='preferred backend override', metrics_payload=metrics_payload)

        available = []
        if settings.enable_local_coder:
            available.append('local_coder')
        if settings.enable_local_reasoner:
            available.append('local_reasoner')
        if settings.enable_codex_cli and self.codex_client:
            available.append('codex_cli')

        if not available:
            return RouteDecision(route='local_reasoner', confidence=0.0, reason='no configured backends', metrics_payload=metrics_payload)

        metrics = metrics_payload['metrics']
        eligible = list(available)

        if metrics['backend_request_tokens'] > settings.reasoner_num_ctx and 'local_reasoner' in eligible:
            eligible.remove('local_reasoner')
        if metrics['backend_request_tokens'] > settings.coder_num_ctx and 'local_coder' in eligible:
            eligible.remove('local_coder')

        metrics_payload['available_routes'] = eligible
        payload_tokens = _router_payload_tokens(metrics_payload)
        metrics['router_payload_tokens'] = payload_tokens
        metrics['router_request_tokens'] = payload_tokens

        if metrics['router_request_tokens'] > settings.router_num_ctx:
            route = 'codex_cli' if 'codex_cli' in available else self._fallback_route(eligible or available)
            return RouteDecision(
                route=route,
                confidence=1.0,
                reason='router request exceeds router context window',
                metrics_payload=metrics_payload,
            )

        if not eligible:
            route = 'codex_cli' if 'codex_cli' in available else self._fallback_route(available)
            return RouteDecision(route=route, confidence=1.0, reason='local context windows exceeded', metrics_payload=metrics_payload)

        if len(eligible) == 1:
            return RouteDecision(route=eligible[0], confidence=1.0, reason='only one route fits context limits', metrics_payload=metrics_payload)

        raw = self.router_client.chat(
            settings.router_model,
            [{'role': 'user', 'content': json.dumps(metrics_payload, ensure_ascii=False)}],
            temperature=settings.router_temperature,
            num_ctx=settings.router_num_ctx,
            system=_load_router_system(),
            response_format='json',
            think=False,
        )
        content = raw.get('message', {}).get('content', '')
        try:
            parsed = json.loads(content)
            route = parsed.get('route', 'codex_cli')
            if route not in eligible:
                route = self._fallback_route(eligible)
            return RouteDecision(
                route=route,
                confidence=float(parsed.get('confidence', 0.0) or 0.0),
                reason=str(parsed.get('reason', '')),
                metrics_payload=metrics_payload,
            )
        except Exception:
            route = self._fallback_route(eligible)
            return RouteDecision(route=route, confidence=0.0, reason='router JSON parse fallback', metrics_payload=metrics_payload)

    def _fallback_route(self, available: List[str]) -> str:
        if 'local_coder' in available:
            return 'local_coder'
        if 'local_reasoner' in available:
            return 'local_reasoner'
        if 'codex_cli' in available:
            return 'codex_cli'
        return available[0] if available else 'local_reasoner'

    def invoke(self, req: InvokeRequest) -> InvokeResponse:
        decision = self.route(req.system or '', req.prompt, req.metadata, req.preferred_backend)
        backend_model, output_text, thinking, raw = self._dispatch(decision.route, req.system or '', req.prompt, metadata=req.metadata)
        return InvokeResponse(route_decision=decision, backend_model=backend_model, output_text=output_text, thinking=thinking, raw=raw)

    def compact_session(
        self,
        session_id: str,
        items: List[Dict[str, Any]],
        *,
        current_request: str,
        repo_context: Optional[Dict[str, Any]] = None,
        refresh_if_needed: bool = False,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        if refresh_if_needed:
            handoff = self.compaction_service.refresh_if_needed(
                session_id,
                items,
                current_request=current_request,
                repo_context=repo_context,
                progress_callback=progress_callback,
            )
        else:
            handoff = self.compaction_service.compact_transcript(
                session_id,
                items,
                current_request=current_request,
                repo_context=repo_context,
                progress_callback=progress_callback,
            )
        return handoff.model_dump()

    def load_compaction_handoff(self, session_id: str) -> Optional[Dict[str, Any]]:
        handoff = self.compaction_service.load_latest_handoff(session_id)
        return handoff.model_dump() if handoff is not None else None

    def build_compaction_handoff_flow(self, session_id: str, *, current_request: Optional[str] = None) -> Optional[Dict[str, Any]]:
        flow = self.compaction_service.build_codex_handoff_flow(session_id, current_request=current_request)
        return flow.model_dump() if flow is not None else None

    def invoke_inline_compact_from_anthropic(
        self,
        req: AnthropicMessagesRequest,
        *,
        progress_callback: Optional[Callable[..., None]] = None,
        response_model: Optional[str] = None,
        raw_backend_mode: str = 'chunked_durable_compaction',
    ) -> Dict[str, Any]:
        text, raw_backend = self._inline_compaction_result(
            req,
            progress_callback=progress_callback,
            raw_backend_mode=raw_backend_mode,
        )
        return {
            'id': 'msg_router_local',
            'type': 'message',
            'role': 'assistant',
            'model': response_model or self._inline_compaction_response_model(),
            'content': [{'type': 'text', 'text': text}],
            'stop_reason': 'end_turn',
            'stop_sequence': None,
            'usage': {
                'input_tokens': raw_backend['usage']['input_tokens'],
                'output_tokens': raw_backend['usage']['output_tokens'],
            },
            'request_metadata': req.metadata or {},
            'raw_backend': raw_backend,
        }

    def stream_inline_compact_from_anthropic(
        self,
        req: AnthropicMessagesRequest,
        *,
        progress_callback: Optional[Callable[..., None]] = None,
        response_model: Optional[str] = None,
        raw_backend_mode: str = 'chunked_durable_compaction',
    ) -> Iterator[Dict[str, Any]]:
        full_text, final_raw = self._inline_compaction_result(
            req,
            progress_callback=progress_callback,
            raw_backend_mode=raw_backend_mode,
        )
        if full_text:
            yield {'type': 'text_delta', 'delta': full_text}
        yield {
            'type': 'final',
            'response': {
                'id': 'msg_router_local',
                'type': 'message',
                'role': 'assistant',
                'model': response_model or self._inline_compaction_response_model(),
                'content': [{'type': 'text', 'text': full_text}],
                'stop_reason': 'end_turn',
                'stop_sequence': None,
                'usage': final_raw['usage'],
                'request_metadata': req.metadata or {},
                'raw_backend': final_raw,
            },
        }

    def _inline_compaction_result(
        self,
        req: AnthropicMessagesRequest,
        *,
        progress_callback: Optional[Callable[..., None]] = None,
        raw_backend_mode: str = 'chunked_durable_compaction',
    ) -> tuple[str, Dict[str, Any]]:
        transcript_items, current_request = _inline_compaction_inputs(req)
        session_id = _inline_compaction_session_id(req, transcript_items, current_request)
        repo_context = _inline_compaction_repo_context(req.metadata)
        handoff = self.compact_session(
            session_id,
            transcript_items,
            current_request=current_request,
            repo_context=repo_context,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback('inline_handoff_loaded', session_id=session_id)
        flow = self.build_compaction_handoff_flow(session_id, current_request=current_request)
        if flow is None:
            raise RuntimeError(f'Inline compaction failed to build compacted flow for session {session_id}')
        rendered = render_inline_compaction_summary(flow, current_request=current_request)
        if progress_callback is not None:
            progress_callback('inline_render_completed', session_id=session_id)
        raw_backend = {
            'mode': raw_backend_mode,
            'session_id': session_id,
            'current_request': current_request,
            'handoff': handoff,
            'compacted_flow': flow,
            'machine_compacted_flow': render_compacted_flow(flow, current_request=current_request),
            'usage': {
                'input_tokens': estimate_tokens(transcript_items),
                'output_tokens': estimate_tokens(rendered),
            },
        }
        return rendered, raw_backend

    def _inline_compaction_response_model(self) -> str:
        extractor = getattr(getattr(self, 'compaction_service', None), 'extractor', None)
        model = getattr(extractor, 'model', None)
        if isinstance(model, str) and model:
            return model
        return settings.compactor_model

    def invoke_from_anthropic(self, req: AnthropicMessagesRequest) -> Dict[str, Any]:
        transformed = anthropic_messages_to_prompt(req)
        metadata = dict(req.metadata or {})
        metadata['router_user_prompt'] = transformed['user_prompt']
        metadata['router_trajectory'] = transformed['trajectory']
        decision = self.route(
            transformed['system'],
            transformed['prompt'],
            metadata,
            preferred_backend=metadata.get('preferred_backend'),
        )
        structured_coder = decision.route == 'local_coder' and bool(req.tools)
        devstral_coder = decision.route == 'local_coder' and is_devstral_model(settings.coder_model)
        backend_messages = anthropic_messages_to_ollama(req.messages) if structured_coder else None
        backend_tools = normalize_ollama_tools(req.tools) if structured_coder else None
        backend_model, output_text, thinking, raw = self._dispatch(
            decision.route,
            transformed['system'],
            transformed['prompt'],
            max_tokens=req.max_tokens,
            messages=backend_messages,
            tools=backend_tools,
            metadata=metadata,
        )
        message = raw.get('message', {})
        content = ollama_message_to_anthropic_content(message) if structured_coder else [{'type': 'text', 'text': output_text}]
        stop_reason = 'tool_use' if any(block.get('type') == 'tool_use' for block in content) else 'end_turn'
        return {
            'id': 'msg_router_local',
            'type': 'message',
            'role': 'assistant',
            'model': backend_model,
            'content': content,
            'stop_reason': stop_reason,
            'stop_sequence': None,
            'usage': raw.get('usage', {'input_tokens': 0, 'output_tokens': 0}),
            'route_decision': decision.model_dump(),
            'request_metadata': req.metadata or {},
            'thinking': thinking,
            'raw_backend': raw,
        }

    def stream_from_anthropic(self, req: AnthropicMessagesRequest) -> Iterator[Dict[str, Any]]:
        transformed = anthropic_messages_to_prompt(req)
        metadata = dict(req.metadata or {})
        metadata['router_user_prompt'] = transformed['user_prompt']
        metadata['router_trajectory'] = transformed['trajectory']
        decision = self.route(
            transformed['system'],
            transformed['prompt'],
            metadata,
            preferred_backend=metadata.get('preferred_backend'),
        )
        structured_coder = decision.route == 'local_coder' and bool(req.tools)
        devstral_coder = decision.route == 'local_coder' and is_devstral_model(settings.coder_model)
        backend_messages = anthropic_messages_to_ollama(req.messages) if structured_coder else None
        backend_tools = normalize_ollama_tools(req.tools) if structured_coder else None

        if decision.route == 'local_coder':
            yield from self._stream_ollama_route(
                decision,
                self.coder_client,
                settings.coder_model,
                settings.coder_temperature,
                settings.coder_num_ctx,
                req.max_tokens,
                transformed['system'] or load_prompt('coder_system.md'),
                backend_messages or [{'role': 'user', 'content': transformed['prompt']}],
                tools=backend_tools,
                structured=structured_coder,
                devstral=devstral_coder,
                request_metadata=req.metadata or {},
            )
            return

        if decision.route == 'local_reasoner':
            yield from self._stream_ollama_route(
                decision,
                self.reasoner_client,
                settings.reasoner_model,
                settings.reasoner_temperature,
                settings.reasoner_num_ctx,
                req.max_tokens,
                transformed['system'] or load_prompt('reasoner_system.md'),
                backend_messages or [{'role': 'user', 'content': transformed['prompt']}],
                tools=None,
                structured=False,
                devstral=False,
                request_metadata=req.metadata or {},
            )
            return

        yield {'type': 'final', 'response': self.invoke_from_anthropic(req)}

    def _stream_ollama_route(
        self,
        decision: RouteDecision,
        client: OllamaClient,
        model: str,
        temperature: float,
        num_ctx: int,
        max_tokens: int,
        system: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        structured: bool,
        devstral: bool,
        request_metadata: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]:
        full_text = ''
        visible_text = ''
        final_raw: Dict[str, Any] = {}
        streamed_tool_calls: List[Dict[str, Any]] = []
        seen_tool_call_signatures: set[str] = set()
        for raw in client.chat_stream(
            model,
            messages,
            temperature=temperature,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
        ):
            final_raw = raw
            raw_message = raw.get('message', {})
            raw_delta = raw_message.get('content', '')
            recovered_tool_calls: List[Dict[str, Any]] = []
            if raw_delta:
                full_text += raw_delta
                if not structured:
                    yield {'type': 'text_delta', 'delta': raw_delta}
            chunk_message = recover_stream_ollama_message(raw_message)
            if structured:
                recovered = recover_stream_ollama_message({'content': full_text})
                recovered_text = recovered.get('content', '')
                recovered_tool_calls = _normalize_stream_tool_calls(recovered.get('tool_calls'))
                if recovered_text.startswith(visible_text):
                    safe_delta = recovered_text[len(visible_text):]
                else:
                    safe_delta = ''
                visible_text = recovered_text
                if safe_delta:
                    yield {'type': 'text_delta', 'delta': safe_delta}
            candidate_tool_calls = _normalize_stream_tool_calls(chunk_message.get('tool_calls'))
            if recovered_tool_calls:
                candidate_tool_calls.extend(recovered_tool_calls)
            for tool_call in candidate_tool_calls:
                signature = json.dumps(
                    {
                        'name': tool_call['function']['name'],
                        'arguments': tool_call['function']['arguments'],
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                if signature in seen_tool_call_signatures:
                    continue
                seen_tool_call_signatures.add(signature)
                tool_call = dict(tool_call)
                tool_call['id'] = tool_call.get('id') or f"call_{len(streamed_tool_calls) + 1}"
                streamed_tool_calls.append(tool_call)
                yield {'type': 'tool_calls', 'tool_calls': [tool_call]}

        final_message = dict(final_raw.get('message') or {})
        final_message['content'] = full_text
        if devstral:
            final_message = recover_ollama_message(final_message)
        elif structured:
            final_message = recover_ollama_message(final_message)
        if streamed_tool_calls:
            final_message['tool_calls'] = streamed_tool_calls
        final_raw['message'] = final_message
        final_raw['usage'] = {
            'input_tokens': final_raw.get('prompt_eval_count', 0),
            'output_tokens': final_raw.get('eval_count', 0),
        }
        content = ollama_message_to_anthropic_content(final_message) if structured else [{'type': 'text', 'text': full_text}]
        stop_reason = 'tool_use' if any(block.get('type') == 'tool_use' for block in content) else 'end_turn'
        yield {
            'type': 'final',
            'response': {
                'id': 'msg_router_local',
                'type': 'message',
                'role': 'assistant',
                'model': model,
                'content': content,
                'stop_reason': stop_reason,
                'stop_sequence': None,
                'usage': final_raw['usage'],
                'route_decision': decision.model_dump(),
                'request_metadata': request_metadata,
                'thinking': final_message.get('thinking'),
                'raw_backend': final_raw,
            },
        }

    def _dispatch(
        self,
        route: str,
        system: str,
        prompt: str,
        max_tokens: int = 2048,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str, Optional[str], Dict[str, Any]]:
        messages = messages or [{'role': 'user', 'content': prompt}]

        if route == 'local_coder':
            raw = self.coder_client.chat(
                settings.coder_model,
                messages,
                temperature=settings.coder_temperature,
                num_ctx=settings.coder_num_ctx,
                max_tokens=max_tokens,
                system=system or load_prompt('coder_system.md'),
                tools=tools,
            )
            if is_devstral_model(settings.coder_model):
                raw['message'] = recover_ollama_message(raw.get('message', {}))
            return settings.coder_model, raw.get('message', {}).get('content', ''), raw.get('message', {}).get('thinking'), raw

        if route == 'local_reasoner':
            raw = self.reasoner_client.chat(
                settings.reasoner_model,
                messages,
                temperature=settings.reasoner_temperature,
                num_ctx=settings.reasoner_num_ctx,
                max_tokens=max_tokens,
                system=system or load_prompt('reasoner_system.md'),
            )
            return settings.reasoner_model, raw.get('message', {}).get('content', ''), raw.get('message', {}).get('thinking'), raw

        if route == 'codex_cli':
            if not self.codex_client:
                raise RuntimeError('Codex CLI backend is not configured')
            codex_prompt = self._build_codex_cli_prompt(system, prompt, metadata or {})
            raw = self.codex_client.exec_prompt(codex_prompt, workdir=_codex_workdir(metadata))
            output = raw.get('stdout', '') or raw.get('stderr', '')
            return 'codex-cli', output, None, raw

        raise RuntimeError(f'Unknown route: {route}')

    def _build_codex_cli_prompt(self, system: str, prompt: str, metadata: Dict[str, Any]) -> str:
        session_id = metadata.get('compaction_session_id') or metadata.get('session_id')
        current_request = str(metadata.get('router_user_prompt') or prompt)
        if session_id:
            try:
                flow = self.compaction_service.build_codex_handoff_flow(session_id, current_request=current_request)
                if flow is not None:
                    return render_codex_support_prompt(flow, system=system, current_request=current_request)
            except Exception:
                logger.exception('failed to render compaction handoff for codex session %s', session_id)
        return f'{system}\n\n{prompt}'.strip()


def _normalize_stream_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get('function') if isinstance(tool_call.get('function'), dict) else tool_call
        if not isinstance(function, dict):
            continue
        arguments = function.get('arguments', {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {'raw': arguments}
        if not isinstance(arguments, dict):
            arguments = {'value': arguments}
        name = function.get('name', '')
        if not name:
            continue
        normalized.append(
            {
                'id': tool_call.get('id'),
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': arguments,
                },
            }
        )
    return normalized


def _inline_compaction_inputs(req: AnthropicMessagesRequest) -> tuple[List[Dict[str, Any]], str]:
    trigger_index: Optional[int] = None
    for index in range(len(req.messages) - 1, -1, -1):
        message = req.messages[index]
        text = _content_text(message.content).strip()
        if message.role == 'user' and settings.inline_compact_sentinel in text:
            trigger_index = index
            break

    transcript_items: List[Dict[str, Any]] = []
    latest_real_user = ''
    fallback_request = ''
    for index, message in enumerate(req.messages):
        content = deepcopy(message.content)
        if message.role == 'user' and index == trigger_index:
            content = _strip_compaction_sentinel_from_content(content)
            fallback_request = _content_text(content).strip()
            continue
        if _content_is_empty(content):
            continue
        if message.role == 'user':
            text = _content_text(content).strip()
            if text:
                latest_real_user = text
        transcript_items.append({'role': message.role, 'content': content})

    current_request = latest_real_user or fallback_request
    return transcript_items, current_request


def _inline_compaction_session_id(
    req: AnthropicMessagesRequest,
    transcript_items: List[Dict[str, Any]],
    current_request: str,
) -> str:
    metadata = req.metadata or {}
    session_id = metadata.get('compaction_session_id') or metadata.get('session_id')
    if isinstance(session_id, str) and session_id:
        return session_id
    digest = hashlib.sha256(
        json.dumps(
            {
                'messages': transcript_items,
                'current_request': current_request,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode('utf-8')
    ).hexdigest()[:24]
    return f'inline_{digest}'


def _inline_compaction_repo_context(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cwd = _codex_workdir(metadata)
    return {'cwd': cwd} if cwd else {}


def _strip_compaction_sentinel(text: str) -> str:
    sentinel = settings.inline_compact_sentinel
    if sentinel and sentinel in text:
        return text.replace(sentinel, '').strip()
    return text


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get('type') in {'text', 'input_text', 'output_text'}:
            text = item.get('text') or item.get('input_text') or item.get('output_text') or ''
            if text:
                parts.append(text)
    return '\n'.join(parts)


def _content_is_empty(content: Any) -> bool:
    if isinstance(content, str):
        return not content.strip()
    if not isinstance(content, list):
        return content is None
    return not any(isinstance(item, dict) for item in content)


def _strip_compaction_sentinel_from_content(content: Any) -> Any:
    if isinstance(content, str):
        return _strip_compaction_sentinel(content)
    if not isinstance(content, list):
        return content

    stripped: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        if updated.get('type') in {'text', 'input_text', 'output_text'}:
            text = updated.get('text') or updated.get('input_text') or updated.get('output_text') or ''
            text = _strip_compaction_sentinel(text)
            if not text:
                continue
            if 'text' in updated:
                updated['text'] = text
            elif 'input_text' in updated:
                updated['input_text'] = text
            elif 'output_text' in updated:
                updated['output_text'] = text
        stripped.append(updated)
    return stripped
