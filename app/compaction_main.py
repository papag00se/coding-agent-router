from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .clients.responses_client import ResponsesClient
from .compaction import CompactionService
from .compaction.extractor import CompactionExtractor
from .compaction.normalize import sanitize_inline_compaction_payload
from .compaction.refiner import CompactionRefiner
from .compaction_transport import compaction_payload_fields, record_transport_event
from .compat import (
    anthropic_request_from_responses,
    iter_responses_progress,
    ollama_tags_response,
    ollama_version_response,
    openai_models_response,
    responses_response,
)
from .config import settings
from .inline_compaction_jobs import InlineCompactionJobManager, inline_compaction_request_key
from .models import CompactRequest, InvokeRequest
from .router import RoutingService
from .task_metrics import estimate_openai_tokens, estimate_tokens
from .transport_metrics import transport_metrics_snapshot

app = FastAPI(title="Local Agent Router Compaction Service", version="0.1.0")
service = RoutingService()
inline_compaction_jobs = InlineCompactionJobManager(service)
_TRANSPORT_LOG_PATH = Path('state/compaction_transport.jsonl')
_UPSTREAM = requests.Session()
_SPARK_MAX_REQUEST_TOKENS = 114_688
_MINI_MAX_FILE_REFS = 10
_IMAGE_BLOCK_TYPES = {'image', 'input_image', 'localImage', 'local_image'}
_SPARK_SUPPORTED_REASONING_EFFORTS = {'low', 'medium', 'high', 'xhigh'}
_SHELL_WRAPPER = re.compile(r"^(?:/bin/)?bash\s+-lc\s+(['\"])(?P<inner>.*)\1$", re.DOTALL)
_FILE_REF_RE = re.compile(r"(?:(?:\.\.?/|/)?[\w.-]+/)+[\w.-]+\.[A-Za-z0-9_+-]{1,10}")
_STACKTRACE_RE = re.compile(
    r"(?:Traceback \(most recent call last\)|AssertionError|TypeError|ValueError|RuntimeError|SyntaxError|ReferenceError|Error:|FAILED\b|E\s+.+)",
    re.IGNORECASE,
)


def _record_transport_event(event: str, **fields: Any) -> None:
    record_transport_event(_TRANSPORT_LOG_PATH, event, **fields)


def _configured_compactor_model() -> str:
    return str(getattr(settings, 'compactor_model', None) or 'unknown')


def _inline_compaction_local_model() -> str:
    resolver = getattr(service, '_inline_compaction_response_model', None)
    if callable(resolver):
        try:
            resolved = resolver()
        except Exception:
            resolved = None
        if resolved:
            return str(resolved)
    return _configured_compactor_model()


def _responses_request_tokens(payload: Dict[str, Any]) -> int:
    model = str(payload.get('model') or 'gpt-5.4')
    return max(0, estimate_openai_tokens(payload, model=model))


def _compact_request_tokens(req: CompactRequest) -> int:
    return max(
        0,
        estimate_tokens(
            {
                'session_id': req.session_id,
                'items': req.items,
                'current_request': req.current_request,
                'repo_context': req.repo_context,
                'refresh_if_needed': req.refresh_if_needed,
            }
        ),
    )


def _log_inline_compaction_stream(events: Iterable[Dict[str, Any]], payload: Dict[str, Any], *, request_key: str) -> Iterable[Dict[str, Any]]:
    request_tokens = _responses_request_tokens(payload)
    local_model = _inline_compaction_local_model()
    for event in events:
        if event.get('type') == 'final':
            response = event.get('response')
            _record_transport_event(
                'inline_compaction_completed',
                path='/v1/responses',
                request_key=request_key,
                stream=True,
                request_tokens=request_tokens,
                **compaction_payload_fields(after=response),
            )
        if event.get('type') == 'failed':
            _record_transport_event(
                'inline_compaction_failed',
                path='/v1/responses',
                request_key=request_key,
                stream=True,
                request_tokens=request_tokens,
                local_model=local_model,
                error=str(event.get('message') or 'inline compaction failed'),
            )
        yield event


def _unsupported_transport_response(transport: str) -> JSONResponse:
    _record_transport_event(
        'unsupported_transport',
        transport=transport,
        path={
            'v1_messages': '/v1/messages',
            'chat_completions': '/v1/chat/completions',
            'ollama_chat': '/api/chat',
        }.get(transport),
    )
    return JSONResponse(
        {'error': 'Unsupported transport for the compaction companion service.'},
        status_code=409,
    )


def _contains_sentinel(value: Any) -> bool:
    sentinel = settings.inline_compact_sentinel
    if not sentinel:
        return False
    if isinstance(value, str):
        return sentinel in value
    if isinstance(value, list):
        return any(_contains_sentinel(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_sentinel(item) for item in value.values())
    return False


def _iter_message_text(content: Any) -> Iterable[str]:
    if isinstance(content, str):
        if content:
            yield content
        return
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get('type') in {'input_text', 'output_text', 'text'}:
            text = part.get('text')
            if isinstance(text, str) and text:
                yield text


def _current_turn_contains_sentinel(input_items: Any) -> bool:
    if isinstance(input_items, str):
        return _contains_sentinel(input_items)
    if not isinstance(input_items, list):
        return False
    for item in reversed(input_items):
        if not isinstance(item, dict) or item.get('type') != 'message':
            continue
        if item.get('role') != 'user':
            continue
        return any(_contains_sentinel(text) for text in _iter_message_text(item.get('content')))
    return False


def _is_inline_compaction(payload: Dict[str, Any]) -> bool:
    return _contains_sentinel(payload.get('instructions')) or _current_turn_contains_sentinel(payload.get('input'))


def _proxy_headers(request: Request) -> Dict[str, str]:
    blocked = {'host', 'content-length', 'connection'}
    return {key: value for key, value in request.headers.items() if key.lower() not in blocked}


def _proxy_url(path: str) -> str:
    return f"{settings.openai_passthrough_base_url.rstrip('/')}{path}"


def _auth_header_kind(headers: Dict[str, str]) -> str:
    value = headers.get('authorization') or headers.get('Authorization')
    if not value:
        return 'none'
    if value.startswith('Bearer '):
        return 'bearer'
    return 'other'


def _payload_contains_image_input(payload: Dict[str, Any]) -> bool:
    def _walk(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get('type') in _IMAGE_BLOCK_TYPES:
                return True
            if value.get('image_url'):
                return True
            return any(_walk(child) for child in value.values())
        if isinstance(value, list):
            return any(_walk(child) for child in value)
        return False

    return _walk(payload.get('input'))


def _build_spark_chunked_compaction_service(headers: Dict[str, str]) -> CompactionService:
    client = ResponsesClient(
        settings.openai_passthrough_base_url,
        (settings.ollama_connect_timeout_seconds, settings.codex_timeout_seconds),
        headers=headers,
        session=_UPSTREAM,
    )
    shared_storage = getattr(getattr(service, 'compaction_service', None), 'storage', None)
    return CompactionService(
        extractor=CompactionExtractor(
            client=client,
            model=settings.codex_spark_model,
            temperature=settings.compactor_temperature,
            num_ctx=settings.compactor_num_ctx,
        ),
        refiner=CompactionRefiner(
            client=client,
            model=settings.codex_spark_model,
            temperature=settings.compactor_temperature,
            num_ctx=settings.compactor_num_ctx,
        ),
        storage=shared_storage,
    )


def _stream_chunked_inline_compaction_with_service(
    req: Any,
    compaction_service: CompactionService,
    *,
    progress_callback=None,
):
    fallback_router = RoutingService.__new__(RoutingService)
    fallback_router.compaction_service = compaction_service
    yield from RoutingService.stream_inline_compact_from_anthropic(
        fallback_router,
        req,
        progress_callback=progress_callback,
        response_model=settings.codex_spark_model,
        raw_backend_mode='spark_chunked_durable_compaction',
    )


def _stream_spark_inline_compaction_fallback(
    payload: Dict[str, Any],
    _request: Any,
    headers: Dict[str, str],
    cause: Exception,
    *,
    request_key: str,
    progress_callback=None,
):
    if progress_callback is not None:
        progress_callback('spark_fallback_started', model=settings.codex_spark_model, error=str(cause))
    _record_transport_event(
        'inline_compaction_spark_fallback',
        path='/v1/responses',
        request_key=request_key,
        upstream_url=_proxy_url('/responses'),
        auth_header=_auth_header_kind(headers),
        original_model=payload.get('model'),
        upstream_model=settings.codex_spark_model,
        request_tokens=_responses_request_tokens(payload),
        error=str(cause),
        mode='chunked',
    )
    spark_service = _build_spark_chunked_compaction_service(headers)
    yield from _stream_chunked_inline_compaction_with_service(
        _request,
        spark_service,
        progress_callback=progress_callback,
    )


inline_compaction_jobs.fallback_callback = _stream_spark_inline_compaction_fallback


def _rewrite_passthrough_payload_for_model(payload: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    requested_model = payload.get('model')
    original_model = str(requested_model or 'gpt-5.4')
    request_tokens = estimate_openai_tokens(payload, model=original_model)
    requested_gpt_54 = requested_model == 'gpt-5.4'
    requested_spark_model = requested_model == settings.codex_spark_model
    image_inputs_present = _payload_contains_image_input(payload)
    spark_category = _qualifying_spark_category(payload)
    spark_eligible = (
        requested_gpt_54
        and not image_inputs_present
        and request_tokens <= _SPARK_MAX_REQUEST_TOKENS
        and spark_category is not None
    )
    sample_value = _stable_sample_value(payload)
    mini_category = _qualifying_mini_category(payload, request_tokens=request_tokens)
    mini_eligible = requested_gpt_54 and mini_category is not None
    image_spark_fallback_applied = requested_spark_model and image_inputs_present
    mini_applied = mini_eligible or image_spark_fallback_applied
    spark_applied = spark_eligible and not mini_applied and sample_value < settings.codex_spark_qualified_rate

    rewrite_family: str | None = None
    target_model = requested_model
    rewritten = payload
    if spark_applied:
        rewrite_family = 'spark'
        target_model = settings.codex_spark_model
        rewritten = _normalize_payload_for_target_model(payload, target_model=target_model)
        rewritten['model'] = target_model
    elif mini_applied:
        rewrite_family = 'mini'
        target_model = settings.codex_mini_model
        rewritten = dict(payload)
        rewritten['model'] = target_model

    return rewritten, {
        'applied': rewrite_family is not None,
        'rewrite_family': rewrite_family,
        'target_model': target_model,
        'request_tokens': request_tokens,
        'image_inputs_present': image_inputs_present,
        'spark': {
            'applied': spark_applied,
            'eligible': spark_eligible,
            'category': spark_category,
            'sample_value': sample_value,
        },
        'mini': {
            'applied': mini_applied,
            'eligible': mini_eligible,
            'category': mini_category,
            'image_spark_fallback_applied': image_spark_fallback_applied,
        },
    }


def _normalize_payload_for_target_model(payload: Dict[str, Any], *, target_model: str) -> Dict[str, Any]:
    normalized = deepcopy(payload)
    if target_model != settings.codex_spark_model:
        return normalized
    _strip_message_phases(normalized.get('input'))
    _normalize_reasoning_for_spark(normalized)
    return normalized


def _strip_message_phases(input_items: Any) -> None:
    if not isinstance(input_items, list):
        return
    for item in input_items:
        if isinstance(item, dict) and item.get('type') == 'message':
            item.pop('phase', None)


def _normalize_reasoning_for_spark(payload: Dict[str, Any]) -> None:
    reasoning = payload.get('reasoning')
    if not isinstance(reasoning, dict):
        return
    effort = reasoning.get('effort')
    normalized_effort = _normalize_spark_reasoning_effort(effort)
    if normalized_effort is None:
        reasoning.pop('effort', None)
    else:
        reasoning['effort'] = normalized_effort
    if not reasoning:
        payload.pop('reasoning', None)


def _normalize_spark_reasoning_effort(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in _SPARK_SUPPORTED_REASONING_EFFORTS:
        return normalized
    return 'low'


def _qualifying_spark_category(payload: Dict[str, Any]) -> str | None:
    context = _passthrough_context(payload)
    latest_item = context['latest_item']
    if isinstance(latest_item, dict) and latest_item.get('type') == 'function_call_output':
        direct_category = _direct_spark_category(context['latest_output_command'], context['latest_output_text'])
        if direct_category is not None:
            return direct_category
    if _is_diff_followup(context):
        return 'diff_followup'
    if _is_stacktrace_triage(context):
        return 'stacktrace_triage'
    if _is_test_fix_loop(context):
        return 'test_fix_loop'
    if _is_small_refactor(context):
        return 'small_refactor'
    if _is_localized_edit(context):
        return 'localized_edit'
    if _is_simple_synthesis(context):
        return 'simple_synthesis'
    return _direct_spark_category(context['latest_output_command'], context['latest_output_text'])


def _qualifying_mini_category(payload: Dict[str, Any], *, request_tokens: int | None = None) -> str | None:
    if payload.get('model') != 'gpt-5.4':
        return None
    if request_tokens is None:
        request_tokens = estimate_openai_tokens(payload, model='gpt-5.4')
    if request_tokens > _SPARK_MAX_REQUEST_TOKENS:
        return 'oversize_for_spark'

    context = _passthrough_context(payload)
    if _is_bounded_investigation(context):
        return 'bounded_investigation'
    if _is_cross_file_analysis(context):
        return 'cross_file_analysis'
    if _is_medium_diff_followup(context):
        return 'medium_diff_followup'
    if _is_medium_test_triage(context):
        return 'medium_test_triage'
    if _is_multi_file_refactor(context):
        return 'multi_file_refactor'
    return None


def _direct_spark_category(command: str, output: str) -> str | None:
    if not command:
        return None
    if _is_file_read_command(command):
        return 'file_read'
    if _is_search_inventory_command(command):
        return 'search_inventory'
    if _is_targeted_test_command(command):
        return 'targeted_test'
    if _is_polling_command(command, output):
        return 'polling'
    return None


def _latest_significant_input_item(input_items: Any) -> Dict[str, Any] | None:
    if not isinstance(input_items, list):
        return None
    for item in reversed(input_items):
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'reasoning':
            continue
        return item
    return None


def _recent_non_reasoning_items(input_items: Any, limit: int = 12) -> list[Dict[str, Any]]:
    if not isinstance(input_items, list):
        return []
    items: list[Dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict) or item.get('type') == 'reasoning':
            continue
        items.append(item)
    if len(items) <= limit:
        return items
    return items[-limit:]


def _latest_message_text(items: list[Dict[str, Any]], role: str) -> str:
    for item in reversed(items):
        if item.get('type') != 'message' or item.get('role') != role:
            continue
        return "\n".join(_iter_message_text(item.get('content'))).strip()
    return ''


def _passthrough_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = _recent_non_reasoning_items(payload.get('input'))
    latest_item = items[-1] if items else None
    output_items = [item for item in items if item.get('type') == 'function_call_output']
    latest_output_item = output_items[-1] if output_items else None
    latest_output_text = ''
    latest_output_command = ''
    if latest_output_item is not None:
        output = latest_output_item.get('output')
        if isinstance(output, str):
            latest_output_text = output
            latest_output_command = _extract_command_from_tool_output(output)
    recent_output_text = "\n".join(
        item.get('output', '') for item in output_items[-6:] if isinstance(item.get('output'), str)
    )
    recent_commands = []
    for item in output_items[-6:]:
        output = item.get('output')
        if not isinstance(output, str):
            continue
        command = _extract_command_from_tool_output(output)
        if command:
            recent_commands.append(command)
    latest_user_text = _latest_message_text(items, 'user')
    latest_assistant_text = _latest_message_text(items, 'assistant')
    recent_text = "\n".join(
        part
        for part in [latest_user_text, latest_assistant_text, recent_output_text]
        if part
    )
    file_refs = sorted(_extract_file_refs(recent_text + "\n" + "\n".join(recent_commands)))
    return {
        'latest_item': latest_item,
        'latest_output_item': latest_output_item,
        'latest_output_text': latest_output_text,
        'latest_output_command': latest_output_command,
        'recent_output_text': recent_output_text,
        'recent_commands': recent_commands,
        'latest_user_text': latest_user_text,
        'latest_assistant_text': latest_assistant_text,
        'latest_text': "\n".join(part for part in [latest_user_text, latest_assistant_text] if part),
        'file_refs': file_refs,
        'small_scope': 0 < len(file_refs) <= 3,
        'medium_scope': 0 < len(file_refs) <= _MINI_MAX_FILE_REFS,
    }


def _extract_file_refs(text: str) -> set[str]:
    if not text:
        return set()
    refs: set[str] = set()
    for match in _FILE_REF_RE.findall(text):
        normalized = match
        if normalized.startswith('a/') or normalized.startswith('b/'):
            normalized = normalized[2:]
        refs.add(normalized)
    return refs


def _extract_command_from_tool_output(output: str) -> str:
    first_line = output.splitlines()[0].strip()
    if not first_line.startswith('Command:'):
        return ''
    command = first_line[len('Command:'):].strip()
    wrapped = _SHELL_WRAPPER.match(command)
    if wrapped:
        return wrapped.group('inner').strip()
    if len(command) >= 2 and command[0] == command[-1] and command[0] in {'"', "'"}:
        return command[1:-1].strip()
    return command


def _is_file_read_command(command: str) -> bool:
    return (
        command.startswith('sed -n ')
        or command.startswith('cat ')
        or command.startswith('nl -ba ')
    )


def _is_search_inventory_command(command: str) -> bool:
    return (
        'rg -n ' in command
        or command.startswith('rg --files')
        or command.startswith('find ')
        or command.startswith('ls ')
        or command.startswith('stat ')
        or command.startswith('wc -')
        or command.startswith('du -')
    )


def _is_targeted_test_command(command: str) -> bool:
    if 'pytest' not in command and 'unittest' not in command:
        return False
    return 'tests/' in command or 'tests.' in command


def _is_polling_command(command: str, output: str) -> bool:
    if 'Process running with session ID' in output:
        return True
    return (
        command.startswith('journalctl ')
        or command.startswith('tail -n ')
        or command.startswith('tail -f ')
        or command.startswith('ss -tnp')
        or command.startswith('ps -eo')
        or command.startswith('sleep ')
        or command == 'date'
        or command.startswith('date;')
        or command.startswith('systemctl ')
    )


def _is_diff_command(command: str) -> bool:
    return command.startswith('git diff') or command.startswith('git show') or command.startswith('git status')


def _is_context_gathering_command(command: str) -> bool:
    return (
        _is_file_read_command(command)
        or _is_search_inventory_command(command)
        or _is_targeted_test_command(command)
        or _is_diff_command(command)
    )


def _has_repo_wide_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            'entire repo',
            'whole repo',
            'codebase',
            'repo-wide',
            'across the repo',
            'across the codebase',
            'architecture',
            'migration',
            'review the whole',
        )
    )


def _has_broad_scope_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            'entire repo',
            'whole repo',
            'codebase',
            'repo-wide',
            'across the repo',
            'across the codebase',
            'architecture',
            'migration',
            'root cause',
            'investigate',
            'review the whole',
        )
    )


def _is_test_failure_output(output: str) -> bool:
    lowered = output.lower()
    return any(
        token in lowered
        for token in ('failed', 'traceback', 'assertionerror', 'error:', 'exception', 'not ok', 'exit code: 1')
    )


def _has_recent_command(context: Dict[str, Any], predicate) -> bool:
    return any(predicate(command) for command in context['recent_commands'])


def _is_test_fix_loop(context: Dict[str, Any]) -> bool:
    if _has_broad_scope_keywords(context['latest_text']):
        return False
    if not context['small_scope']:
        return False
    if not _has_recent_command(context, _is_targeted_test_command):
        return False
    if not _is_test_failure_output(context['recent_output_text']):
        return False
    return True


def _is_diff_followup(context: Dict[str, Any]) -> bool:
    if _has_broad_scope_keywords(context['latest_text']):
        return False
    if not context['small_scope']:
        return False
    lowered = context['latest_text'].lower()
    if not (
        'review comment' in lowered
        or 'follow up' in lowered
        or 'cleanup' in lowered
        or 'tweak' in lowered
        or 'adjust' in lowered
        or 'polish' in lowered
        or 'nit' in lowered
    ):
        return False
    return _has_recent_command(context, _is_diff_command)


def _is_stacktrace_triage(context: Dict[str, Any]) -> bool:
    if _has_broad_scope_keywords(context['latest_text']):
        return False
    if not context['small_scope']:
        return False
    if not _STACKTRACE_RE.search(context['recent_output_text']):
        return False
    lowered = context['latest_text'].lower()
    return any(token in lowered for token in ('debug', 'triage', 'error', 'failing', 'failure', 'stacktrace'))


def _is_localized_edit(context: Dict[str, Any]) -> bool:
    if _has_broad_scope_keywords(context['latest_text']):
        return False
    if not context['small_scope']:
        return False
    lowered = context['latest_text'].lower()
    if not any(token in lowered for token in ('change', 'update', 'adjust', 'edit', 'fix', 'modify', 'tweak')):
        return False
    return _has_recent_command(
        context,
        lambda command: _is_file_read_command(command)
        or _is_search_inventory_command(command)
        or _is_targeted_test_command(command)
        or command.startswith('git diff')
        or command.startswith('git show'),
    )


def _is_small_refactor(context: Dict[str, Any]) -> bool:
    if _has_broad_scope_keywords(context['latest_text']):
        return False
    if not context['small_scope']:
        return False
    lowered = context['latest_text'].lower()
    if not any(token in lowered for token in ('rename', 'extract', 'move', 'refactor', 'inline', 'simplify')):
        return False
    return _has_recent_command(context, lambda command: _is_file_read_command(command) or _is_search_inventory_command(command))


def _is_simple_synthesis(context: Dict[str, Any]) -> bool:
    if _has_broad_scope_keywords(context['latest_text']):
        return False
    if not context['small_scope']:
        return False
    lowered = context['latest_text'].lower()
    if not any(
        token in lowered
        for token in ('summarize', 'summary', 'next steps', 'what did you find', 'what do you think', 'recommend', 'conclusion')
    ):
        return False
    return _has_recent_command(
        context,
        lambda command: _is_file_read_command(command) or _is_search_inventory_command(command),
    )


def _is_bounded_investigation(context: Dict[str, Any]) -> bool:
    if _has_repo_wide_keywords(context['latest_text']):
        return False
    if not context['medium_scope']:
        return False
    lowered = context['latest_text'].lower()
    if not any(
        token in lowered
        for token in ('investigate', 'root cause', 'likely cause', 'what happened', 'why', 'analyze', 'analysis', 'debug', 'triage')
    ):
        return False
    return _has_recent_command(context, _is_context_gathering_command)


def _is_cross_file_analysis(context: Dict[str, Any]) -> bool:
    if _has_repo_wide_keywords(context['latest_text']):
        return False
    if not context['medium_scope'] or context['small_scope']:
        return False
    lowered = context['latest_text'].lower()
    if not any(
        token in lowered
        for token in ('summarize', 'summary', 'recommend', 'next steps', 'what did you find', 'what do you think', 'explain', 'plan')
    ):
        return False
    return _has_recent_command(context, _is_context_gathering_command)


def _is_medium_diff_followup(context: Dict[str, Any]) -> bool:
    if _has_repo_wide_keywords(context['latest_text']):
        return False
    if not context['medium_scope'] or context['small_scope']:
        return False
    lowered = context['latest_text'].lower()
    if not (
        'review comment' in lowered
        or 'follow up' in lowered
        or 'cleanup' in lowered
        or 'tweak' in lowered
        or 'adjust' in lowered
        or 'polish' in lowered
        or 'nit' in lowered
    ):
        return False
    return _has_recent_command(context, _is_diff_command)


def _is_medium_test_triage(context: Dict[str, Any]) -> bool:
    if _has_repo_wide_keywords(context['latest_text']):
        return False
    if not context['medium_scope']:
        return False
    if not _has_recent_command(context, _is_targeted_test_command):
        return False
    if not (_is_test_failure_output(context['recent_output_text']) or _STACKTRACE_RE.search(context['recent_output_text'])):
        return False
    lowered = context['latest_text'].lower()
    return any(
        token in lowered
        for token in ('debug', 'triage', 'error', 'failing', 'failure', 'stacktrace', 'root cause', 'investigate', 'explain')
    )


def _is_multi_file_refactor(context: Dict[str, Any]) -> bool:
    if _has_repo_wide_keywords(context['latest_text']):
        return False
    if not context['medium_scope'] or context['small_scope']:
        return False
    lowered = context['latest_text'].lower()
    if not any(token in lowered for token in ('rename', 'extract', 'move', 'refactor', 'inline', 'simplify')):
        return False
    return _has_recent_command(context, _is_context_gathering_command)


def _stable_sample_value(payload: Dict[str, Any]) -> float:
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / float(1 << 64)


def _proxy_response(request: Request, path: str, payload: Dict[str, Any]) -> Response:
    headers = _proxy_headers(request)
    stream = bool(payload.get('stream'))
    url = _proxy_url(path)
    upstream_payload, rewrite = _rewrite_passthrough_payload_for_model(payload)
    upstream = _UPSTREAM.post(
        url,
        json=upstream_payload,
        headers=headers,
        timeout=(settings.ollama_connect_timeout_seconds, settings.codex_timeout_seconds),
        stream=stream,
    )
    _record_transport_event(
        'responses_passthrough',
        path=path,
        upstream_url=url,
        stream=stream,
        status=upstream.status_code,
        auth_header=_auth_header_kind(headers),
        original_model=payload.get('model'),
        upstream_model=upstream_payload.get('model'),
        model_rewrite_applied=rewrite['applied'],
        model_rewrite_family=rewrite['rewrite_family'],
        model_rewrite_target=rewrite['target_model'],
        request_tokens=rewrite['request_tokens'],
        image_inputs_present=rewrite['image_inputs_present'],
        spark_rewrite_applied=rewrite['spark']['applied'],
        spark_eligible=rewrite['spark']['eligible'],
        spark_category=rewrite['spark']['category'],
        spark_sample_value=rewrite['spark']['sample_value'],
        mini_rewrite_applied=rewrite['mini']['applied'],
        mini_eligible=rewrite['mini']['eligible'],
        mini_category=rewrite['mini']['category'],
    )
    if not stream or upstream.status_code >= 400:
        body = upstream.content
        media_type = upstream.headers.get('content-type')
        upstream.close()
        return Response(content=body, status_code=upstream.status_code, media_type=media_type)

    def iterator():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return StreamingResponse(iterator(), status_code=upstream.status_code, media_type=upstream.headers.get('content-type'))


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "router_model": settings.router_model,
        "coder_model": settings.coder_model,
        "reasoner_model": settings.reasoner_model,
        "codex_cli_enabled": settings.enable_codex_cli,
    }


@app.get("/api/version")
def ollama_version() -> dict:
    return ollama_version_response()


@app.get("/api/tags")
def ollama_tags() -> dict:
    return ollama_tags_response()


@app.get("/internal/metrics")
def internal_metrics() -> dict:
    return transport_metrics_snapshot(_TRANSPORT_LOG_PATH)


@app.get("/v1/models")
def openai_models() -> dict:
    return openai_models_response()


@app.post("/invoke")
def invoke(req: InvokeRequest):
    response = service.invoke(req)
    return JSONResponse(response.model_dump())


@app.post("/internal/compact")
def compact(req: CompactRequest):
    request_tokens = _compact_request_tokens(req)
    _record_transport_event(
        'internal_compact_start',
        path='/internal/compact',
        session_id=req.session_id,
        refresh_if_needed=req.refresh_if_needed,
        local_model=_configured_compactor_model(),
        request_tokens=request_tokens,
        **compaction_payload_fields(
            before={
                'session_id': req.session_id,
                'items': req.items,
                'current_request': req.current_request,
                'repo_context': req.repo_context,
                'refresh_if_needed': req.refresh_if_needed,
            }
        ),
    )
    handoff = service.compact_session(
        req.session_id,
        req.items,
        current_request=req.current_request,
        repo_context=req.repo_context,
        refresh_if_needed=req.refresh_if_needed,
    )
    _record_transport_event(
        'internal_compact_completed',
        path='/internal/compact',
        session_id=req.session_id,
        refresh_if_needed=req.refresh_if_needed,
        local_model=_configured_compactor_model(),
        request_tokens=request_tokens,
        **compaction_payload_fields(after=handoff),
    )
    return JSONResponse(handoff)


@app.post("/v1/messages")
def anthropic_messages(_: Dict[str, Any]):
    return _unsupported_transport_response('v1_messages')


@app.post("/v1/chat/completions")
def openai_chat_completions(_: Dict[str, Any]):
    return _unsupported_transport_response('chat_completions')


@app.post("/v1/responses")
async def openai_responses(request: Request):
    payload = await request.json()
    if _is_inline_compaction(payload):
        sanitized_payload = sanitize_inline_compaction_payload(payload)
        job_key = inline_compaction_request_key(sanitized_payload)
        local_model = _inline_compaction_local_model()
        request_tokens = _responses_request_tokens(payload)
        _record_transport_event(
            'inline_compaction_detected',
            path='/v1/responses',
            request_key=job_key,
            stream=bool(payload.get('stream')),
            local_model=local_model,
            request_tokens=request_tokens,
            **compaction_payload_fields(before=payload),
        )
        anthropic_req = anthropic_request_from_responses(sanitized_payload)
        job, created = inline_compaction_jobs.get_or_create(
            sanitized_payload,
            anthropic_req,
            headers=_proxy_headers(request),
        )
        _record_transport_event(
            'inline_compaction_job_started' if created else 'inline_compaction_job_reused',
            path='/v1/responses',
            request_key=job.key,
            stream=bool(payload.get('stream')),
            local_model=local_model,
            request_tokens=request_tokens,
        )
        if payload.get('stream'):
            return StreamingResponse(
                iter_responses_progress(_log_inline_compaction_stream(job.iter_events(), payload, request_key=job.key), payload),
                media_type='text/event-stream',
            )
        try:
            response = job.wait()
        except RuntimeError as exc:
            _record_transport_event(
                'inline_compaction_failed',
                path='/v1/responses',
                request_key=job.key,
                stream=False,
                local_model=local_model,
                request_tokens=request_tokens,
                error=str(exc),
            )
            return JSONResponse({'error': str(exc)}, status_code=500)
        _record_transport_event(
            'inline_compaction_completed',
            path='/v1/responses',
            request_key=job.key,
            stream=False,
            request_tokens=request_tokens,
            **compaction_payload_fields(after=response),
        )
        return JSONResponse(responses_response(response, payload))
    return _proxy_response(request, '/responses', payload)


@app.post("/api/chat")
def ollama_chat(_: Dict[str, Any]):
    return _unsupported_transport_response('ollama_chat')
