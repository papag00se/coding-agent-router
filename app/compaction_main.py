from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable
from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .compaction.normalize import sanitize_inline_compaction_payload
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

app = FastAPI(title="Local Agent Router Compaction Service", version="0.1.0")
service = RoutingService()
inline_compaction_jobs = InlineCompactionJobManager(service)
_TRANSPORT_LOG_PATH = Path('state/compaction_transport.jsonl')
_UPSTREAM = requests.Session()
_SPARK_MAX_REQUEST_TOKENS = 114_688
_SHELL_WRAPPER = re.compile(r"^(?:/bin/)?bash\s+-lc\s+(['\"])(?P<inner>.*)\1$", re.DOTALL)
_FILE_REF_RE = re.compile(r"(?:(?:\.\.?/|/)?[\w.-]+/)+[\w.-]+\.[A-Za-z0-9_+-]{1,10}")
_STACKTRACE_RE = re.compile(
    r"(?:Traceback \(most recent call last\)|AssertionError|TypeError|ValueError|RuntimeError|SyntaxError|ReferenceError|Error:|FAILED\b|E\s+.+)",
    re.IGNORECASE,
)


def _record_transport_event(event: str, **fields: Any) -> None:
    record_transport_event(_TRANSPORT_LOG_PATH, event, **fields)


def _log_inline_compaction_stream(events: Iterable[Dict[str, Any]], payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for event in events:
        if event.get('type') == 'final':
            response = event.get('response')
            _record_transport_event(
                'inline_compaction_completed',
                stream=True,
                **compaction_payload_fields(after=response),
            )
        if event.get('type') == 'failed':
            _record_transport_event(
                'inline_compaction_failed',
                stream=True,
                error=str(event.get('message') or 'inline compaction failed'),
            )
        yield event


def _unsupported_transport_response(transport: str) -> JSONResponse:
    _record_transport_event('unsupported_transport', transport=transport)
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


def _rewrite_passthrough_payload_for_spark(payload: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    request_tokens = estimate_openai_tokens(payload, model=str(payload.get('model') or 'gpt-5.4'))
    category = _qualifying_spark_category(payload)
    eligible = (
        payload.get('model') == 'gpt-5.4'
        and request_tokens <= _SPARK_MAX_REQUEST_TOKENS
        and category is not None
    )
    sample_value = _stable_sample_value(payload)
    applied = eligible and sample_value < settings.codex_spark_qualified_rate
    if not applied:
        return payload, {
            'applied': False,
            'eligible': eligible,
            'category': category,
            'request_tokens': request_tokens,
            'sample_value': sample_value,
        }
    rewritten = dict(payload)
    rewritten['model'] = settings.codex_spark_model
    return rewritten, {
        'applied': True,
        'eligible': True,
        'category': category,
        'request_tokens': request_tokens,
        'sample_value': sample_value,
    }


def _qualifying_spark_category(payload: Dict[str, Any]) -> str | None:
    context = _spark_context(payload)
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


def _spark_context(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    return _has_recent_command(
        context,
        lambda command: command.startswith('git diff') or command.startswith('git show') or command.startswith('git status'),
    )


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


def _stable_sample_value(payload: Dict[str, Any]) -> float:
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / float(1 << 64)


def _proxy_response(request: Request, path: str, payload: Dict[str, Any]) -> Response:
    headers = _proxy_headers(request)
    stream = bool(payload.get('stream'))
    url = _proxy_url(path)
    upstream_payload, rewrite = _rewrite_passthrough_payload_for_spark(payload)
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
        spark_rewrite_applied=rewrite['applied'],
        spark_eligible=rewrite['eligible'],
        spark_category=rewrite['category'],
        spark_request_tokens=rewrite['request_tokens'],
        spark_sample_value=rewrite['sample_value'],
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


@app.get("/v1/models")
def openai_models() -> dict:
    return openai_models_response()


@app.post("/invoke")
def invoke(req: InvokeRequest):
    response = service.invoke(req)
    return JSONResponse(response.model_dump())


@app.post("/internal/compact")
def compact(req: CompactRequest):
    _record_transport_event(
        'internal_compact_start',
        session_id=req.session_id,
        refresh_if_needed=req.refresh_if_needed,
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
        session_id=req.session_id,
        refresh_if_needed=req.refresh_if_needed,
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
        _record_transport_event(
            'inline_compaction_detected',
            request_key=job_key,
            stream=bool(payload.get('stream')),
            **compaction_payload_fields(before=payload),
        )
        anthropic_req = anthropic_request_from_responses(sanitized_payload)
        job, created = inline_compaction_jobs.get_or_create(sanitized_payload, anthropic_req)
        _record_transport_event(
            'inline_compaction_job_started' if created else 'inline_compaction_job_reused',
            request_key=job.key,
            stream=bool(payload.get('stream')),
        )
        if payload.get('stream'):
            return StreamingResponse(
                iter_responses_progress(_log_inline_compaction_stream(job.iter_events(), payload), payload),
                media_type='text/event-stream',
            )
        try:
            response = job.wait()
        except RuntimeError as exc:
            _record_transport_event(
                'inline_compaction_failed',
                request_key=job.key,
                stream=False,
                error=str(exc),
            )
            return JSONResponse({'error': str(exc)}, status_code=500)
        _record_transport_event(
            'inline_compaction_completed',
            request_key=job.key,
            stream=False,
            **compaction_payload_fields(after=response),
        )
        return JSONResponse(responses_response(response, payload))
    return _proxy_response(request, '/responses', payload)


@app.post("/api/chat")
def ollama_chat(_: Dict[str, Any]):
    return _unsupported_transport_response('ollama_chat')
