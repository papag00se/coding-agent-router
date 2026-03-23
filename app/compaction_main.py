from __future__ import annotations

from typing import Any, Dict, Iterable
from pathlib import Path

import requests
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .app_server import CodexAppServerBridge
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

app = FastAPI(title="Local Agent Router Compaction Service", version="0.1.0")
service = RoutingService()
app_server = CodexAppServerBridge(service, mode="compaction_only")
inline_compaction_jobs = InlineCompactionJobManager(service)
_TRANSPORT_LOG_PATH = Path('state/compaction_transport.jsonl')
_UPSTREAM = requests.Session()


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
        {
            'error': (
                'The compaction service is app-server only. '
                'Use /app-server/ws for extension traffic and thread/compact/start for local compaction.'
            )
        },
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


def _proxy_response(request: Request, path: str, payload: Dict[str, Any]) -> Response:
    headers = _proxy_headers(request)
    stream = bool(payload.get('stream'))
    url = _proxy_url(path)
    upstream = _UPSTREAM.post(
        url,
        json=payload,
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
        "app_server_mode": "compaction_only",
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


@app.websocket("/app-server/ws")
async def codex_app_server(websocket: WebSocket):
    await app_server.handle_websocket(websocket)
