from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

from .compaction_transport import compaction_payload_fields, record_transport_event
from .compat import (
    anthropic_request_from_ollama_chat,
    anthropic_request_from_openai_chat,
    anthropic_request_from_responses,
    iter_ollama_chat_response,
    iter_openai_chat_response,
    iter_responses_progress,
    iter_responses_response,
    ollama_chat_response,
    ollama_tags_response,
    ollama_version_response,
    openai_chat_response,
    openai_models_response,
    responses_response,
)
from .config import settings
from .models import AnthropicMessagesRequest, CompactRequest, InvokeRequest
from .router import RoutingService
from .app_server import CodexAppServerBridge

app = FastAPI(title="Local Agent Router Starter", version="0.1.0")
service = RoutingService()
app_server = CodexAppServerBridge(service)
_TRANSPORT_LOG_PATH = Path('state/compaction_transport.jsonl')


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
    record_transport_event(
        _TRANSPORT_LOG_PATH,
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
    record_transport_event(
        _TRANSPORT_LOG_PATH,
        'internal_compact_completed',
        session_id=req.session_id,
        refresh_if_needed=req.refresh_if_needed,
        **compaction_payload_fields(after=handoff),
    )
    return JSONResponse(handoff)


@app.post("/v1/messages")
def anthropic_messages(req: AnthropicMessagesRequest):
    response = service.invoke_from_anthropic(req)
    return JSONResponse(response)


@app.post("/v1/chat/completions")
def openai_chat_completions(req: Dict[str, Any]):
    response = service.invoke_from_anthropic(anthropic_request_from_openai_chat(req))
    if req.get("stream"):
        return StreamingResponse(iter_openai_chat_response(response), media_type="text/event-stream")
    return JSONResponse(openai_chat_response(response))


@app.post("/v1/responses")
def openai_responses(req: Dict[str, Any]):
    if req.get("stream"):
        return StreamingResponse(
            iter_responses_progress(service.stream_from_anthropic(anthropic_request_from_responses(req)), req),
            media_type="text/event-stream",
        )
    response = service.invoke_from_anthropic(anthropic_request_from_responses(req))
    return JSONResponse(responses_response(response, req))


@app.post("/api/chat")
def ollama_chat(req: Dict[str, Any]):
    response = service.invoke_from_anthropic(anthropic_request_from_ollama_chat(req))
    if req.get("stream"):
        return StreamingResponse(iter_ollama_chat_response(response), media_type="application/x-ndjson")
    return JSONResponse(ollama_chat_response(response))


@app.websocket("/app-server/ws")
async def codex_app_server(websocket: WebSocket):
    await app_server.handle_websocket(websocket)
