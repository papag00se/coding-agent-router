from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests


class ResponsesClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: Union[float, int, Tuple[float, float]],
        *,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.headers = dict(headers or {})
        self.session = session or requests.Session()

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        num_ctx: int,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        think: Optional[Union[bool, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        del temperature, num_ctx, think, tools

        payload: Dict[str, Any] = {
            "model": model,
            "input": _responses_input(messages),
            "store": False,
            "stream": True,
        }
        if system:
            payload["instructions"] = system
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if isinstance(response_format, dict):
            payload["text"] = {"format": _json_schema_text_format(response_format)}

        with self.session.post(
            f"{self.base_url}/responses",
            json=payload,
            headers=self.headers,
            timeout=self.timeout_seconds,
            stream=True,
        ) as response:
            if response.status_code >= 400:
                preview = getattr(response, "text", "").strip().replace("\n", " ")[:500]
                raise requests.HTTPError(
                    f"responses client upstream returned HTTP {response.status_code}: {preview or 'no response body'}",
                    response=response,
                )
            body = _read_streaming_response_body(response)

        usage = body.get("usage") or {}
        return {
            "message": {"content": _response_text(body)},
            "prompt_eval_count": usage.get("input_tokens", 0),
            "eval_count": usage.get("output_tokens", 0),
            "raw_response": body,
        }


def _read_streaming_response_body(response: requests.Response) -> Dict[str, Any]:
    completed: Optional[Dict[str, Any]] = None
    parts: List[str] = []
    saw_stream_data = False

    for event_name, payload in _iter_sse_events(response):
        saw_stream_data = True
        if event_name == "response.output_text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str):
                parts.append(delta)
        elif event_name == "response.output_text.done":
            text = payload.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif event_name == "response.completed":
            response_payload = payload.get("response")
            if isinstance(response_payload, dict):
                completed = response_payload
            break
        elif event_name == "response.failed":
            raise ValueError(f"responses client returned failed stream: {json.dumps(payload, ensure_ascii=False)}")

    if completed is not None:
        return completed
    if saw_stream_data:
        return {"output_text": "".join(parts).strip(), "usage": {}}
    try:
        return response.json()
    except ValueError as exc:
        preview = getattr(response, "text", "").strip().replace("\n", " ")[:500]
        raise ValueError(f"responses client returned non-JSON output: {preview}") from exc


def _iter_sse_events(response: requests.Response) -> Iterator[tuple[str, Dict[str, Any]]]:
    current_event: Optional[str] = None
    data_lines: List[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = raw_line.strip()
        if not line:
            parsed = _parse_sse_event(current_event, data_lines)
            if parsed is not None:
                yield parsed
            current_event = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line.removeprefix("event:").strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    parsed = _parse_sse_event(current_event, data_lines)
    if parsed is not None:
        yield parsed


def _parse_sse_event(event_name: Optional[str], data_lines: List[str]) -> Optional[tuple[str, Dict[str, Any]]]:
    if not event_name:
        return None
    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        payload = {"raw": data}
    if not isinstance(payload, dict):
        payload = {"raw": payload}
    return event_name, payload


def _json_schema_text_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    name = str(schema.get("title") or "structured_output").strip().lower()
    name = "".join(char if char.isalnum() else "_" for char in name).strip("_") or "structured_output"
    return {
        "type": "json_schema",
        "name": name,
        "schema": schema,
        "strict": True,
    }


def _responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = _content_blocks(message.get("content"))
        items.append(
            {
                "type": "message",
                "role": role,
                "content": content,
            }
        )
    return items


def _content_blocks(content: Any) -> List[Dict[str, str]]:
    if isinstance(content, str):
        text = content.strip()
        return [{"type": "input_text", "text": text}] if text else []
    if isinstance(content, list):
        blocks: List[Dict[str, str]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("input_text") or item.get("output_text") or ""
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "input_text", "text": text.strip()})
        return blocks
    if content is None:
        return []
    text = str(content).strip()
    return [{"type": "input_text", "text": text}] if text else []


def _response_text(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text

    parts: List[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content_part in item.get("content") or []:
            if not isinstance(content_part, dict):
                continue
            if content_part.get("type") != "output_text":
                continue
            text = content_part.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts).strip()
