from __future__ import annotations

import json
import time
from queue import Empty, Queue
from pathlib import Path
from threading import Thread
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from .models import AnthropicMessagesRequest
from .router import flatten_content
from .tool_surface import map_stream_tool_call_to_original, map_tool_call_to_original, translate_responses_tools

ROUTER_MODEL_ID = 'router'
CODEX_MODEL_ID = 'gpt-5.4'
_CODEX_MODELS_CACHE_PATH = Path.home() / '.codex' / 'models_cache.json'
_SSE_KEEPALIVE_INTERVAL_SECONDS = 2.0


def _text_blocks(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    if isinstance(content, list):
        blocks: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and item.get('type') in {'text', 'input_text'}:
                text = item.get('text') or item.get('input_text') or ''
                if text:
                    blocks.append({'type': 'text', 'text': text})
        if blocks:
            return blocks
    text = flatten_content(content)
    return [{'type': 'text', 'text': text}] if text else []


def _tool_input(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {'input': arguments}
    return {}


def _anthropic_tools(tools: Optional[Iterable[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    normalized: List[Dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get('function') if tool.get('type') == 'function' else tool
        if not isinstance(function, dict) or not function.get('name'):
            continue
        normalized.append(
            {
                'name': function['name'],
                'description': function.get('description', ''),
                'input_schema': function.get('parameters') or {'type': 'object', 'properties': {}},
            }
        )
    return normalized or None


def _anthropic_messages(messages: Optional[Iterable[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get('role') or 'user'
        if role == 'system':
            text = flatten_content(message.get('content', ''))
            if text:
                system_parts.append(text)
            continue
        if role == 'tool':
            converted.append(
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': message.get('tool_call_id', ''),
                            'content': flatten_content(message.get('content', '')),
                        }
                    ],
                }
            )
            continue
        if role == 'assistant':
            blocks = _text_blocks(message.get('content', ''))
            for tool_call in message.get('tool_calls') or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get('function') or {}
                if not isinstance(function, dict) or not function.get('name'):
                    continue
                blocks.append(
                    {
                        'type': 'tool_use',
                        'id': tool_call.get('id') or function['name'],
                        'name': function['name'],
                        'input': _tool_input(function.get('arguments')),
                    }
                )
            converted.append({'role': 'assistant', 'content': blocks if blocks else ''})
            continue
        converted.append({'role': role, 'content': message.get('content', '')})
    return '\n\n'.join(system_parts), converted


def anthropic_request_from_openai_chat(payload: Dict[str, Any]) -> AnthropicMessagesRequest:
    system, messages = _anthropic_messages(payload.get('messages'))
    return AnthropicMessagesRequest.model_validate(
        {
            'model': payload.get('model') or ROUTER_MODEL_ID,
            'max_tokens': payload.get('max_tokens') or payload.get('max_completion_tokens') or 2048,
            'system': system or None,
            'messages': messages,
            'metadata': payload.get('metadata'),
            'tools': _anthropic_tools(payload.get('tools')),
        }
    )


def anthropic_request_from_ollama_chat(payload: Dict[str, Any]) -> AnthropicMessagesRequest:
    system, messages = _anthropic_messages(payload.get('messages'))
    return AnthropicMessagesRequest.model_validate(
        {
            'model': payload.get('model') or ROUTER_MODEL_ID,
            'max_tokens': payload.get('options', {}).get('num_predict') or payload.get('max_tokens') or 2048,
            'system': system or None,
            'messages': messages,
            'metadata': payload.get('metadata'),
            'tools': _anthropic_tools(payload.get('tools')),
        }
    )


def anthropic_request_from_responses(payload: Dict[str, Any]) -> AnthropicMessagesRequest:
    input_items = payload.get('input')
    system_parts: List[str] = []
    messages: List[Dict[str, Any]] = []
    if isinstance(input_items, str):
        messages.append({'role': 'user', 'content': input_items})
    else:
        for item in input_items or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get('type')
            if item_type == 'function_call_output':
                messages.append(
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'tool_result',
                                'tool_use_id': item.get('call_id') or item.get('id') or '',
                                'content': flatten_content(item.get('output', '')),
                            }
                        ],
                    }
                )
                continue
            if item_type == 'function_call':
                messages.append(
                    {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'tool_use',
                                'id': item.get('call_id') or item.get('id') or item.get('name', 'tool_call'),
                                'name': item.get('name', ''),
                                'input': _tool_input(item.get('arguments')),
                            }
                        ],
                    }
                )
                continue
            if item_type != 'message':
                continue
            role = item.get('role') or 'user'
            content = item.get('content', '')
            if role in {'system', 'developer'}:
                text = flatten_content(content)
                if text:
                    system_parts.append(text)
                continue
            if isinstance(content, str):
                messages.append({'role': role, 'content': content})
                continue
            blocks: List[Dict[str, Any]] = []
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                part_type = part.get('type')
                if part_type in {'input_text', 'output_text', 'text'}:
                    text = part.get('text') or ''
                    if text:
                        blocks.append({'type': 'text', 'text': text})
                elif part_type == 'function_call' and role == 'assistant':
                    blocks.append(
                        {
                            'type': 'tool_use',
                            'id': part.get('call_id') or part.get('id') or part.get('name', 'tool_call'),
                            'name': part.get('name', ''),
                            'input': _tool_input(part.get('arguments')),
                        }
                    )
                elif part_type == 'function_call_output':
                    blocks.append(
                        {
                            'type': 'tool_result',
                            'tool_use_id': part.get('call_id') or part.get('id') or '',
                            'content': flatten_content(part.get('output', '')),
                        }
                    )
            messages.append({'role': role, 'content': blocks if blocks else ''})
    instructions = payload.get('instructions')
    if instructions:
        system_parts.insert(0, flatten_content(instructions))
    translated_tools, tool_aliases = translate_responses_tools(payload.get('tools'))
    metadata = dict(payload.get('metadata') or {})
    if tool_aliases:
        metadata['_tool_aliases'] = tool_aliases
    return AnthropicMessagesRequest.model_validate(
        {
            'model': payload.get('model') or ROUTER_MODEL_ID,
            'max_tokens': payload.get('max_output_tokens') or payload.get('max_tokens') or 2048,
            'system': '\n\n'.join(part for part in system_parts if part) or None,
            'messages': messages,
            'metadata': metadata or None,
            'tools': translated_tools,
        }
    )


def _response_parts(response: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in response.get('content') or []:
        if not isinstance(block, dict):
            continue
        if block.get('type') == 'text':
            text = block.get('text', '')
            if text:
                text_parts.append(text)
        if block.get('type') == 'tool_use':
            tool_calls.append(
                {
                    'id': block.get('id') or block.get('name', 'tool_call'),
                    'type': 'function',
                    'function': {
                        'name': block.get('name', ''),
                        'arguments': json.dumps(block.get('input') or {}, ensure_ascii=False),
                    },
                }
            )
    return '\n'.join(text_parts).strip(), tool_calls


def _route_banner(response: Dict[str, Any]) -> str:
    decision = response.get('route_decision') or {}
    route = decision.get('route')
    model = response.get('model') or ROUTER_MODEL_ID
    if not route:
        return ''
    return f'[router] {route} -> {model}'


def openai_models_response() -> Dict[str, Any]:
    cache_entry = _codex_model_entry(CODEX_MODEL_ID)
    if cache_entry:
        model = dict(cache_entry)
        model.setdefault('id', model.get('slug', CODEX_MODEL_ID))
        model.setdefault('object', 'model')
        model.setdefault('created', 0)
        model.setdefault('owned_by', 'local')
        return {'object': 'list', 'data': [model]}
    return {'object': 'list', 'data': [{'id': CODEX_MODEL_ID, 'object': 'model', 'created': 0, 'owned_by': 'local'}]}


def ollama_version_response() -> Dict[str, Any]:
    return {'version': '0.0.0-router'}


def ollama_tags_response() -> Dict[str, Any]:
    return {
        'models': [
            {
                'name': ROUTER_MODEL_ID,
                'model': ROUTER_MODEL_ID,
                'modified_at': '1970-01-01T00:00:00Z',
                'size': 0,
                'digest': 'router',
                'details': {'format': 'router', 'family': 'router', 'parameter_size': '0B', 'quantization_level': 'NA'},
            }
        ]
    }


def openai_chat_response(response: Dict[str, Any]) -> Dict[str, Any]:
    text, tool_calls = _response_parts(response)
    usage = response.get('usage') or {}
    return {
        'id': 'chatcmpl_router_local',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': response.get('model') or ROUTER_MODEL_ID,
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': text or (None if tool_calls else ''),
                    **({'tool_calls': tool_calls} if tool_calls else {}),
                },
                'finish_reason': 'tool_calls' if tool_calls else 'stop',
            }
        ],
        'usage': {
            'prompt_tokens': usage.get('input_tokens', 0),
            'completion_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
        },
    }


def _response_created_at(response: Dict[str, Any]) -> int:
    created_at = ((response.get('raw_backend') or {}).get('created_at')) or response.get('created_at')
    if isinstance(created_at, str):
        try:
            return int(datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp())
        except ValueError:
            pass
    if isinstance(created_at, (int, float)):
        return int(created_at)
    return int(time.time())


def responses_response(response: Dict[str, Any], request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text, tool_calls = _response_parts(response)
    aliases = ((response.get('request_metadata') or {}).get('_tool_aliases')) or {}
    tool_calls = [map_tool_call_to_original(tool_call, aliases) for tool_call in tool_calls]
    banner = _route_banner(response)
    display_text = '\n\n'.join(part for part in [banner, text] if part)
    usage = response.get('usage') or {}
    created_at = _response_created_at(response)
    resp_id = response.get('id') or 'resp_router_local'
    output: List[Dict[str, Any]] = []
    if display_text or not tool_calls:
        output.append(
            {
                'id': 'msg_router_local',
                'type': 'message',
                'status': 'completed',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': display_text, 'annotations': []}] if display_text else [],
            }
        )
    for tool_call in tool_calls:
        output.append(
            {
                'id': tool_call['id'],
                'type': 'function_call',
                'call_id': tool_call['id'],
                'name': tool_call['function']['name'],
                'arguments': tool_call['function']['arguments'],
                'status': 'completed',
            }
        )
    return {
        'id': resp_id,
        'object': 'response',
        'created_at': created_at,
        'status': 'completed',
        'error': None,
        'incomplete_details': None,
        'instructions': (request or {}).get('instructions'),
        'max_output_tokens': (request or {}).get('max_output_tokens') or (request or {}).get('max_tokens'),
        'model': (request or {}).get('model') or response.get('model') or CODEX_MODEL_ID,
        'output': output,
        'output_text': display_text,
        'parallel_tool_calls': bool(tool_calls),
        'previous_response_id': (request or {}).get('previous_response_id'),
        'reasoning': {'effort': None, 'summary': None},
        'store': False,
        'text': {'format': {'type': 'text'}},
        'tool_choice': (request or {}).get('tool_choice', 'auto'),
        'tools': (request or {}).get('tools') or [],
        'truncation': 'disabled',
        'usage': {
            'input_tokens': usage.get('input_tokens', 0),
            'output_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
        },
        'metadata': (request or {}).get('metadata') or {},
    }


def _codex_model_entry(slug: str) -> Optional[Dict[str, Any]]:
    try:
        cache = json.loads(_CODEX_MODELS_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for model in cache.get('models') or []:
        if isinstance(model, dict) and model.get('slug') == slug:
            return model
    return None


def iter_openai_chat_response(response: Dict[str, Any]) -> Iterator[str]:
    text, tool_calls = _response_parts(response)
    chunk = {
        'id': 'chatcmpl_router_local',
        'object': 'chat.completion.chunk',
        'created': int(time.time()),
        'model': response.get('model') or ROUTER_MODEL_ID,
        'choices': [
            {
                'index': 0,
                'delta': {
                    'role': 'assistant',
                    **({'content': text} if text else {}),
                    **({'tool_calls': tool_calls} if tool_calls else {}),
                },
                'finish_reason': None,
            }
        ],
    }
    yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
    final_chunk = {
        'id': 'chatcmpl_router_local',
        'object': 'chat.completion.chunk',
        'created': int(time.time()),
        'model': response.get('model') or ROUTER_MODEL_ID,
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if tool_calls else 'stop'}],
    }
    yield f'data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n'
    yield 'data: [DONE]\n\n'


def _sse_event(event_type: str, payload: Dict[str, Any]) -> str:
    return f'event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'


def _sse_comment(comment: str) -> str:
    return f': {comment}\n\n'


def iter_completed_response_with_keepalive(
    result_factory: Callable[[], Dict[str, Any]],
    response_iterator: Callable[[Dict[str, Any]], Iterator[str]],
    *,
    keepalive_chunk: str,
) -> Iterator[str]:
    event_queue: Queue[tuple[str, Any]] = Queue()

    def _produce() -> None:
        try:
            event_queue.put(('result', result_factory()))
        except BaseException as exc:
            event_queue.put(('error', exc))

    producer = Thread(target=_produce, daemon=True)
    producer.start()

    while True:
        try:
            queue_item_type, queue_item = event_queue.get(timeout=_SSE_KEEPALIVE_INTERVAL_SECONDS)
        except Empty:
            yield keepalive_chunk
            continue

        if queue_item_type == 'error':
            raise queue_item
        yield from response_iterator(queue_item)
        return


def iter_responses_response(response: Dict[str, Any], request: Optional[Dict[str, Any]] = None) -> Iterator[str]:
    completed = responses_response(response, request)
    in_progress = dict(completed)
    in_progress['status'] = 'in_progress'
    in_progress['output'] = []
    in_progress['usage'] = None
    yield _sse_event('response.created', {'type': 'response.created', 'response': in_progress})
    yield _sse_event('response.in_progress', {'type': 'response.in_progress', 'response': in_progress})
    for output_index, item in enumerate(completed['output']):
        item_id = item['id']
        if item['type'] == 'message':
            added_item = {'id': item_id, 'type': 'message', 'status': 'in_progress', 'role': 'assistant', 'content': []}
            yield _sse_event('response.output_item.added', {'type': 'response.output_item.added', 'output_index': output_index, 'item': added_item})
            part = {'type': 'output_text', 'text': '', 'annotations': []}
            yield _sse_event(
                'response.content_part.added',
                {'type': 'response.content_part.added', 'item_id': item_id, 'output_index': output_index, 'content_index': 0, 'part': part},
            )
            text = ((item.get('content') or [{}])[0].get('text')) if item.get('content') else ''
            if text:
                yield _sse_event(
                    'response.output_text.delta',
                    {'type': 'response.output_text.delta', 'item_id': item_id, 'output_index': output_index, 'content_index': 0, 'delta': text},
                )
            done_part = {'type': 'output_text', 'text': text, 'annotations': []}
            yield _sse_event(
                'response.output_text.done',
                {'type': 'response.output_text.done', 'item_id': item_id, 'output_index': output_index, 'content_index': 0, 'text': text},
            )
            yield _sse_event(
                'response.content_part.done',
                {'type': 'response.content_part.done', 'item_id': item_id, 'output_index': output_index, 'content_index': 0, 'part': done_part},
            )
            yield _sse_event('response.output_item.done', {'type': 'response.output_item.done', 'output_index': output_index, 'item': item})
            continue
        added_item = dict(item)
        added_item['status'] = 'in_progress'
        yield _sse_event('response.output_item.added', {'type': 'response.output_item.added', 'output_index': output_index, 'item': added_item})
        yield _sse_event(
            'response.function_call_arguments.done',
            {
                'type': 'response.function_call_arguments.done',
                'item_id': item_id,
                'output_index': output_index,
                'arguments': item.get('arguments', ''),
            },
        )
        yield _sse_event('response.output_item.done', {'type': 'response.output_item.done', 'output_index': output_index, 'item': item})
    yield _sse_event('response.completed', {'type': 'response.completed', 'response': completed})


def iter_responses_progress(events: Iterable[Dict[str, Any]], request: Optional[Dict[str, Any]] = None) -> Iterator[str]:
    created_at = int(time.time())
    response_id = 'resp_router_local'
    model = (request or {}).get('model') or CODEX_MODEL_ID
    shell = {
        'id': response_id,
        'object': 'response',
        'created_at': created_at,
        'status': 'in_progress',
        'error': None,
        'incomplete_details': None,
        'instructions': (request or {}).get('instructions'),
        'max_output_tokens': (request or {}).get('max_output_tokens') or (request or {}).get('max_tokens'),
        'model': model,
        'output': [],
        'output_text': '',
        'parallel_tool_calls': False,
        'previous_response_id': (request or {}).get('previous_response_id'),
        'reasoning': {'effort': None, 'summary': None},
        'store': False,
        'text': {'format': {'type': 'text'}},
        'tool_choice': (request or {}).get('tool_choice', 'auto'),
        'tools': (request or {}).get('tools') or [],
        'truncation': 'disabled',
        'usage': None,
        'metadata': (request or {}).get('metadata') or {},
    }
    yield _sse_event('response.created', {'type': 'response.created', 'response': shell})
    yield _sse_event('response.in_progress', {'type': 'response.in_progress', 'response': shell})

    item_id = 'msg_router_local'
    text_started = False
    text_output_index = 0
    text = ''
    next_output_index = 1
    emitted_tool_call_ids: set[str] = set()
    _, aliases = translate_responses_tools((request or {}).get('tools'))
    event_queue: Queue[tuple[str, Any]] = Queue()

    def _produce_events() -> None:
        try:
            for event in events:
                event_queue.put(('event', event))
        except BaseException as exc:
            event_queue.put(('error', exc))
        finally:
            event_queue.put(('done', None))

    producer = Thread(target=_produce_events, daemon=True)
    producer.start()

    while True:
        try:
            queue_item_type, queue_item = event_queue.get(timeout=_SSE_KEEPALIVE_INTERVAL_SECONDS)
        except Empty:
            yield _sse_comment('keepalive')
            continue

        if queue_item_type == 'done':
            return
        if queue_item_type == 'error':
            raise queue_item

        event = queue_item
        if event.get('type') == 'progress':
            progress_response = dict(shell)
            metadata = dict(progress_response.get('metadata') or {})
            progress_payload = {
                'stage': event.get('stage'),
                'message': event.get('message'),
                'heartbeat': bool(event.get('heartbeat')),
            }
            for key in ('timestamp', 'elapsed_seconds', 'chunk_id', 'chunk_index', 'chunk_count', 'iteration', 'iteration_count'):
                if key in event:
                    progress_payload[key] = event.get(key)
            metadata['_compaction_progress'] = progress_payload
            progress_response['metadata'] = metadata
            yield _sse_event('response.in_progress', {'type': 'response.in_progress', 'response': progress_response})
            continue

        if event.get('type') == 'failed':
            failed = dict(shell)
            failed['status'] = 'failed'
            failed['error'] = {
                'message': str(event.get('message') or 'request failed'),
                'type': 'server_error',
            }
            yield _sse_event('response.failed', {'type': 'response.failed', 'response': failed})
            return

        if event.get('type') == 'final':
            completed = responses_response(event['response'], request)
            banner = _route_banner(event['response'])
            if banner and not text_started:
                text_started = True
                text = banner
                yield _sse_event(
                    'response.output_item.added',
                    {
                        'type': 'response.output_item.added',
                        'output_index': text_output_index,
                        'item': {'id': item_id, 'type': 'message', 'status': 'in_progress', 'role': 'assistant', 'content': []},
                    },
                )
                yield _sse_event(
                    'response.content_part.added',
                    {
                        'type': 'response.content_part.added',
                        'item_id': item_id,
                        'output_index': text_output_index,
                        'content_index': 0,
                        'part': {'type': 'output_text', 'text': '', 'annotations': []},
                    },
                )
                yield _sse_event(
                    'response.output_text.delta',
                    {
                        'type': 'response.output_text.delta',
                        'item_id': item_id,
                        'output_index': text_output_index,
                        'content_index': 0,
                        'delta': banner,
                    },
                )
            elif banner and text_started and not text.startswith(banner):
                text = '\n\n'.join([banner, text]) if text else banner
            if text_started:
                yield _sse_event(
                    'response.output_text.done',
                    {'type': 'response.output_text.done', 'item_id': item_id, 'output_index': text_output_index, 'content_index': 0, 'text': text},
                )
                yield _sse_event(
                    'response.content_part.done',
                    {
                        'type': 'response.content_part.done',
                        'item_id': item_id,
                        'output_index': text_output_index,
                        'content_index': 0,
                        'part': {'type': 'output_text', 'text': text, 'annotations': []},
                    },
                )
                yield _sse_event('response.output_item.done', {'type': 'response.output_item.done', 'output_index': text_output_index, 'item': completed['output'][0]})
                for item in completed['output'][1:]:
                    if item.get('id') in emitted_tool_call_ids:
                        continue
                    output_index = next_output_index
                    next_output_index += 1
                    yield _sse_event('response.output_item.added', {'type': 'response.output_item.added', 'output_index': output_index, 'item': item})
                    if item['type'] == 'function_call':
                        yield _sse_event(
                            'response.function_call_arguments.done',
                            {
                                'type': 'response.function_call_arguments.done',
                                'item_id': item['id'],
                                'output_index': output_index,
                                'arguments': item.get('arguments', ''),
                            },
                        )
                    yield _sse_event('response.output_item.done', {'type': 'response.output_item.done', 'output_index': output_index, 'item': item})
            elif completed['output']:
                for item in completed['output']:
                    if item.get('id') in emitted_tool_call_ids:
                        continue
                    output_index = 0 if not emitted_tool_call_ids else next_output_index
                    if emitted_tool_call_ids:
                        next_output_index += 1
                    yield _sse_event('response.output_item.added', {'type': 'response.output_item.added', 'output_index': output_index, 'item': item})
                    if item['type'] == 'function_call':
                        yield _sse_event(
                            'response.function_call_arguments.done',
                            {
                                'type': 'response.function_call_arguments.done',
                                'item_id': item['id'],
                                'output_index': output_index,
                                'arguments': item.get('arguments', ''),
                            },
                        )
                    yield _sse_event('response.output_item.done', {'type': 'response.output_item.done', 'output_index': output_index, 'item': item})
            yield _sse_event('response.completed', {'type': 'response.completed', 'response': completed})
            return

        if event.get('type') == 'text_delta':
            if not text_started:
                text_started = True
                banner = ''
                yield _sse_event(
                    'response.output_item.added',
                    {
                        'type': 'response.output_item.added',
                        'output_index': text_output_index,
                        'item': {'id': item_id, 'type': 'message', 'status': 'in_progress', 'role': 'assistant', 'content': []},
                    },
                )
                yield _sse_event(
                    'response.content_part.added',
                    {
                        'type': 'response.content_part.added',
                        'item_id': item_id,
                        'output_index': text_output_index,
                        'content_index': 0,
                        'part': {'type': 'output_text', 'text': '', 'annotations': []},
                    },
                )
            separator = ''
            if text:
                separator = '\n\n'
            delta_text = f'{separator}{event.get("delta", "")}' if separator else event.get('delta', '')
            text += delta_text
            yield _sse_event(
                'response.output_text.delta',
                {
                    'type': 'response.output_text.delta',
                    'item_id': item_id,
                    'output_index': text_output_index,
                    'content_index': 0,
                    'delta': delta_text,
                },
            )
            continue

        if event.get('type') == 'tool_calls':
            for tool_call in event.get('tool_calls') or []:
                tool_call = map_stream_tool_call_to_original(tool_call, aliases)
                call_id = tool_call.get('id')
                if not call_id or call_id in emitted_tool_call_ids:
                    continue
                emitted_tool_call_ids.add(call_id)
                function = tool_call.get('function', {})
                item = {
                    'id': call_id,
                    'type': 'function_call',
                    'call_id': call_id,
                    'name': function.get('name', ''),
                    'arguments': json.dumps(function.get('arguments') or {}, ensure_ascii=False),
                    'status': 'completed',
                }
                output_index = next_output_index if text_started else 0
                if text_started:
                    next_output_index += 1
                yield _sse_event('response.output_item.added', {'type': 'response.output_item.added', 'output_index': output_index, 'item': item})
                yield _sse_event(
                    'response.function_call_arguments.done',
                    {
                        'type': 'response.function_call_arguments.done',
                        'item_id': call_id,
                        'output_index': output_index,
                        'arguments': item['arguments'],
                    },
                )
                yield _sse_event('response.output_item.done', {'type': 'response.output_item.done', 'output_index': output_index, 'item': item})
            continue

        raise RuntimeError('unexpected event type in iter_responses_progress')


def anthropic_messages_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal fields to produce standard Anthropic Messages API response."""
    return {
        'id': response.get('id', 'msg_router_local'),
        'type': 'message',
        'role': 'assistant',
        'content': response.get('content', []),
        'model': response.get('model', 'router'),
        'stop_reason': response.get('stop_reason', 'end_turn'),
        'stop_sequence': response.get('stop_sequence'),
        'usage': response.get('usage', {'input_tokens': 0, 'output_tokens': 0}),
    }


def iter_anthropic_messages_response(response: Dict[str, Any]) -> Iterator[str]:
    """Wrap a completed response in Anthropic Messages SSE stream format."""
    clean = anthropic_messages_response(response)
    message_start = dict(clean)
    message_start['content'] = []
    message_start['stop_reason'] = None
    message_start['usage'] = {'input_tokens': clean['usage'].get('input_tokens', 0), 'output_tokens': 0}
    yield _sse_event('message_start', {'type': 'message_start', 'message': message_start})

    for index, block in enumerate(clean.get('content') or []):
        block_type = block.get('type')
        if block_type == 'text':
            yield _sse_event('content_block_start', {
                'type': 'content_block_start', 'index': index,
                'content_block': {'type': 'text', 'text': ''},
            })
            text = block.get('text', '')
            if text:
                yield _sse_event('content_block_delta', {
                    'type': 'content_block_delta', 'index': index,
                    'delta': {'type': 'text_delta', 'text': text},
                })
            yield _sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index})
        elif block_type == 'tool_use':
            yield _sse_event('content_block_start', {
                'type': 'content_block_start', 'index': index,
                'content_block': {'type': 'tool_use', 'id': block.get('id', ''), 'name': block.get('name', ''), 'input': {}},
            })
            input_json = json.dumps(block.get('input', {}), ensure_ascii=False)
            if input_json:
                yield _sse_event('content_block_delta', {
                    'type': 'content_block_delta', 'index': index,
                    'delta': {'type': 'input_json_delta', 'partial_json': input_json},
                })
            yield _sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index})

    yield _sse_event('message_delta', {
        'type': 'message_delta',
        'delta': {'stop_reason': clean['stop_reason'], 'stop_sequence': clean.get('stop_sequence')},
        'usage': {'output_tokens': clean['usage'].get('output_tokens', 0)},
    })
    yield _sse_event('message_stop', {'type': 'message_stop'})


def iter_anthropic_messages_progress(events: Iterable[Dict[str, Any]], request: Optional[Dict[str, Any]] = None) -> Iterator[str]:
    """Convert streaming internal events to Anthropic Messages SSE stream with keepalive."""
    model = (request or {}).get('model') or 'router'
    message_shell = {
        'id': 'msg_router_local', 'type': 'message', 'role': 'assistant',
        'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None,
        'usage': {'input_tokens': 0, 'output_tokens': 0},
    }
    yield _sse_event('message_start', {'type': 'message_start', 'message': message_shell})

    content_index = 0
    text_started = False
    event_queue: Queue[tuple[str, Any]] = Queue()

    def _produce() -> None:
        try:
            for event in events:
                event_queue.put(('event', event))
        except BaseException as exc:
            event_queue.put(('error', exc))
        finally:
            event_queue.put(('done', None))

    producer = Thread(target=_produce, daemon=True)
    producer.start()

    while True:
        try:
            item_type, item = event_queue.get(timeout=_SSE_KEEPALIVE_INTERVAL_SECONDS)
        except Empty:
            yield _sse_comment('keepalive')
            continue

        if item_type == 'done':
            break
        if item_type == 'error':
            raise item

        event = item

        if event.get('type') == 'progress':
            yield _sse_comment(event.get('message', 'processing'))
            continue

        if event.get('type') == 'text_delta':
            if not text_started:
                text_started = True
                yield _sse_event('content_block_start', {
                    'type': 'content_block_start', 'index': content_index,
                    'content_block': {'type': 'text', 'text': ''},
                })
            delta = event.get('delta', '')
            if delta:
                yield _sse_event('content_block_delta', {
                    'type': 'content_block_delta', 'index': content_index,
                    'delta': {'type': 'text_delta', 'text': delta},
                })
            continue

        if event.get('type') == 'tool_calls':
            if text_started:
                yield _sse_event('content_block_stop', {'type': 'content_block_stop', 'index': content_index})
                content_index += 1
                text_started = False
            for tool_call in event.get('tool_calls') or []:
                func = tool_call.get('function', {})
                tool_id = tool_call.get('id') or f'toolu_{content_index}'
                yield _sse_event('content_block_start', {
                    'type': 'content_block_start', 'index': content_index,
                    'content_block': {'type': 'tool_use', 'id': tool_id, 'name': func.get('name', ''), 'input': {}},
                })
                arguments = func.get('arguments', {})
                partial_json = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments)
                yield _sse_event('content_block_delta', {
                    'type': 'content_block_delta', 'index': content_index,
                    'delta': {'type': 'input_json_delta', 'partial_json': partial_json},
                })
                yield _sse_event('content_block_stop', {'type': 'content_block_stop', 'index': content_index})
                content_index += 1
            continue

        if event.get('type') == 'final':
            response = event.get('response', {})
            if text_started:
                yield _sse_event('content_block_stop', {'type': 'content_block_stop', 'index': content_index})
                content_index += 1
                text_started = False
            usage = response.get('usage', {})
            stop_reason = response.get('stop_reason', 'end_turn')
            yield _sse_event('message_delta', {
                'type': 'message_delta',
                'delta': {'stop_reason': stop_reason, 'stop_sequence': response.get('stop_sequence')},
                'usage': {'output_tokens': usage.get('output_tokens', 0)},
            })
            yield _sse_event('message_stop', {'type': 'message_stop'})
            return

        if event.get('type') == 'failed':
            if text_started:
                yield _sse_event('content_block_stop', {'type': 'content_block_stop', 'index': content_index})
            yield _sse_event('message_delta', {
                'type': 'message_delta',
                'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
                'usage': {'output_tokens': 0},
            })
            yield _sse_event('message_stop', {'type': 'message_stop'})
            return

    # Stream ended without final event — close gracefully
    if text_started:
        yield _sse_event('content_block_stop', {'type': 'content_block_stop', 'index': content_index})
    yield _sse_event('message_delta', {
        'type': 'message_delta',
        'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
        'usage': {'output_tokens': 0},
    })
    yield _sse_event('message_stop', {'type': 'message_stop'})


def ollama_chat_response(response: Dict[str, Any]) -> Dict[str, Any]:
    text, tool_calls = _response_parts(response)
    raw_backend = response.get('raw_backend') or {}
    return {
        'model': response.get('model') or ROUTER_MODEL_ID,
        'created_at': raw_backend.get('created_at') or datetime.now(timezone.utc).isoformat(),
        'message': {
            'role': 'assistant',
            'content': text,
            **(
                {
                    'tool_calls': [
                        {
                            'function': {
                                'name': tool_call['function']['name'],
                                'arguments': json.loads(tool_call['function']['arguments']),
                            }
                        }
                        for tool_call in tool_calls
                    ]
                }
                if tool_calls
                else {}
            ),
        },
        'done': True,
        'done_reason': 'tool_calls' if tool_calls else 'stop',
        'total_duration': raw_backend.get('total_duration', 0),
        'load_duration': raw_backend.get('load_duration', 0),
        'prompt_eval_count': raw_backend.get('prompt_eval_count', 0),
        'eval_count': raw_backend.get('eval_count', 0),
    }


def iter_ollama_chat_response(response: Dict[str, Any]) -> Iterator[str]:
    yield json.dumps(ollama_chat_response(response), ensure_ascii=False) + '\n'
