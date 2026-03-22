from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple


def translate_responses_tools(tools: Iterable[Dict[str, Any]] | None) -> Tuple[List[Dict[str, Any]] | None, Dict[str, Dict[str, Any]]]:
    translated: List[Dict[str, Any]] = []
    aliases: Dict[str, Dict[str, Any]] = {}

    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get('type')
        name = tool.get('name') or (tool.get('function') or {}).get('name')

        if tool_type == 'custom' and name == 'apply_patch':
            alias = 'write_patch'
            translated.append(
                {
                    'name': alias,
                    'description': 'Apply a unified patch to files.',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'patch': {
                                'type': 'string',
                                'description': 'Unified patch text beginning with *** Begin Patch.',
                            }
                        },
                        'required': ['patch'],
                    },
                }
            )
            aliases[alias] = {'original_name': 'apply_patch', 'mode': 'patch'}
            continue

        if tool_type != 'function' or not name:
            continue

        if name == 'rg':
            alias = 'search_text'
            translated.append(
                {
                    'name': alias,
                    'description': 'Search repository text by pattern.',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'pattern': {'type': 'string', 'description': 'Text or regex to search for.'},
                            'path': {'type': 'string', 'description': 'Optional directory or file path to search.'},
                        },
                        'required': ['pattern'],
                    },
                }
            )
            aliases[alias] = {'original_name': 'rg', 'mode': 'identity'}
            continue

        if name == 'exec_command':
            alias = 'run_shell'
            translated.append(
                {
                    'name': alias,
                    'description': 'Run one shell command in the repository.',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'command': {'type': 'string', 'description': 'Shell command to run.'},
                            'cwd': {'type': 'string', 'description': 'Optional working directory.'},
                            'wait_ms': {'type': 'number', 'description': 'Optional wait time in milliseconds.'},
                        },
                        'required': ['command'],
                    },
                }
            )
            aliases[alias] = {'original_name': 'exec_command', 'mode': 'exec_command'}
            continue

        if name == 'write_stdin':
            alias = 'send_terminal_input'
            translated.append(
                {
                    'name': alias,
                    'description': 'Send input to a running terminal session.',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'session_id': {'type': 'number', 'description': 'Running session id.'},
                            'text': {'type': 'string', 'description': 'Text to send.'},
                            'wait_ms': {'type': 'number', 'description': 'Optional wait time in milliseconds.'},
                        },
                        'required': ['session_id'],
                    },
                }
            )
            aliases[alias] = {'original_name': 'write_stdin', 'mode': 'write_stdin'}
            continue

        if name == 'update_plan':
            alias = 'set_plan'
            translated.append(
                {
                    'name': alias,
                    'description': 'Set the current task plan.',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'explanation': {'type': 'string', 'description': 'Optional short explanation.'},
                            'steps': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'step': {'type': 'string'},
                                        'status': {'type': 'string'},
                                    },
                                    'required': ['step', 'status'],
                                },
                            },
                        },
                        'required': ['steps'],
                    },
                }
            )
            aliases[alias] = {'original_name': 'update_plan', 'mode': 'update_plan'}
            continue

        if name == 'view_image':
            alias = 'view_image'
            translated.append(
                {
                    'name': alias,
                    'description': 'Open a local image file.',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'path': {'type': 'string', 'description': 'Local image file path.'},
                        },
                        'required': ['path'],
                    },
                }
            )
            aliases[alias] = {'original_name': 'view_image', 'mode': 'identity'}
            continue

    return translated or None, aliases


def map_tool_call_to_original(tool_call: Dict[str, Any], aliases: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    function = tool_call.get('function') or {}
    alias = function.get('name', '')
    mapped = aliases.get(alias)
    if not mapped:
        return tool_call

    args = function.get('arguments', {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {'raw': args}
    if not isinstance(args, dict):
        args = {'value': args}

    original_name = mapped['original_name']
    original_args = _map_args(args, mapped['mode'])
    return {
        'id': tool_call.get('id'),
        'type': 'function',
        'function': {
            'name': original_name,
            'arguments': json.dumps(original_args, ensure_ascii=False),
        },
    }


def map_stream_tool_call_to_original(tool_call: Dict[str, Any], aliases: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    function = tool_call.get('function') or {}
    alias = function.get('name', '')
    mapped = aliases.get(alias)
    if not mapped:
        return tool_call
    args = function.get('arguments', {})
    if not isinstance(args, dict):
        args = {'value': args}
    return {
        'id': tool_call.get('id'),
        'type': 'function',
        'function': {
            'name': mapped['original_name'],
            'arguments': _map_args(args, mapped['mode']),
        },
    }


def _map_args(args: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == 'patch':
        return {'input': args.get('patch', '')}
    if mode == 'exec_command':
        mapped = {'cmd': args.get('command', '')}
        if args.get('cwd'):
            mapped['workdir'] = args['cwd']
        if args.get('wait_ms') is not None:
            mapped['yield_time_ms'] = args['wait_ms']
        return mapped
    if mode == 'write_stdin':
        mapped = {'session_id': args.get('session_id')}
        if 'text' in args:
            mapped['chars'] = args.get('text', '')
        if args.get('wait_ms') is not None:
            mapped['yield_time_ms'] = args['wait_ms']
        return mapped
    if mode == 'update_plan':
        return {'explanation': args.get('explanation'), 'plan': args.get('steps', [])}
    return args
