from __future__ import annotations

from typing import Any, Dict, Iterable, List


_STRING_LIST_FIELDS = (
    "files_touched",
    "commands_run",
    "errors",
    "accepted_fixes",
    "rejected_ideas",
    "constraints",
    "environment_assumptions",
    "pending_todos",
    "unresolved_bugs",
    "external_references",
)


def normalize_chunk_extraction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized["repo_state"] = _normalize_repo_state(normalized.get("repo_state"))
    for field_name in _STRING_LIST_FIELDS:
        normalized[field_name] = _normalize_string_list(normalized.get(field_name))
    normalized["test_status"] = _normalize_test_status(normalized.get("test_status"))
    normalized["latest_plan"] = _normalize_latest_plan(normalized.get("latest_plan"))
    return normalized


def normalize_merged_state_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized["repo_state"] = _normalize_repo_state(normalized.get("repo_state"))
    for field_name in _STRING_LIST_FIELDS:
        normalized[field_name] = _normalize_string_list(normalized.get(field_name))
    normalized["test_status"] = _normalize_test_status(normalized.get("test_status"))
    normalized["latest_plan"] = _normalize_latest_plan(normalized.get("latest_plan"))
    return normalized


def _normalize_string_list(value: Any) -> List[str]:
    return _normalize_list(value, formatter=_format_default)


def _normalize_repo_state(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if _stringify_scalar(key)}
    if isinstance(value, list):
        entries = _normalize_list(value, formatter=_format_default)
        return {"summary": "; ".join(entries)} if entries else {}
    text = _stringify_scalar(value)
    if not text:
        return {}
    return {"summary": text}


def _normalize_test_status(value: Any) -> List[str]:
    if isinstance(value, dict):
        return [f"{key}: {_stringify_scalar(item)}" for key, item in value.items() if _stringify_scalar(key)]
    return _normalize_list(value, formatter=_format_default)


def _normalize_latest_plan(value: Any) -> List[str]:
    return _normalize_list(value, formatter=_format_plan_step)


def _normalize_list(value: Any, *, formatter) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items: Iterable[Any] = value
    else:
        items = [value]
    normalized: List[str] = []
    for item in items:
        text = formatter(item)
        if text:
            normalized.append(text)
    return normalized


def _format_plan_step(item: Any) -> str:
    if isinstance(item, dict):
        step = _stringify_scalar(item.get("step"))
        status = _stringify_scalar(item.get("status"))
        if step and status:
            return f"{step} [{status}]"
        if step:
            return step
        return _format_default(item)
    return _format_default(item)


def _format_default(item: Any) -> str:
    if isinstance(item, dict):
        parts = []
        for key, value in item.items():
            key_text = _stringify_scalar(key)
            value_text = _stringify_scalar(value)
            if key_text and value_text:
                parts.append(f"{key_text}: {value_text}")
            elif value_text:
                parts.append(value_text)
        return "; ".join(parts)
    return _stringify_scalar(item)


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return ""
