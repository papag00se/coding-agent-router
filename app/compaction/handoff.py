from __future__ import annotations

import json
from typing import Any, Dict

from ..prompt_loader import render_prompt
from .models import CodexHandoffFlow, DurableMemorySet, SessionHandoff


def build_codex_handoff_flow(memory: DurableMemorySet, handoff: SessionHandoff, *, current_request: str | None = None) -> CodexHandoffFlow:
    session_handoff = _strip_current_request_section(memory.session_handoff)
    return CodexHandoffFlow(
        durable_memory=[
            {"name": "TASK_STATE.md", "content": memory.task_state},
            {"name": "DECISIONS.md", "content": memory.decisions},
            {"name": "FAILURES_TO_AVOID.md", "content": memory.failures_to_avoid},
            {"name": "NEXT_STEPS.md", "content": memory.next_steps},
            {"name": "SESSION_HANDOFF.md", "content": session_handoff},
        ],
        structured_handoff=handoff.model_dump(exclude={"recent_raw_turns", "current_request"}),
        recent_raw_turns=handoff.recent_raw_turns,
        current_request=current_request if current_request is not None else handoff.current_request,
    )


def render_compacted_flow(flow: CodexHandoffFlow | Dict[str, Any], *, current_request: str = "") -> str:
    if isinstance(flow, CodexHandoffFlow):
        payload = flow.model_dump()
    else:
        payload = flow
    durable_memory_blocks = "\n\n".join(
        f"### {item['name']}\n{item['content'].rstrip()}" for item in payload.get("durable_memory") or []
    )
    return render_prompt(
        "compacted_flow.md",
        {
            "DURABLE_MEMORY_BLOCKS": durable_memory_blocks,
            "STRUCTURED_HANDOFF": json.dumps(payload.get("structured_handoff") or {}, ensure_ascii=False, indent=2),
            "RECENT_RAW_TURNS": json.dumps(payload.get("recent_raw_turns") or [], ensure_ascii=False, indent=2),
            "CURRENT_REQUEST": current_request or payload.get("current_request") or "",
        },
    )


def render_inline_compaction_summary(flow: CodexHandoffFlow | Dict[str, Any], *, current_request: str = "") -> str:
    if isinstance(flow, CodexHandoffFlow):
        payload = flow.model_dump()
    else:
        payload = flow

    sections = [
        item["content"].rstrip()
        for item in payload.get("durable_memory") or []
        if isinstance(item, dict) and isinstance(item.get("content"), str) and item["content"].strip()
    ]
    request_text = current_request or payload.get("current_request") or ""
    if request_text:
        sections.append(f"# Current Request\n{request_text}".rstrip())
    return "\n\n".join(section for section in sections if section).strip()


def render_codex_support_prompt(flow: CodexHandoffFlow, *, system: str = "", current_request: str = "") -> str:
    request_text = current_request or flow.current_request
    durable_memory_blocks = "\n\n".join(
        f"### {item['name']}\n{item['content'].rstrip()}" for item in flow.durable_memory
    )
    return render_prompt(
        'codex_support_prompt.md',
        {
            'SYSTEM_SECTION': f"System instructions:\n{system}\n\n" if system else "",
            'DURABLE_MEMORY_BLOCKS': durable_memory_blocks,
            'STRUCTURED_HANDOFF': json.dumps(flow.structured_handoff, ensure_ascii=False, indent=2),
            'RECENT_RAW_TURNS': json.dumps(flow.recent_raw_turns, ensure_ascii=False, indent=2),
            'CURRENT_REQUEST': request_text,
        },
    )


def _strip_current_request_section(content: str) -> str:
    if "## Current Request" not in content:
        return content

    lines = content.splitlines()
    filtered: list[str] = []
    skipping = False
    for line in lines:
        if line == "## Current Request":
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            filtered.append(line)

    normalized = "\n".join(filtered).rstrip()
    return normalized + "\n" if normalized else ""
