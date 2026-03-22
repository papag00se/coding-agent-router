from __future__ import annotations

import json

from ..prompt_loader import render_prompt
from .models import CodexHandoffFlow, DurableMemorySet, SessionHandoff


def build_codex_handoff_flow(memory: DurableMemorySet, handoff: SessionHandoff, *, current_request: str | None = None) -> CodexHandoffFlow:
    return CodexHandoffFlow(
        durable_memory=[
            {"name": "TASK_STATE.md", "content": memory.task_state},
            {"name": "DECISIONS.md", "content": memory.decisions},
            {"name": "FAILURES_TO_AVOID.md", "content": memory.failures_to_avoid},
            {"name": "NEXT_STEPS.md", "content": memory.next_steps},
            {"name": "SESSION_HANDOFF.md", "content": memory.session_handoff},
        ],
        structured_handoff=handoff.model_dump(),
        recent_raw_turns=handoff.recent_raw_turns,
        current_request=current_request if current_request is not None else handoff.current_request,
    )


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
