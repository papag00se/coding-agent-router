from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import settings
from ..prompt_loader import load_prompt
from ..task_metrics import estimate_model_tokens
from .models import ChunkExtraction, MergedState, MergedStatePatch, TranscriptChunk


EXTRACTION_SYSTEM_PROMPT = load_prompt('compaction_extraction_system.md')
REFINEMENT_SYSTEM_PROMPT = load_prompt('compaction_refinement_system.md')


def build_extraction_payload(chunk: TranscriptChunk, repo_context: Optional[Dict[str, Any]] = None) -> str:
    return json.dumps(
        {
            "task": "Extract chunk-local durable coding-session state.",
            "output_contract": {
                "format": "json_object_only",
                "required_keys": list(ChunkExtraction.model_fields.keys()),
            },
            "chunk": chunk.model_dump(),
            "repo_context": repo_context or {},
        },
        ensure_ascii=False,
    )


def estimate_extraction_request_tokens(
    chunk: TranscriptChunk,
    repo_context: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
) -> int:
    payload = build_extraction_payload(chunk, repo_context)
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": payload},
    ]
    return estimate_model_tokens(messages, model=model or settings.compactor_model)


def build_refinement_payload(
    state: MergedState,
    recent_raw_turns: List[Dict[str, object]],
    current_request: str,
    repo_context: Optional[Dict[str, Any]] = None,
) -> str:
    return json.dumps(
        {
            "task": "Return a bounded patch over merged durable coding-session state using newer recent raw turns.",
            "output_contract": {
                "format": "json_object_only",
                "required_keys": list(MergedStatePatch.model_fields.keys()),
            },
            "current_state": state.model_dump(),
            "recent_raw_turns": recent_raw_turns,
            "current_request": current_request,
            "repo_context": repo_context or {},
        },
        ensure_ascii=False,
    )


def estimate_refinement_request_tokens(
    state: MergedState,
    recent_raw_turns: List[Dict[str, object]],
    current_request: str,
    repo_context: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
) -> int:
    payload = build_refinement_payload(state, recent_raw_turns, current_request, repo_context)
    messages = [
        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": payload},
    ]
    return estimate_model_tokens(messages, model=model or settings.compactor_model)
