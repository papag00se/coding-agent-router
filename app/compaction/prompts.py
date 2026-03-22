from __future__ import annotations

import json
from typing import Dict, Optional

from ..prompt_loader import load_prompt
from .models import ChunkExtraction, TranscriptChunk


EXTRACTION_SYSTEM_PROMPT = load_prompt('compaction_extraction_system.md')


def build_extraction_payload(chunk: TranscriptChunk, repo_context: Optional[Dict[str, Any]] = None) -> str:
    return json.dumps(
        {
            "task": "Extract chunk-local durable coding-session state.",
            "output_contract": {
                "format": "json_object_only",
                "required_keys": list(ChunkExtraction.model_fields.keys()),
            },
            "schema": ChunkExtraction.model_json_schema(),
            "chunk": chunk.model_dump(),
            "repo_context": repo_context or {},
        },
        ensure_ascii=False,
    )
