from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import settings
from ..task_metrics import estimate_tokens
from .chunking import chunk_transcript_items, split_recent_raw_turns
from .durable_memory import build_session_handoff, render_durable_memory
from .extractor import CompactionExtractor
from .handoff import build_codex_handoff_flow
from .merger import merge_states
from .models import CodexHandoffFlow, SessionHandoff
from .storage import CompactionStorage


class CompactionService:
    def __init__(
        self,
        extractor: Optional[CompactionExtractor] = None,
        storage: Optional[CompactionStorage] = None,
    ) -> None:
        self.extractor = extractor or CompactionExtractor()
        self.storage = storage or CompactionStorage()

    def compact_transcript(
        self,
        session_id: str,
        items: List[Dict[str, Any]],
        *,
        current_request: str,
        repo_context: Optional[Dict[str, Any]] = None,
    ) -> SessionHandoff:
        compactable, recent_raw_turns = split_recent_raw_turns(items, settings.compactor_keep_raw_tokens)
        source_items = compactable or items
        chunks = chunk_transcript_items(
            source_items,
            target_tokens=settings.compactor_target_chunk_tokens,
            max_tokens=settings.compactor_max_chunk_tokens,
            overlap_tokens=settings.compactor_overlap_tokens,
        )
        extractions = [self.extractor.extract_chunk(chunk, repo_context) for chunk in chunks]
        merged = merge_states(extractions)
        memory = render_durable_memory(merged, recent_raw_turns, current_request)
        handoff = build_session_handoff(merged, recent_raw_turns, current_request)

        self.storage.save_chunk_extractions(session_id, extractions)
        self.storage.save_merged_state(session_id, merged)
        self.storage.save_durable_memory(session_id, memory)
        self.storage.save_handoff(session_id, handoff)
        return handoff

    def load_latest_handoff(self, session_id: str) -> Optional[SessionHandoff]:
        return self.storage.load_handoff(session_id)

    def build_codex_handoff_flow(self, session_id: str, *, current_request: Optional[str] = None) -> Optional[CodexHandoffFlow]:
        handoff = self.storage.load_handoff(session_id)
        memory = self.storage.load_durable_memory(session_id)
        if handoff is None or memory is None:
            return None
        return build_codex_handoff_flow(memory, handoff, current_request=current_request)

    def refresh_if_needed(
        self,
        session_id: str,
        items: List[Dict[str, Any]],
        *,
        current_request: str,
        repo_context: Optional[Dict[str, Any]] = None,
    ) -> SessionHandoff:
        if estimate_tokens(items) < settings.compactor_target_chunk_tokens:
            existing = self.load_latest_handoff(session_id)
            if existing is not None:
                return existing
        return self.compact_transcript(session_id, items, current_request=current_request, repo_context=repo_context)
