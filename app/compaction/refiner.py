from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..clients.ollama_client import OllamaClient
from ..config import settings
from .models import MergedState
from .prompts import REFINEMENT_SYSTEM_PROMPT, build_refinement_payload, estimate_refinement_request_tokens
from .structured_output import normalize_merged_state_payload

logger = logging.getLogger(__name__)


class CompactionRefiner:
    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        self.client = client or OllamaClient(
            settings.compactor_ollama_base_url,
            (settings.ollama_connect_timeout_seconds, settings.compactor_timeout_seconds),
            pool_connections=settings.ollama_pool_connections,
            pool_maxsize=settings.ollama_pool_maxsize,
        )
        self.model = model or settings.compactor_model
        self.temperature = settings.compactor_temperature if temperature is None else temperature
        self.num_ctx = settings.compactor_num_ctx if num_ctx is None else num_ctx
        self.max_output_tokens = (
            settings.compactor_response_headroom_tokens if max_output_tokens is None else max_output_tokens
        )

    def refine_state(
        self,
        state: MergedState,
        recent_raw_turns: List[Dict[str, Any]],
        *,
        current_request: str,
        repo_context: Optional[Dict[str, Any]] = None,
    ) -> MergedState:
        if not recent_raw_turns:
            return state
        payload = build_refinement_payload(state, recent_raw_turns, current_request, repo_context)
        prompt_tokens = estimate_refinement_request_tokens(state, recent_raw_turns, current_request, repo_context)
        logger.warning(
            "compaction_llm_request stage=refine model=%s prompt_tokens=%s recent_turn_count=%s merged_chunk_count=%s",
            self.model,
            prompt_tokens,
            len(recent_raw_turns),
            state.merged_chunk_count,
        )
        started_at = time.monotonic()
        raw = self.client.chat(
            self.model,
            [{"role": "user", "content": payload}],
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            max_tokens=self.max_output_tokens,
            system=REFINEMENT_SYSTEM_PROMPT,
            response_format="json",
        )
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        logger.warning(
            "compaction_llm_response stage=refine model=%s elapsed_ms=%s prompt_eval_count=%s eval_count=%s",
            self.model,
            elapsed_ms,
            raw.get("prompt_eval_count"),
            raw.get("eval_count"),
        )
        content = raw.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError as exc:
            preview = content[:500] if isinstance(content, str) else repr(content)
            raise ValueError(f"compaction refiner returned non-JSON output: {preview}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"compaction refiner returned non-object output: {type(parsed).__name__}")
        parsed = normalize_merged_state_payload(parsed)
        parsed.setdefault("merged_chunk_count", state.merged_chunk_count)
        return MergedState.model_validate(parsed)
