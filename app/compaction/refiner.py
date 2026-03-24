from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..clients.ollama_client import OllamaClient
from ..config import settings
from .models import MergedState, MergedStatePatch
from .prompts import REFINEMENT_SYSTEM_PROMPT, build_refinement_payload, estimate_refinement_request_tokens
from .structured_output import normalize_merged_state_patch_payload

logger = logging.getLogger(__name__)
_TOKEN_ESTIMATION_SLACK = 256
_MIN_REFINEMENT_RESPONSE_TOKENS = 512


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
        self.minimum_response_tokens = min(self.max_output_tokens, _MIN_REFINEMENT_RESPONSE_TOKENS)
        self.token_estimation_slack = _TOKEN_ESTIMATION_SLACK

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
        prompt_tokens = estimate_refinement_request_tokens(
            state,
            recent_raw_turns,
            current_request,
            repo_context,
            model=self.model,
        )
        response_token_budget = self._response_token_budget(prompt_tokens)
        if response_token_budget < self.minimum_response_tokens:
            raise ValueError(
                f"insufficient compaction refiner output budget: prompt_tokens={prompt_tokens} "
                f"available={response_token_budget} minimum={self.minimum_response_tokens}"
            )
        logger.warning(
            "compaction_llm_request stage=refine model=%s prompt_tokens=%s response_tokens=%s recent_turn_count=%s merged_chunk_count=%s",
            self.model,
            prompt_tokens,
            response_token_budget,
            len(recent_raw_turns),
            state.merged_chunk_count,
        )
        started_at = time.monotonic()
        raw = self.client.chat(
            self.model,
            [{"role": "user", "content": payload}],
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            max_tokens=response_token_budget,
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
        patch = MergedStatePatch.model_validate(normalize_merged_state_patch_payload(parsed))
        if not _patch_has_effect(patch):
            logger.warning(
                "compaction_refine_noop model=%s recent_turn_count=%s merged_chunk_count=%s",
                self.model,
                len(recent_raw_turns),
                state.merged_chunk_count,
            )
            return state
        return _apply_patch_to_state(state, patch)

    def _response_token_budget(self, prompt_tokens: int) -> int:
        available = self.num_ctx - prompt_tokens - self.token_estimation_slack
        if available < self.minimum_response_tokens:
            return 0
        return min(self.max_output_tokens, available)


def _patch_has_effect(patch: MergedStatePatch) -> bool:
    return bool(
        patch.objective_update
        or patch.repo_state_updates
        or patch.add_files_touched
        or patch.add_commands_run
        or patch.add_errors
        or patch.add_accepted_fixes
        or patch.add_rejected_ideas
        or patch.add_constraints
        or patch.add_environment_assumptions
        or patch.add_pending_todos
        or patch.add_unresolved_bugs
        or patch.add_test_status
        or patch.add_external_references
        or patch.latest_plan_update
    )


def _apply_patch_to_state(state: MergedState, patch: MergedStatePatch) -> MergedState:
    data = state.model_dump()
    if patch.objective_update:
        data["objective"] = patch.objective_update
    if patch.repo_state_updates:
        repo_state = dict(data.get("repo_state") or {})
        repo_state.update(patch.repo_state_updates)
        data["repo_state"] = repo_state
    _merge_unique(data, "files_touched", patch.add_files_touched)
    _merge_unique(data, "commands_run", patch.add_commands_run)
    _merge_unique(data, "errors", patch.add_errors)
    _merge_unique(data, "accepted_fixes", patch.add_accepted_fixes)
    _merge_unique(data, "rejected_ideas", patch.add_rejected_ideas)
    _merge_unique(data, "constraints", patch.add_constraints)
    _merge_unique(data, "environment_assumptions", patch.add_environment_assumptions)
    _merge_unique(data, "pending_todos", patch.add_pending_todos)
    _merge_unique(data, "unresolved_bugs", patch.add_unresolved_bugs)
    _merge_unique(data, "test_status", patch.add_test_status)
    _merge_unique(data, "external_references", patch.add_external_references)
    if patch.latest_plan_update:
        data["latest_plan"] = patch.latest_plan_update
    return MergedState.model_validate(data)


def _merge_unique(data: Dict[str, Any], field_name: str, additions: List[str]) -> None:
    if not additions:
        return
    existing = list(data.get(field_name) or [])
    for item in additions:
        if item not in existing:
            existing.append(item)
    data[field_name] = existing
