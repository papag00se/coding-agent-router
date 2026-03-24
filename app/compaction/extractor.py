from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from ..clients.ollama_client import OllamaClient
from ..config import settings
from .models import ChunkExtraction, TranscriptChunk
from .prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_payload, estimate_extraction_request_tokens
from .structured_output import normalize_chunk_extraction_payload

logger = logging.getLogger(__name__)
_TOKEN_ESTIMATION_SLACK = 256
_MIN_EXTRACTION_RESPONSE_TOKENS = 1024


class CompactionExtractor:
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
        self.minimum_response_tokens = min(self.max_output_tokens, _MIN_EXTRACTION_RESPONSE_TOKENS)
        self.token_estimation_slack = _TOKEN_ESTIMATION_SLACK

    def extract_chunk(self, chunk: TranscriptChunk, repo_context: Optional[Dict[str, Any]] = None) -> ChunkExtraction:
        payload = build_extraction_payload(chunk, repo_context)
        prompt_tokens = estimate_extraction_request_tokens(chunk, repo_context, model=self.model)
        response_token_budget = self._response_token_budget(prompt_tokens)
        if response_token_budget < self.minimum_response_tokens:
            raise ValueError(
                f"insufficient compaction extractor output budget: prompt_tokens={prompt_tokens} "
                f"available={response_token_budget} minimum={self.minimum_response_tokens}"
            )
        logger.warning(
            "compaction_llm_request stage=extract chunk_id=%s model=%s prompt_tokens=%s response_tokens=%s chunk_tokens=%s item_count=%s",
            chunk.chunk_id,
            self.model,
            prompt_tokens,
            response_token_budget,
            chunk.token_count,
            len(chunk.items),
        )
        started_at = time.monotonic()
        raw = self.client.chat(
            self.model,
            [{"role": "user", "content": payload}],
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            max_tokens=response_token_budget,
            system=EXTRACTION_SYSTEM_PROMPT,
            response_format="json",
        )
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        logger.warning(
            "compaction_llm_response stage=extract chunk_id=%s model=%s elapsed_ms=%s prompt_eval_count=%s eval_count=%s",
            chunk.chunk_id,
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
            raise ValueError(f"compaction extractor returned non-JSON output: {preview}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"compaction extractor returned non-object output: {type(parsed).__name__}")
        parsed = normalize_chunk_extraction_payload(parsed)
        parsed.setdefault("chunk_id", chunk.chunk_id)
        parsed.setdefault("source_token_count", chunk.token_count)
        return ChunkExtraction.model_validate(parsed)

    def _response_token_budget(self, prompt_tokens: int) -> int:
        available = self.num_ctx - prompt_tokens - self.token_estimation_slack
        if available < self.minimum_response_tokens:
            return 0
        return min(self.max_output_tokens, available)
