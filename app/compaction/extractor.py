from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..clients.ollama_client import OllamaClient
from ..config import settings
from .models import ChunkExtraction, TranscriptChunk
from .prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_payload


class CompactionExtractor:
    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
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

    def extract_chunk(self, chunk: TranscriptChunk, repo_context: Optional[Dict[str, Any]] = None) -> ChunkExtraction:
        raw = self.client.chat(
            self.model,
            [{"role": "user", "content": build_extraction_payload(chunk, repo_context)}],
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            system=EXTRACTION_SYSTEM_PROMPT,
            response_format="json",
        )
        content = raw.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError as exc:
            preview = content[:500] if isinstance(content, str) else repr(content)
            raise ValueError(f"compaction extractor returned non-JSON output: {preview}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"compaction extractor returned non-object output: {type(parsed).__name__}")
        parsed.setdefault("chunk_id", chunk.chunk_id)
        parsed.setdefault("source_token_count", chunk.token_count)
        return ChunkExtraction.model_validate(parsed)
