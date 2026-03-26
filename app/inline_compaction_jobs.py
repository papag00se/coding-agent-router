from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from threading import Condition, Lock, Thread
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_HEARTBEAT_SECONDS = 2.0
_RESULT_TTL_SECONDS = 300.0


def inline_compaction_request_key(payload: Dict[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop('stream', None)
    digest = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return f'inline_job_{digest[:24]}'


@dataclass
class InlineCompactionJob:
    key: str
    request_payload: Dict[str, Any]
    request: Any
    request_headers: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    events: List[Dict[str, Any]] = field(default_factory=list)
    condition: Condition = field(default_factory=Condition)
    completed: bool = False
    final_response: Optional[Dict[str, Any]] = None
    failure_message: Optional[str] = None
    last_progress: Dict[str, Any] = field(
        default_factory=lambda: {
            'type': 'progress',
            'stage': 'queued',
            'message': 'Queued local compaction.',
            'heartbeat': False,
        }
    )

    def append_event(self, event: Dict[str, Any]) -> None:
        with self.condition:
            self.events.append(event)
            self.updated_at = time.monotonic()
            if event.get('type') == 'progress':
                self.last_progress = dict(event)
            if event.get('type') == 'final':
                self.final_response = event.get('response')
                self.completed = True
            if event.get('type') == 'failed':
                self.failure_message = str(event.get('message') or 'inline compaction failed')
                self.completed = True
            self.condition.notify_all()

    def iter_events(self) -> Iterator[Dict[str, Any]]:
        index = 0
        while True:
            heartbeat: Optional[Dict[str, Any]] = None
            pending: List[Dict[str, Any]] = []
            completed = False
            with self.condition:
                while index >= len(self.events) and not self.completed:
                    notified = self.condition.wait(timeout=_HEARTBEAT_SECONDS)
                    if not notified and not self.completed:
                        heartbeat = dict(self.last_progress)
                        heartbeat['heartbeat'] = True
                        heartbeat['timestamp'] = time.time()
                        heartbeat['elapsed_seconds'] = int(max(1.0, time.monotonic() - self.created_at))
                        heartbeat['message'] = _heartbeat_message(heartbeat)
                        break
                while index < len(self.events):
                    pending.append(self.events[index])
                    index += 1
                completed = self.completed
            if heartbeat is not None:
                yield heartbeat
                continue
            for event in pending:
                yield event
            if completed:
                return

    def wait(self) -> Dict[str, Any]:
        with self.condition:
            while not self.completed:
                self.condition.wait()
            if self.final_response is not None:
                return self.final_response
            raise RuntimeError(self.failure_message or 'inline compaction failed')

    def is_expired(self, now: float) -> bool:
        return self.completed and (now - self.updated_at) > _RESULT_TTL_SECONDS


class InlineCompactionJobManager:
    def __init__(
        self,
        service: Any,
        *,
        fallback_callback: Optional[Callable[..., Iterator[Dict[str, Any]]]] = None,
    ) -> None:
        self.service = service
        self.fallback_callback = fallback_callback
        self._lock = Lock()
        self._jobs: Dict[str, InlineCompactionJob] = {}

    def get_or_create(
        self,
        payload: Dict[str, Any],
        request: Any,
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> tuple[InlineCompactionJob, bool]:
        key = inline_compaction_request_key(payload)
        with self._lock:
            self._cleanup_locked()
            existing = self._jobs.get(key)
            if existing is not None:
                if existing.completed and existing.failure_message is not None:
                    self._jobs.pop(key, None)
                else:
                    existing.updated_at = time.monotonic()
                    return existing, False
            job = InlineCompactionJob(
                key=key,
                request_payload=dict(payload),
                request=request,
                request_headers=dict(headers or {}),
            )
            self._jobs[key] = job
            worker = Thread(target=self._run_job, args=(job,), daemon=True)
            worker.start()
            return job, True

    def _cleanup_locked(self) -> None:
        now = time.monotonic()
        expired = [key for key, job in self._jobs.items() if job.is_expired(now)]
        for key in expired:
            self._jobs.pop(key, None)

    def _run_job(self, job: InlineCompactionJob) -> None:
        def progress(stage: str, **fields: Any) -> None:
            message = str(fields.get('message') or _default_progress_message(stage, fields))
            event = {'type': 'progress', 'stage': stage, 'message': message, **fields}
            job.append_event(event)

        try:
            progress('job_started', request_key=job.key)
            for event in self.service.stream_inline_compact_from_anthropic(job.request, progress_callback=progress):
                job.append_event(dict(event))
            if not job.completed:
                raise RuntimeError('inline compaction stream exited without a terminal event')
        except Exception as exc:
            logger.exception("inline compaction job failed key=%s", job.key)
            if self.fallback_callback is None:
                job.append_event({'type': 'failed', 'message': str(exc)})
                return
            try:
                progress('spark_fallback_started', request_key=job.key, error=str(exc))
                for event in self.fallback_callback(
                    job.request_payload,
                    job.request,
                    job.request_headers,
                    exc,
                    request_key=job.key,
                    progress_callback=progress,
                ):
                    job.append_event(dict(event))
                if not job.completed:
                    raise RuntimeError('inline compaction spark fallback exited without a terminal event')
            except Exception as fallback_exc:
                logger.exception("inline compaction spark fallback failed key=%s", job.key)
                job.append_event({'type': 'failed', 'message': str(fallback_exc)})


def _default_progress_message(stage: str, fields: Dict[str, Any]) -> str:
    if stage == 'job_started':
        return 'Starting local compaction.'
    if stage == 'normalize_completed':
        return 'Normalized transcript for compaction.'
    if stage == 'chunking_completed':
        skipped = int(fields.get('skipped_item_count') or 0)
        if skipped:
            return f"Prepared {fields.get('chunk_count', 0)} compaction chunk(s); preserved {skipped} oversize item(s) raw."
        return f"Prepared {fields.get('chunk_count', 0)} compaction chunk(s)."
    if stage == 'extract_chunk_started':
        return f"Extracting chunk {fields.get('chunk_index', '?')} of {fields.get('chunk_count', '?')}."
    if stage == 'extract_chunk_completed':
        return f"Finished chunk {fields.get('chunk_index', '?')} of {fields.get('chunk_count', '?')}."
    if stage == 'merge_completed':
        return 'Merged extracted compaction state.'
    if stage == 'refine_iteration_started':
        return f"Refining recent context {fields.get('iteration', '?')} of {fields.get('iteration_count', '?')}."
    if stage == 'refine_iteration_completed':
        return f"Finished refinement pass {fields.get('iteration', '?')} of {fields.get('iteration_count', '?')}."
    if stage == 'render_completed':
        return 'Rendering compacted continuation state.'
    if stage == 'inline_render_completed':
        return 'Finalizing compacted continuation.'
    if stage == 'spark_fallback_started':
        return 'Local compaction failed validation; retrying compaction with Spark.'
    return 'Local compaction in progress.'


def _heartbeat_message(event: Dict[str, Any]) -> str:
    elapsed = int(event.get('elapsed_seconds') or 0)
    base = str(event.get('message') or 'Local compaction in progress.').rstrip('.')
    if elapsed <= 0:
        return f"{base}."
    return f"{base}. Waiting on local model ({elapsed}s elapsed)."
