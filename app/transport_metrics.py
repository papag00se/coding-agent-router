from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from .config import settings
from .task_metrics import estimate_openai_tokens, estimate_tokens

logger = logging.getLogger(__name__)

_STATE_VERSION = 3
_DEFAULT_LOCAL_COMPACTION_REQUEST_TOKENS = 250_000
_MINI_SAVINGS_FACTOR = 1.0 - (1.0 / 3.3)
_SPARK_SAVINGS_FACTOR = 0.70

_STORES: Dict[str, "TransportMetricsStore"] = {}
_STORES_LOCK = Lock()


def record_metrics_event(log_path: Path, payload: Dict[str, Any]) -> None:
    get_transport_metrics_store(log_path).record_event(payload)


def transport_metrics_snapshot(log_path: Path) -> Dict[str, Any]:
    return get_transport_metrics_store(log_path).snapshot()


def clear_transport_metrics_caches() -> None:
    with _STORES_LOCK:
        _STORES.clear()


def get_transport_metrics_store(log_path: Path) -> "TransportMetricsStore":
    resolved_log_path = log_path.resolve()
    state_path = metrics_state_path_for_log(resolved_log_path)
    key = str(state_path)
    with _STORES_LOCK:
        existing = _STORES.get(key)
        if existing is not None:
            return existing
        store = TransportMetricsStore(log_path=resolved_log_path, state_path=state_path)
        _STORES[key] = store
        return store


def metrics_state_path_for_log(log_path: Path) -> Path:
    stem = log_path.stem
    if stem.endswith("_transport"):
        stem = stem[: -len("_transport")]
    return log_path.with_name(f"{stem}_metrics.json")


class TransportMetricsStore:
    def __init__(self, *, log_path: Path, state_path: Path) -> None:
        self.log_path = log_path
        self.state_path = state_path
        self._lock = Lock()
        self._pending_inline_request_tokens: Dict[str, list[int]] = {}
        self._pending_inline_request_order: list[tuple[str, int]] = []
        self._pending_internal_request_tokens: Dict[str, list[int]] = {}
        self._state = self._load_or_bootstrap()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def record_event(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            _apply_event(
                self._state,
                payload,
                pending_inline_request_tokens=self._pending_inline_request_tokens,
                pending_inline_request_order=self._pending_inline_request_order,
                pending_internal_request_tokens=self._pending_internal_request_tokens,
            )
            self._state["updated_at"] = _utc_now()
            self._write_state()

    def _load_or_bootstrap(self) -> Dict[str, Any]:
        default_state = _default_state(self.log_path, self.state_path)
        if self.state_path.exists():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    loaded_version = _coerce_int(loaded.get("version"))
                    if loaded_version == _STATE_VERSION:
                        return _deep_merge(default_state, loaded)
                    if self.log_path.exists():
                        logger.info(
                            "rebuilding transport metrics state from %s due to version change (%s -> %s)",
                            self.log_path,
                            loaded_version,
                            _STATE_VERSION,
                        )
                        bootstrapped = _bootstrap_state_from_log(self.log_path, self.state_path)
                        self._write_state_data(bootstrapped)
                        return bootstrapped
                    return _deep_merge(default_state, loaded)
            except Exception:
                logger.exception("failed to load transport metrics state from %s", self.state_path)
        bootstrapped = _bootstrap_state_from_log(self.log_path, self.state_path)
        self._write_state_data(bootstrapped)
        return bootstrapped

    def _write_state(self) -> None:
        self._write_state_data(self._state)

    def _write_state_data(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self.state_path)


def _bootstrap_state_from_log(log_path: Path, state_path: Path) -> Dict[str, Any]:
    state = _default_state(log_path, state_path)
    pending_inline_request_tokens: Dict[str, list[int]] = {}
    pending_inline_request_order: list[tuple[str, int]] = []
    pending_internal_request_tokens: Dict[str, list[int]] = {}
    if not log_path.exists():
        return state

    bootstrapped_events = 0
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            _apply_event(
                state,
                payload,
                pending_inline_request_tokens=pending_inline_request_tokens,
                pending_inline_request_order=pending_inline_request_order,
                pending_internal_request_tokens=pending_internal_request_tokens,
            )
            bootstrapped_events += 1

    state["source"]["bootstrapped_from_log"] = True
    state["source"]["bootstrapped_event_count"] = bootstrapped_events
    state["updated_at"] = _utc_now()
    return state


def _default_state(log_path: Path, state_path: Path) -> Dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "updated_at": _utc_now(),
        "source": {
            "transport_log_path": str(log_path),
            "metrics_state_path": str(state_path),
            "bootstrapped_from_log": False,
            "bootstrapped_event_count": 0,
        },
        "events_total": 0,
        "events_by_type": {},
        "paths": {},
        "estimated_token_savings": {
            "total": 0,
            "spark": 0,
            "mini": 0,
            "local": 0,
        },
        "responses": {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "status_counts": {},
            "by_upstream_model": {},
            "estimated_token_savings": {
                "total": 0,
                "spark": 0,
                "mini": 0,
            },
            "spark": {
                "eligible": 0,
                "expected": 0,
                "successful": 0,
                "failed": 0,
                "skipped_sampling": 0,
                "blocked_by_mini": 0,
                "by_category": {},
            },
            "mini": {
                "eligible": 0,
                "expected": 0,
                "successful": 0,
                "failed": 0,
                "by_category": {},
            },
        },
        "compaction": {
            "detected": 0,
            "started": 0,
            "reused": 0,
            "completed": 0,
            "failed": 0,
            "spark_fallbacks": 0,
            "estimated_token_savings": {
                "total": 0,
                "spark": 0,
                "local": 0,
            },
            "started_by_model": {},
            "failed_by_model": {},
            "spark_fallbacks_by_model": {},
            "completed_by_model": {},
            "completed_by_mode": {},
        },
        "internal_compact": {
            "started": 0,
            "completed": 0,
            "estimated_token_savings": {
                "total": 0,
                "local": 0,
            },
        },
        "unsupported": {
            "total": 0,
            "by_transport": {},
        },
    }


def _deep_merge(base: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in loaded.items():
        if key not in merged:
            merged[key] = value
            continue
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_event(
    state: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    pending_inline_request_tokens: Dict[str, list[int]],
    pending_inline_request_order: list[tuple[str, int]],
    pending_internal_request_tokens: Dict[str, list[int]],
) -> None:
    event = str(payload.get("event") or "").strip()
    if not event:
        return

    state["events_total"] += 1
    _increment_count(state["events_by_type"], event)

    path = _infer_path(payload)
    if path:
        _path_bucket(state, path)["events_total"] += 1

    if event == "responses_passthrough":
        _apply_responses_passthrough(state, payload, path=path or "/v1/responses")
        return

    if event == "inline_compaction_detected":
        state["compaction"]["detected"] += 1
        if path:
            _path_bucket(state, path)["inline_detected"] += 1
        _queue_pending_inline_tokens(
            pending_inline_request_tokens,
            pending_inline_request_order,
            str(payload.get("request_key") or "").strip(),
            _inline_compaction_request_tokens(payload),
        )
        return

    if event == "inline_compaction_job_started":
        state["compaction"]["started"] += 1
        if path:
            _path_bucket(state, path)["inline_started"] += 1
        _increment_count(
            state["compaction"]["started_by_model"],
            _compaction_local_model(payload),
        )
        return

    if event == "inline_compaction_job_reused":
        state["compaction"]["reused"] += 1
        if path:
            _path_bucket(state, path)["inline_reused"] += 1
        return

    if event == "inline_compaction_spark_fallback":
        state["compaction"]["spark_fallbacks"] += 1
        if path:
            _path_bucket(state, path)["inline_spark_fallbacks"] += 1
        _increment_count(
            state["compaction"]["spark_fallbacks_by_model"],
            str(payload.get("upstream_model") or settings.codex_spark_model),
        )
        return

    if event == "inline_compaction_completed":
        state["compaction"]["completed"] += 1
        if path:
            _path_bucket(state, path)["inline_completed"] += 1
        after_payload = payload.get("after_payload") or {}
        model = "unknown"
        if isinstance(after_payload, dict):
            model = str(after_payload.get("model") or "unknown")
            raw_backend = after_payload.get("raw_backend") or {}
            mode = "local_inline_compaction"
            if isinstance(raw_backend, dict):
                mode = str(raw_backend.get("mode") or mode)
            _increment_count(state["compaction"]["completed_by_mode"], mode)
        _increment_count(state["compaction"]["completed_by_model"], model)
        family = _inline_compaction_completion_family(payload)
        request_tokens = _consume_pending_inline_tokens(
            pending_inline_request_tokens,
            pending_inline_request_order,
            str(payload.get("request_key") or "").strip(),
            preferred=_inline_compaction_request_tokens(payload),
            default=(
                _DEFAULT_LOCAL_COMPACTION_REQUEST_TOKENS
                if family == "local"
                else 0
            ),
        )
        _record_estimated_token_savings(
            state,
            scope="compaction",
            family=family,
            request_tokens=request_tokens,
            path=path,
        )
        return

    if event == "inline_compaction_failed":
        state["compaction"]["failed"] += 1
        if path:
            _path_bucket(state, path)["inline_failed"] += 1
        _increment_count(
            state["compaction"]["failed_by_model"],
            _compaction_local_model(payload),
        )
        _consume_pending_inline_tokens(
            pending_inline_request_tokens,
            pending_inline_request_order,
            str(payload.get("request_key") or "").strip(),
            preferred=_inline_compaction_request_tokens(payload),
        )
        return

    if event == "internal_compact_start":
        state["internal_compact"]["started"] += 1
        if path:
            _path_bucket(state, path)["internal_compact_started"] += 1
        _queue_pending_tokens(
            pending_internal_request_tokens,
            str(payload.get("session_id") or "").strip(),
            _internal_compact_request_tokens(payload),
        )
        return

    if event == "internal_compact_completed":
        state["internal_compact"]["completed"] += 1
        if path:
            _path_bucket(state, path)["internal_compact_completed"] += 1
        request_tokens = _consume_pending_tokens(
            pending_internal_request_tokens,
            str(payload.get("session_id") or "").strip(),
            preferred=_internal_compact_request_tokens(payload),
        )
        _record_estimated_token_savings(
            state,
            scope="internal_compact",
            family="local",
            request_tokens=request_tokens,
            path=path,
        )
        return

    if event == "unsupported_transport":
        state["unsupported"]["total"] += 1
        _increment_count(state["unsupported"]["by_transport"], str(payload.get("transport") or "unknown"))
        if path:
            _path_bucket(state, path)["unsupported"] += 1


def _apply_responses_passthrough(state: Dict[str, Any], payload: Dict[str, Any], *, path: str) -> None:
    status = payload.get("status")
    successful = _is_success_status(status)
    request_tokens = _responses_request_tokens(payload)

    path_bucket = _path_bucket(state, path)
    path_bucket["passthrough_total"] += 1
    if successful:
        path_bucket["passthrough_successful"] += 1
    else:
        path_bucket["passthrough_failed"] += 1

    responses = state["responses"]
    responses["total"] += 1
    if successful:
        responses["successful"] += 1
    else:
        responses["failed"] += 1
    _increment_count(responses["status_counts"], str(status if status is not None else "unknown"))
    _increment_outcome_bucket(
        responses["by_upstream_model"],
        str(payload.get("upstream_model") or "unknown"),
        successful,
    )

    spark = responses["spark"]
    if payload.get("spark_eligible"):
        spark["eligible"] += 1
    if payload.get("spark_rewrite_applied"):
        spark["expected"] += 1
        _increment_count(spark["by_category"], str(payload.get("spark_category") or "unknown"))
        if successful:
            spark["successful"] += 1
            _record_estimated_token_savings(
                state,
                scope="responses",
                family="spark",
                request_tokens=request_tokens,
                path=path,
            )
        else:
            spark["failed"] += 1
    elif payload.get("spark_eligible") and payload.get("mini_rewrite_applied"):
        spark["blocked_by_mini"] += 1
    elif payload.get("spark_eligible"):
        spark["skipped_sampling"] += 1

    mini = responses["mini"]
    if payload.get("mini_eligible"):
        mini["eligible"] += 1
    if payload.get("mini_rewrite_applied"):
        mini["expected"] += 1
        _increment_count(mini["by_category"], str(payload.get("mini_category") or "unknown"))
        if successful:
            mini["successful"] += 1
            _record_estimated_token_savings(
                state,
                scope="responses",
                family="mini",
                request_tokens=request_tokens,
                path=path,
            )
        else:
            mini["failed"] += 1


def _record_estimated_token_savings(
    state: Dict[str, Any],
    *,
    scope: str,
    family: str,
    request_tokens: int,
    path: str | None,
) -> None:
    estimated_saved_tokens = _estimate_saved_tokens(request_tokens, family)
    if estimated_saved_tokens <= 0:
        return
    _increment_savings_bucket(state["estimated_token_savings"], family, estimated_saved_tokens)
    _increment_savings_bucket(state[scope]["estimated_token_savings"], family, estimated_saved_tokens)
    if path:
        _increment_path_savings(_path_bucket(state, path), family, estimated_saved_tokens)


def _path_bucket(state: Dict[str, Any], path: str) -> Dict[str, int]:
    bucket = state["paths"].get(path)
    if bucket is None:
        bucket = {
            "events_total": 0,
            "passthrough_total": 0,
            "passthrough_successful": 0,
            "passthrough_failed": 0,
            "inline_detected": 0,
            "inline_started": 0,
            "inline_reused": 0,
            "inline_completed": 0,
            "inline_failed": 0,
            "inline_spark_fallbacks": 0,
            "internal_compact_started": 0,
            "internal_compact_completed": 0,
            "unsupported": 0,
            "estimated_token_savings_total": 0,
            "estimated_token_savings_spark": 0,
            "estimated_token_savings_mini": 0,
            "estimated_token_savings_local": 0,
        }
        state["paths"][path] = bucket
    return bucket


def _increment_count(counter: Dict[str, Any], key: str, amount: int = 1) -> None:
    counter[key] = int(counter.get(key) or 0) + amount


def _increment_outcome_bucket(counter: Dict[str, Any], key: str, successful: bool) -> None:
    bucket = counter.get(key)
    if bucket is None:
        bucket = {"total": 0, "successful": 0, "failed": 0}
        counter[key] = bucket
    bucket["total"] += 1
    if successful:
        bucket["successful"] += 1
    else:
        bucket["failed"] += 1


def _increment_savings_bucket(bucket: Dict[str, Any], family: str, estimated_saved_tokens: int) -> None:
    bucket["total"] = int(bucket.get("total") or 0) + estimated_saved_tokens
    bucket[family] = int(bucket.get(family) or 0) + estimated_saved_tokens


def _increment_path_savings(bucket: Dict[str, int], family: str, estimated_saved_tokens: int) -> None:
    bucket["estimated_token_savings_total"] += estimated_saved_tokens
    family_key = f"estimated_token_savings_{family}"
    bucket[family_key] = int(bucket.get(family_key) or 0) + estimated_saved_tokens


def _compaction_local_model(payload: Dict[str, Any]) -> str:
    return str(payload.get("local_model") or settings.compactor_model or "unknown")


def _responses_request_tokens(payload: Dict[str, Any]) -> int:
    return _positive_int(payload.get("request_tokens")) or _positive_int(payload.get("spark_request_tokens"))


def _inline_compaction_request_tokens(payload: Dict[str, Any]) -> int:
    direct = _responses_request_tokens(payload)
    if direct > 0:
        return direct
    before_payload = payload.get("before_payload")
    if not isinstance(before_payload, dict):
        return 0
    model = str(before_payload.get("model") or payload.get("original_model") or "gpt-5.4")
    try:
        return max(0, estimate_openai_tokens(before_payload, model=model))
    except Exception:
        logger.exception("failed to estimate inline compaction request tokens")
        return 0


def _internal_compact_request_tokens(payload: Dict[str, Any]) -> int:
    direct = _positive_int(payload.get("request_tokens"))
    if direct > 0:
        return direct
    before_payload = payload.get("before_payload")
    if not isinstance(before_payload, dict):
        return 0
    try:
        return max(0, estimate_tokens(before_payload))
    except Exception:
        logger.exception("failed to estimate internal compact request tokens")
        return 0


def _queue_pending_tokens(pending: Dict[str, list[int]], key: str, request_tokens: int) -> None:
    if not key or request_tokens <= 0:
        return
    pending.setdefault(key, []).append(request_tokens)


def _queue_pending_inline_tokens(
    pending: Dict[str, list[int]],
    pending_order: list[tuple[str, int]],
    key: str,
    request_tokens: int,
) -> None:
    if request_tokens <= 0:
        return
    if key:
        pending.setdefault(key, []).append(request_tokens)
    pending_order.append((key, request_tokens))


def _consume_pending_tokens(pending: Dict[str, list[int]], key: str, *, preferred: int = 0) -> int:
    consumed = 0
    if key:
        queued = pending.get(key)
        if queued:
            consumed = queued.pop(0)
            if not queued:
                pending.pop(key, None)
    if preferred > 0:
        return preferred
    return consumed


def _consume_pending_inline_tokens(
    pending: Dict[str, list[int]],
    pending_order: list[tuple[str, int]],
    key: str,
    *,
    preferred: int = 0,
    default: int = 0,
) -> int:
    consumed = _pop_pending_inline_tokens(pending, pending_order, key)
    if preferred > 0:
        return preferred
    if consumed > 0:
        return consumed
    return default


def _pop_pending_inline_tokens(
    pending: Dict[str, list[int]],
    pending_order: list[tuple[str, int]],
    key: str,
) -> int:
    if key:
        queued = pending.get(key)
        if queued:
            consumed = queued.pop(0)
            if not queued:
                pending.pop(key, None)
            _remove_pending_order_entry(pending_order, key, consumed)
            return consumed
    if pending_order:
        queued_key, consumed = pending_order.pop(0)
        _discard_pending_token(pending, queued_key, consumed)
        return consumed
    return 0


def _remove_pending_order_entry(pending_order: list[tuple[str, int]], key: str, request_tokens: int) -> None:
    for index, (queued_key, queued_tokens) in enumerate(pending_order):
        if queued_key == key and queued_tokens == request_tokens:
            pending_order.pop(index)
            return


def _discard_pending_token(pending: Dict[str, list[int]], key: str, request_tokens: int) -> None:
    if not key:
        return
    queued = pending.get(key)
    if not queued:
        return
    try:
        queued.remove(request_tokens)
    except ValueError:
        return
    if not queued:
        pending.pop(key, None)


def _inline_compaction_completion_family(payload: Dict[str, Any]) -> str:
    after_payload = payload.get("after_payload")
    if isinstance(after_payload, dict):
        raw_backend = after_payload.get("raw_backend")
        if isinstance(raw_backend, dict):
            mode = str(raw_backend.get("mode") or "")
            if "spark" in mode:
                return "spark"
        model = str(after_payload.get("model") or "")
        if model == settings.codex_spark_model:
            return "spark"
    return "local"


def _estimate_saved_tokens(request_tokens: int, family: str) -> int:
    if request_tokens <= 0:
        return 0
    factor = {
        "spark": _SPARK_SAVINGS_FACTOR,
        "mini": _MINI_SAVINGS_FACTOR,
        "local": 1.0,
    }.get(family)
    if factor is None:
        return 0
    return int(round(request_tokens * factor))


def _infer_path(payload: Dict[str, Any]) -> str | None:
    explicit = payload.get("path")
    if isinstance(explicit, str) and explicit:
        return _normalize_metrics_path(explicit)

    event = str(payload.get("event") or "")
    if event.startswith("inline_compaction_"):
        return "/v1/responses"
    if event.startswith("internal_compact_"):
        return "/internal/compact"
    if event == "unsupported_transport":
        transport = payload.get("transport")
        return {
            "v1_messages": "/v1/messages",
            "chat_completions": "/v1/chat/completions",
            "ollama_chat": "/api/chat",
        }.get(str(transport or ""))
    return None


def _normalize_metrics_path(path: str) -> str:
    return "/v1/responses" if path == "/responses" else path


def _positive_int(value: Any) -> int:
    try:
        numeric = int(value)
    except Exception:
        return 0
    return numeric if numeric > 0 else 0


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _is_success_status(value: Any) -> bool:
    try:
        return int(value) < 400
    except Exception:
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
