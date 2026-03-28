from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Tuple

import requests

logger = logging.getLogger(__name__)

_STATE_VERSION = 1
_DEFAULT_PROBE_SECONDS = 60.0
_MAX_PROBE_SECONDS = 900.0
_RESET_HEADER_NAMES = (
    "retry-after",
    "x-ratelimit-reset",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)
_RESET_BODY_KEYS = {
    "blocked_until",
    "reset_after",
    "reset_after_seconds",
    "reset_at",
    "reset_time",
    "retry_after",
    "retry_after_seconds",
}
_QUOTA_ERROR_KEYWORDS = (
    "insufficient_quota",
    "quota exceeded",
    "rate limit",
    "rate_limit_exceeded",
    "too many requests",
    "usage limit",
)


def spark_quota_state_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / "spark_quota.json"


def is_spark_quota_error(response: requests.Response) -> bool:
    status_code = _coerce_int(getattr(response, "status_code", None))
    if status_code == 429:
        return True
    body = _response_text_preview(response).lower()
    return any(keyword in body for keyword in _QUOTA_ERROR_KEYWORDS)


class SparkQuotaState:
    def __init__(self, state_path: Path) -> None:
        self.state_path = Path(state_path)
        self._lock = Lock()
        self._state = self._default_state()
        self._load_locked()

    def snapshot(self, *, now: float | None = None) -> Dict[str, Any]:
        current = _now_epoch(now)
        with self._lock:
            self._clear_expired_locked(current)
            return self._snapshot_locked(current)

    def is_blocked(self, *, now: float | None = None) -> bool:
        current = _now_epoch(now)
        with self._lock:
            return self._clear_expired_locked(current)

    def mark_exhausted(
        self,
        response: requests.Response | None,
        *,
        model: str | None = None,
        now: float | None = None,
        error: str | None = None,
    ) -> Dict[str, Any]:
        current = _now_epoch(now)
        with self._lock:
            reset_at, reset_reason = (None, None)
            if response is not None:
                reset_at, reset_reason = _extract_reset_at(response, now=current)
            probe_interval = _coerce_float(self._state.get("probe_interval_seconds")) or _DEFAULT_PROBE_SECONDS
            if reset_at is None:
                reset_at = current + probe_interval
                probe_interval = min(probe_interval * 2.0, _MAX_PROBE_SECONDS)
                reset_reason = reset_reason or error or "probe_interval"
            else:
                probe_interval = _DEFAULT_PROBE_SECONDS

            existing_blocked_until = _coerce_float(self._state.get("blocked_until"))
            blocked_until = max(value for value in (current, existing_blocked_until, reset_at) if value is not None)
            preview = None if response is None else _response_text_preview(response)
            if preview is None and error is not None:
                preview = error
            self._state.update(
                {
                    "blocked_until": blocked_until,
                    "probe_interval_seconds": probe_interval,
                    "last_error_at": current,
                    "last_error_status": None if response is None else _coerce_int(getattr(response, "status_code", None)),
                    "last_error_model": model or None,
                    "last_error_reason": reset_reason or error or "unknown",
                    "last_error_preview": preview,
                    "updated_at": current,
                }
            )
            self._write_locked()
            return self._snapshot_locked(current)

    def mark_success(self, *, model: str | None = None, now: float | None = None) -> Dict[str, Any]:
        current = _now_epoch(now)
        with self._lock:
            self._state.update(
                {
                    "blocked_until": None,
                    "probe_interval_seconds": _DEFAULT_PROBE_SECONDS,
                    "last_success_at": current,
                    "last_success_model": model or None,
                    "updated_at": current,
                }
            )
            self._write_locked()
            return self._snapshot_locked(current)

    def _default_state(self) -> Dict[str, Any]:
        now = _utc_now()
        return {
            "version": _STATE_VERSION,
            "blocked_until": None,
            "probe_interval_seconds": _DEFAULT_PROBE_SECONDS,
            "last_error_at": None,
            "last_error_status": None,
            "last_error_model": None,
            "last_error_reason": None,
            "last_error_preview": None,
            "last_success_at": None,
            "last_success_model": None,
            "last_cleared_at": None,
            "updated_at": now,
        }

    def _load_locked(self) -> None:
        if not self.state_path.exists():
            return
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("failed to load spark quota state from %s", self.state_path)
            return
        if not isinstance(loaded, dict):
            return
        loaded_version = _coerce_int(loaded.get("version"))
        if loaded_version and loaded_version != _STATE_VERSION:
            logger.info(
                "spark quota state version mismatch at %s (%s -> %s); reusing known fields",
                self.state_path,
                loaded_version,
                _STATE_VERSION,
            )
        for key in self._state:
            if key in loaded:
                self._state[key] = loaded[key]

    def _clear_expired_locked(self, now: float) -> bool:
        blocked_until = _coerce_float(self._state.get("blocked_until"))
        if blocked_until is None or blocked_until > now:
            return blocked_until is not None and blocked_until > now
        self._state.update(
            {
                "blocked_until": None,
                "last_cleared_at": now,
                "updated_at": now,
            }
        )
        self._write_locked()
        return False

    def _snapshot_locked(self, now: float) -> Dict[str, Any]:
        snapshot = dict(self._state)
        blocked_until = _coerce_float(snapshot.get("blocked_until"))
        snapshot["blocked"] = blocked_until is not None and blocked_until > now
        snapshot["remaining_seconds"] = max(0.0, blocked_until - now) if blocked_until is not None else 0.0
        snapshot["state_path"] = str(self.state_path)
        return snapshot

    def _write_locked(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(self.state_path)
        except Exception:
            logger.exception("failed to write spark quota state to %s", self.state_path)


def _extract_reset_at(response: requests.Response, *, now: float) -> Tuple[float | None, str | None]:
    headers = {str(key).lower(): value for key, value in getattr(response, "headers", {}).items()}
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        parsed = _parse_reset_hint(retry_after, now)
        if parsed is not None:
            return parsed, "retry-after"

    candidates: list[tuple[float, str]] = []
    for header_name in _RESET_HEADER_NAMES:
        if header_name == "retry-after":
            continue
        value = headers.get(header_name)
        if value is None:
            continue
        for part in str(value).split(","):
            parsed = _parse_reset_hint(part, now)
            if parsed is not None:
                candidates.append((parsed, header_name))

    body = _response_body_json(response)
    if body is not None:
        candidates.extend(_collect_reset_candidates(body, now))

    if not candidates:
        return None, None

    future_candidates = [item for item in candidates if item[0] > now]
    chosen = max(future_candidates or candidates, key=lambda item: item[0])
    return chosen


def _collect_reset_candidates(value: Any, now: float) -> list[tuple[float, str]]:
    candidates: list[tuple[float, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in _RESET_BODY_KEYS:
                parsed = _parse_reset_hint(child, now)
                if parsed is not None:
                    candidates.append((parsed, normalized_key))
                continue
            candidates.extend(_collect_reset_candidates(child, now))
    elif isinstance(value, list):
        for child in value:
            candidates.extend(_collect_reset_candidates(child, now))
    return candidates


def _parse_reset_hint(value: Any, now: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _parse_numeric_reset_hint(float(value), now)

    text = str(value).strip()
    if not text:
        return None
    try:
        return _parse_numeric_reset_hint(float(text), now)
    except ValueError:
        pass

    parsed: datetime | None = None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _parse_numeric_reset_hint(value: float, now: float) -> float | None:
    if value <= 0:
        return None
    if value >= 10_000_000_000:
        return value / 1000.0
    if value >= 1_000_000_000:
        return value
    return now + value


def _response_body_json(response: requests.Response) -> Dict[str, Any] | None:
    text = _response_text_preview(response, limit=None)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _response_text_preview(response: requests.Response, *, limit: int | None = 500) -> str:
    content = getattr(response, "content", b"")
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace")
    else:
        text = str(content)
    text = text.strip().replace("\n", " ")
    if limit is None:
        return text
    return text[:limit]


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _now_epoch(now: float | None = None) -> float:
    if now is not None:
        return float(now)
    return datetime.now(tz=timezone.utc).timestamp()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
