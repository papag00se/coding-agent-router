from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from .config import settings
from .transport_metrics import record_metrics_event

logger = logging.getLogger(__name__)


def record_transport_event(log_path: Path, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    try:
        record_metrics_event(log_path, payload)
    except Exception:
        logger.exception("failed to update transport metrics for %s", event)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    logger.warning("compaction_transport %s", json.dumps(payload, ensure_ascii=False))


def compaction_payload_fields(*, before: Any = None, after: Any = None) -> Dict[str, Any]:
    if not settings.log_compaction_payloads:
        return {}

    fields: Dict[str, Any] = {}
    if before is not None:
        fields["before_payload"] = before
    if after is not None:
        fields["after_payload"] = after
    return fields
