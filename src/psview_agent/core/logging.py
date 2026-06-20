"""JSON logging helpers."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from psview_agent.core.middleware import get_request_id


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter using the standard library."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        for key in (
            "method",
            "route",
            "status_code",
            "duration_ms",
            "graph_node",
            "provider",
            "model_name",
            "structured_output_mode",
            "provider_request_id",
            "retrieval_candidate_count",
            "retrieved_fact_ids",
            "selected_action",
            "revision_used",
            "fallback_used",
            "error_category",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str) -> None:
    """Configure root logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
