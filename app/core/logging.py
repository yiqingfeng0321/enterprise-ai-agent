import json
import logging
from datetime import UTC, datetime
from typing import Any

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def audit_event(event: str, **payload: Any) -> None:
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        **payload,
    }
    logging.getLogger("audit").info(json.dumps(record, default=str, sort_keys=True))
