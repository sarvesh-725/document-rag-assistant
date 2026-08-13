"""Server-Sent Events formatting helpers."""

import json
from typing import Any


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    """Format one typed SSE frame, separated from the next by a blank line."""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )
