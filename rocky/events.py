from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal[
    "status_changed",
    "user_message",
    "reasoning_update",
    "assistant_delta",
    "assistant_message",
    "tool_call_started",
    "tool_call_result",
    "memory_snapshot_updated",
    "summary_created",
    "error",
]


@dataclass(slots=True)
class AgentEvent:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
