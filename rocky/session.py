from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_TRACE_HISTORY_LIMIT = 8
DEFAULT_TRACE_ENTRY_LIMIT = 24


@dataclass(slots=True)
class SessionState:
    model_name: str
    provider_kind: str
    status: str = "idle"
    memory_integrity: int = 100
    tool_activity: str = ""
    tool_count: int = 0
    active_tool: str | None = None
    turn_index: int = 0
    last_reasoning: str = ""
    last_answer: str = ""
    notice: str = ""
    recent_dialogue: list[dict[str, Any]] = field(default_factory=list)
    current_trace: list[dict[str, Any]] = field(default_factory=list)
    trace_history: list[dict[str, Any]] = field(default_factory=list)
    trace_history_limit: int = DEFAULT_TRACE_HISTORY_LIMIT
    trace_entry_limit: int = DEFAULT_TRACE_ENTRY_LIMIT
    memory_snapshot: dict[str, Any] = field(default_factory=dict)
    tool_history: list[dict[str, Any]] = field(default_factory=list)

    def update_status(self, status: str) -> None:
        self.status = status

    def set_reasoning(self, text: str) -> None:
        self.last_reasoning = text or ""

    def set_answer(self, text: str) -> None:
        self.last_answer = text or ""

    def set_notice(self, text: str) -> None:
        self.notice = text or ""

    def set_tool_activity(self, text: str) -> None:
        self.tool_activity = text or ""

    def add_tool_event(self, event: dict[str, Any]) -> None:
        self.tool_history.append(event)

    def add_trace_entry(
        self,
        phase: str,
        summary: str,
        detail: str = "",
        turn_index: int | None = None,
        **metadata: Any,
    ) -> None:
        entry = {
            "phase": phase,
            "summary": summary or "",
            "detail": detail or "",
        }
        if turn_index is not None:
            entry["turn_index"] = turn_index
        if metadata:
            entry["metadata"] = dict(metadata)
        self.current_trace.append(entry)
        if self.trace_entry_limit > 0:
            self.current_trace = self.current_trace[-self.trace_entry_limit :]

    def clear_current_trace(self) -> None:
        self.current_trace = []

    def commit_trace_history(self) -> None:
        if not self.current_trace:
            return
        self.trace_history.append(
            {
                "turn_index": self.turn_index,
                "entries": list(self.current_trace),
            }
        )
        if self.trace_history_limit > 0:
            self.trace_history = self.trace_history[-self.trace_history_limit :]

    def set_memory_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.memory_snapshot = snapshot
        integrity = snapshot.get("integrity")
        if isinstance(integrity, int):
            self.memory_integrity = integrity
