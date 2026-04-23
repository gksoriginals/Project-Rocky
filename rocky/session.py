from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rocky.tracing import TraceLog


@dataclass(slots=True)
class InteractionState:
    status: str = "idle"
    turn_index: int = 0
    last_reasoning: str = ""
    last_answer: str = ""
    notice: str = ""


@dataclass(slots=True)
class ToolingState:
    activity: str = ""
    count: int = 0
    active_tool: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class MemoryViewState:
    integrity: int = 100
    snapshot: dict[str, Any] = field(default_factory=dict)
    recent_dialogue: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SessionState:
    model_name: str
    provider_kind: str
    interaction: InteractionState = field(default_factory=InteractionState)
    tooling: ToolingState = field(default_factory=ToolingState)
    memory_view: MemoryViewState = field(default_factory=MemoryViewState)
    trace: TraceLog = field(default_factory=TraceLog)

    @property
    def status(self) -> str:
        return self.interaction.status

    @status.setter
    def status(self, value: str) -> None:
        self.interaction.status = value or "idle"

    @property
    def turn_index(self) -> int:
        return self.interaction.turn_index

    @turn_index.setter
    def turn_index(self, value: int) -> None:
        self.interaction.turn_index = int(value or 0)

    @property
    def last_reasoning(self) -> str:
        return self.interaction.last_reasoning

    @last_reasoning.setter
    def last_reasoning(self, value: str) -> None:
        self.interaction.last_reasoning = value or ""

    @property
    def last_answer(self) -> str:
        return self.interaction.last_answer

    @last_answer.setter
    def last_answer(self, value: str) -> None:
        self.interaction.last_answer = value or ""

    @property
    def notice(self) -> str:
        return self.interaction.notice

    @notice.setter
    def notice(self, value: str) -> None:
        self.interaction.notice = value or ""

    @property
    def tool_activity(self) -> str:
        return self.tooling.activity

    @tool_activity.setter
    def tool_activity(self, value: str) -> None:
        self.tooling.activity = value or ""

    @property
    def tool_count(self) -> int:
        return self.tooling.count

    @tool_count.setter
    def tool_count(self, value: int) -> None:
        self.tooling.count = int(value or 0)

    @property
    def active_tool(self) -> str | None:
        return self.tooling.active_tool

    @active_tool.setter
    def active_tool(self, value: str | None) -> None:
        self.tooling.active_tool = value or None

    @property
    def tool_history(self) -> list[dict[str, Any]]:
        return self.tooling.history

    @tool_history.setter
    def tool_history(self, value: list[dict[str, Any]]) -> None:
        self.tooling.history = [item for item in value if isinstance(item, dict)]

    @property
    def memory_integrity(self) -> int:
        return self.memory_view.integrity

    @memory_integrity.setter
    def memory_integrity(self, value: int) -> None:
        self.memory_view.integrity = int(value or 0)

    @property
    def memory_snapshot(self) -> dict[str, Any]:
        return self.memory_view.snapshot

    @memory_snapshot.setter
    def memory_snapshot(self, value: dict[str, Any]) -> None:
        self.memory_view.snapshot = value if isinstance(value, dict) else {}

    @property
    def recent_dialogue(self) -> list[dict[str, Any]]:
        return self.memory_view.recent_dialogue

    @recent_dialogue.setter
    def recent_dialogue(self, value: list[dict[str, Any]]) -> None:
        self.memory_view.recent_dialogue = [item for item in value if isinstance(item, dict)]

    @property
    def current_trace(self) -> list[dict[str, Any]]:
        return self.trace.current()

    @current_trace.setter
    def current_trace(self, value: list[dict[str, Any]]) -> None:
        self.trace.restore(current=value, history=self.trace.history())

    @property
    def trace_history(self) -> list[dict[str, Any]]:
        return self.trace.history()

    @trace_history.setter
    def trace_history(self, value: list[dict[str, Any]]) -> None:
        self.trace.restore(current=self.trace.current(), history=value)

    @property
    def trace_history_limit(self) -> int:
        return self.trace.history_limit

    @trace_history_limit.setter
    def trace_history_limit(self, value: int) -> None:
        self.trace.history_limit = int(value or 0)

    @property
    def trace_entry_limit(self) -> int:
        return self.trace.entry_limit

    @trace_entry_limit.setter
    def trace_entry_limit(self, value: int) -> None:
        self.trace.entry_limit = int(value or 0)

    def update_status(self, status: str) -> None:
        self.status = status

    def set_reasoning(self, text: str) -> None:
        self.last_reasoning = text

    def set_answer(self, text: str) -> None:
        self.last_answer = text

    def set_notice(self, text: str) -> None:
        self.notice = text

    def set_tool_activity(self, text: str) -> None:
        self.tool_activity = text

    def set_tool_count(self, count: int) -> None:
        self.tool_count = count

    def add_tool_event(self, event: dict[str, Any]) -> None:
        if isinstance(event, dict):
            self.tooling.history.append(event)

    def add_trace_entry(
        self,
        phase: str,
        summary: str,
        detail: str = "",
        turn_index: int | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        entry = self.trace.add_entry(
            phase=phase,
            summary=summary,
            detail=detail,
            turn_index=turn_index,
            **metadata,
        )
        return entry.to_dict()

    def clear_current_trace(self) -> None:
        self.trace.clear_current()

    def commit_trace_history(self) -> None:
        self.trace.commit_current(self.turn_index)

    def set_memory_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.memory_snapshot = snapshot
        integrity = self.memory_snapshot.get("integrity")
        if isinstance(integrity, int):
            self.memory_integrity = integrity

    def sync_memory_view(
        self,
        snapshot: dict[str, Any],
        recent_dialogue: list[dict[str, Any]],
    ) -> None:
        self.recent_dialogue = recent_dialogue
        self.set_memory_snapshot(snapshot)

    def advance_turn(self) -> None:
        self.turn_index += 1

    def record_active_tool(self, tool_name: str | None) -> None:
        self.active_tool = tool_name

    def snapshot_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "last_reasoning": self.last_reasoning,
            "last_answer": self.last_answer,
            "notice": self.notice,
            "tool_activity": self.tool_activity,
            "turn_index": self.turn_index,
            "active_tool": self.active_tool,
            "current_trace": list(self.current_trace),
            "trace_history": list(self.trace_history),
            "tool_history": list(self.tool_history),
            "memory_integrity": self.memory_integrity,
        }

    def restore_payload(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            return
        self.status = str(state.get("status") or "idle")
        self.last_reasoning = str(state.get("last_reasoning") or "")
        self.last_answer = str(state.get("last_answer") or "")
        self.notice = str(state.get("notice") or "")
        self.tool_activity = str(state.get("tool_activity") or "")
        self.turn_index = int(state.get("turn_index") or 0)
        self.active_tool = str(state.get("active_tool") or "") or None
        self.current_trace = state.get("current_trace") or []
        self.trace_history = state.get("trace_history") or []
        self.tool_history = state.get("tool_history") or []
        self.memory_integrity = int(state.get("memory_integrity") or self.memory_integrity)

    def reset_runtime(self, notice: str = "Session reset.") -> None:
        self.status = "idle"
        self.tool_activity = ""
        self.tool_history = []
        self.active_tool = None
        self.turn_index = 0
        self.last_reasoning = ""
        self.last_answer = ""
        self.notice = notice
        self.recent_dialogue = []
        self.clear_current_trace()
        self.trace_history = []

    def export(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "provider_kind": self.provider_kind,
            "interaction": asdict(self.interaction),
            "tooling": asdict(self.tooling),
            "memory_view": asdict(self.memory_view),
            "trace": self.trace.export(),
        }
