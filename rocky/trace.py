from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TraceEntry:
    phase: str
    summary: str
    detail: str = ""
    turn_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phase": self.phase,
            "summary": self.summary,
            "detail": self.detail,
        }
        if self.turn_index is not None:
            payload["turn_index"] = self.turn_index
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEntry":
        return cls(
            phase=str(data.get("phase") or "trace"),
            summary=str(data.get("summary") or ""),
            detail=str(data.get("detail") or ""),
            turn_index=data.get("turn_index") if isinstance(data.get("turn_index"), int) else None,
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class TurnTrace:
    entries: list[TraceEntry] = field(default_factory=list)

    def add(
        self,
        phase: str,
        summary: str,
        detail: str = "",
        turn_index: int | None = None,
        **metadata: Any,
    ) -> TraceEntry:
        entry = TraceEntry(
            phase=phase,
            summary=summary,
            detail=detail,
            turn_index=turn_index,
            metadata=metadata,
        )
        self.entries.append(entry)
        return entry

    def clear(self) -> None:
        self.entries = []

    def to_dicts(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.entries]

    @classmethod
    def from_dicts(cls, entries: list[dict[str, Any]]) -> "TurnTrace":
        trace = cls()
        trace.entries = [TraceEntry.from_dict(item) for item in entries if isinstance(item, dict)]
        return trace
