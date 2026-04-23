from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_TRACE_HISTORY_LIMIT = 8
DEFAULT_TRACE_ENTRY_LIMIT = 24


@dataclass(slots=True)
class TraceEntry:
    phase: str
    summary: str = ""
    detail: str = ""
    turn_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
    def from_dict(cls, payload: dict[str, Any]) -> "TraceEntry":
        if not isinstance(payload, dict):
            return cls(phase="trace")
        metadata = payload.get("metadata") or {}
        return cls(
            phase=str(payload.get("phase") or "trace"),
            summary=str(payload.get("summary") or ""),
            detail=str(payload.get("detail") or ""),
            turn_index=int(payload["turn_index"]) if payload.get("turn_index") is not None else None,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )


@dataclass(slots=True)
class TraceFrame:
    turn_index: int
    entries: list[TraceEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceFrame":
        if not isinstance(payload, dict):
            return cls(turn_index=0)
        raw_entries = payload.get("entries") or []
        entries = [
            TraceEntry.from_dict(item)
            for item in raw_entries
            if isinstance(item, dict)
        ]
        return cls(
            turn_index=int(payload.get("turn_index") or 0),
            entries=entries,
        )


@dataclass(slots=True)
class TraceLog:
    history_limit: int = DEFAULT_TRACE_HISTORY_LIMIT
    entry_limit: int = DEFAULT_TRACE_ENTRY_LIMIT
    current_entries: list[TraceEntry] = field(default_factory=list)
    history_frames: list[TraceFrame] = field(default_factory=list)

    def add_entry(
        self,
        phase: str,
        summary: str,
        detail: str = "",
        turn_index: int | None = None,
        **metadata: Any,
    ) -> TraceEntry:
        entry = TraceEntry(
            phase=phase or "trace",
            summary=summary or "",
            detail=detail or "",
            turn_index=turn_index,
            metadata=dict(metadata) if metadata else {},
        )
        self.current_entries.append(entry)
        if self.entry_limit > 0:
            self.current_entries = self.current_entries[-self.entry_limit :]
        return entry

    def clear_current(self) -> None:
        self.current_entries = []

    def commit_current(self, turn_index: int) -> None:
        if not self.current_entries:
            return
        self.history_frames.append(
            TraceFrame(turn_index=turn_index, entries=list(self.current_entries))
        )
        if self.history_limit > 0:
            self.history_frames = self.history_frames[-self.history_limit :]

    def restore(
        self,
        current: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.current_entries = [
            TraceEntry.from_dict(item)
            for item in (current or [])
            if isinstance(item, dict)
        ]
        self.history_frames = [
            TraceFrame.from_dict(item)
            for item in (history or [])
            if isinstance(item, dict)
        ]

    def current(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.current_entries]

    def history(self) -> list[dict[str, Any]]:
        return [frame.to_dict() for frame in self.history_frames]

    def export(self) -> dict[str, Any]:
        return {
            "history_limit": self.history_limit,
            "entry_limit": self.entry_limit,
            "current": self.current(),
            "history": self.history(),
        }
