from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HistoryEntry:
    role: str
    content: str
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptContext:
    system_prompt: str
    context: list[str] = field(default_factory=list)
    dialogue: list[HistoryEntry] = field(default_factory=list)


def system_message(content: str, **metadata):
    return HistoryEntry(role="system", content=content, metadata=metadata)


def user_message(content: str, **metadata):
    return HistoryEntry(role="user", content=content, metadata=metadata)


def assistant_message(content: str, **metadata):
    return HistoryEntry(role="assistant", content=content, metadata=metadata)


def tool_message(tool_name: str, content: str, **metadata):
    return HistoryEntry(role="tool", content=content, tool_name=tool_name, metadata=metadata)
