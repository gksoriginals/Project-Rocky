from dataclasses import dataclass, field


@dataclass(slots=True)
class HistoryEntry:
    role: str
    content: str
    tool_name: str | None = None


@dataclass(slots=True)
class PromptContext:
    system_prompt: str
    dialogue: list[HistoryEntry] = field(default_factory=list)


def system_message(content: str):
    return HistoryEntry(role="system", content=content)


def user_message(content: str):
    return HistoryEntry(role="user", content=content)


def assistant_message(content: str):
    return HistoryEntry(role="assistant", content=content)


def tool_message(tool_name: str, content: str):
    return HistoryEntry(role="tool", content=content, tool_name=tool_name)
