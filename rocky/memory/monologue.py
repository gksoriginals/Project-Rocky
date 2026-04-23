from __future__ import annotations

from dataclasses import dataclass, field

from rocky.memory.emotion import EmotionState


@dataclass(slots=True)
class MonologueEntry:
    thought: str
    turn_index: int
    emotion: EmotionState = EmotionState.neutral


class Monologue:
    def __init__(self, max_entries: int = 3):
        self.max_entries = max_entries
        self.entries: list[MonologueEntry] = []

    def add(
        self,
        thought: str,
        turn_index: int,
        emotion: EmotionState = EmotionState.neutral,
    ) -> MonologueEntry:
        entry = MonologueEntry(thought=thought.strip(), turn_index=turn_index, emotion=emotion)
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        return entry

    def latest(self) -> MonologueEntry | None:
        return self.entries[-1] if self.entries else None

    def build_section(self) -> str:
        if not self.entries:
            return ""
        lines = [f"[Turn {e.turn_index}] {e.thought}" for e in self.entries]
        return "\n".join(lines)

    def clear(self) -> None:
        self.entries = []
