from __future__ import annotations

from dataclasses import dataclass

from rocky.conversation import HistoryEntry


@dataclass(slots=True)
class CompactionDecision:
    should_compact: bool
    reason: str = "none"


@dataclass(slots=True)
class CompactionConfig:
    # Approximate character budget for dialogue before trimming.
    # Rough proxy for token count: 1 token ≈ 4 chars.
    max_dialogue_chars: int = 8000


class CompactionTrigger:
    def __init__(self, config: CompactionConfig | None = None):
        self.config = config or CompactionConfig()

    def evaluate(self, dialogue: list[HistoryEntry]) -> CompactionDecision:
        if not dialogue:
            return CompactionDecision(should_compact=False)

        total_chars = sum(len(entry.content) for entry in dialogue)
        if total_chars >= self.config.max_dialogue_chars:
            return CompactionDecision(should_compact=True, reason="context_limit")

        return CompactionDecision(should_compact=False)
