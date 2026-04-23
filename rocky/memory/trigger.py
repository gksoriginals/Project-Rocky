from __future__ import annotations

from dataclasses import dataclass, field
import re

from rocky.conversation import HistoryEntry


@dataclass(slots=True)
class MemoryWriteDecision:
    should_write: bool
    reason: str = "none"
    priority: int = 0
    matched_signals: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryWritePolicyConfig:
    importance_phrase_bonus: int = 1
    repeated_topic_threshold: int = 2


class MemoryWritePolicy:
    _EXPLICIT_MEMORY_PATTERNS = (
        "remember this",
        "don't forget",
        "do not forget",
        "keep this in mind",
    )
    _UNRESOLVED_PATTERNS = (
        "later",
        "follow up",
        "follow-up",
        "not solved",
        "still need",
        "come back",
        "unfinished",
    )
    _PREFERENCE_PATTERNS = (
        "i prefer",
        "i like",
        "i want",
        "i need",
        "my goal",
        "we should",
    )
    _MILESTONE_PATTERNS = (
        "decided",
        "finalized",
        "completed",
        "finished",
        "milestone",
        "architecture",
        "policy",
    )

    def __init__(self, config: MemoryWritePolicyConfig | None = None):
        self.config = config or MemoryWritePolicyConfig()

    def evaluate(self, dialogue: list[HistoryEntry]) -> MemoryWriteDecision:
        if not dialogue:
            return MemoryWriteDecision(should_write=False)

        matched_signals: list[str] = []
        user_text = "\n".join(
            entry.content.strip()
            for entry in dialogue
            if entry.role == "user" and entry.content.strip()
        ).lower()

        if any(pattern in user_text for pattern in self._EXPLICIT_MEMORY_PATTERNS):
            matched_signals.append("explicit_memory_request")
        if any(pattern in user_text for pattern in self._UNRESOLVED_PATTERNS):
            matched_signals.append("unresolved_thread")
        if any(pattern in user_text for pattern in self._PREFERENCE_PATTERNS):
            matched_signals.append("durable_preference")
        if any(pattern in user_text for pattern in self._MILESTONE_PATTERNS):
            matched_signals.append("project_milestone")
        if self._has_repeated_topic(user_text):
            matched_signals.append("repeated_topic")

        if matched_signals:
            return MemoryWriteDecision(
                should_write=True,
                reason=matched_signals[0],
                priority=len(matched_signals),
                matched_signals=matched_signals,
            )

        return MemoryWriteDecision(should_write=False)

    def _has_repeated_topic(self, user_text: str) -> bool:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9']+", user_text)
            if len(token) > 4
        ]
        if not tokens:
            return False
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return any(count >= self.config.repeated_topic_threshold for count in counts.values())
