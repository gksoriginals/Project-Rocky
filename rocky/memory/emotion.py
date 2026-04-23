from __future__ import annotations

from enum import Enum


class EmotionState(str, Enum):
    neutral = "neutral"
    curious = "curious"
    excited = "excited"
    satisfied = "satisfied"
    confused = "confused"
    concerned = "concerned"

    @classmethod
    def parse(cls, value: str) -> "EmotionState":
        normalized = (value or "").strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            return cls.neutral


class EmotionFSM:
    def __init__(self, state: EmotionState = EmotionState.neutral):
        self._state = state

    def transition(self, new_state: EmotionState) -> None:
        self._state = new_state

    def current(self) -> EmotionState:
        return self._state

    def clear(self) -> None:
        self._state = EmotionState.neutral
