from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EpisodicCandidate:
    summary: str
    excerpt: str
    importance: int = 5
    tags: list[str] = field(default_factory=list)
    episode_type: str = "conversation"
    emotion: str = ""
    source_session_key: str = ""
    status: str = "resolved"


@dataclass(slots=True)
class SemanticCandidate:
    title: str
    content: str
    importance: int = 5
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    source_episode_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryWriteCandidateSet:
    episodic: EpisodicCandidate | None = None
    semantic: list[SemanticCandidate] = field(default_factory=list)


@dataclass(slots=True)
class MemoryPolicyConfig:
    episodic_min_importance: int = 7
    semantic_min_confidence: float = 0.7
    store_unresolved_episodes: bool = True


@dataclass(slots=True)
class MemoryWritePlan:
    episodic: EpisodicCandidate | None = None
    semantic: list[SemanticCandidate] = field(default_factory=list)


@dataclass(slots=True)
class MemoryWriteResult:
    episodic_written: int = 0
    semantic_written: int = 0

    @property
    def total_written(self) -> int:
        return self.episodic_written + self.semantic_written


class MemoryPolicy:
    def __init__(self, config: MemoryPolicyConfig | None = None):
        self.config = config or MemoryPolicyConfig()

    def evaluate(self, candidates: MemoryWriteCandidateSet) -> MemoryWritePlan:
        episodic = candidates.episodic
        approved_episode = None
        if episodic is not None and self.should_store_episode(episodic):
            approved_episode = episodic

        approved_semantic = [
            candidate
            for candidate in candidates.semantic
            if self.should_store_semantic(candidate)
        ]
        return MemoryWritePlan(episodic=approved_episode, semantic=approved_semantic)

    def should_store_episode(self, candidate: EpisodicCandidate) -> bool:
        if not candidate.summary.strip():
            return False
        if self.config.store_unresolved_episodes and candidate.status.strip().lower() != "resolved":
            return True
        return int(candidate.importance) >= int(self.config.episodic_min_importance)

    def should_store_semantic(self, candidate: SemanticCandidate) -> bool:
        if not candidate.title.strip() or not candidate.content.strip():
            return False
        return float(candidate.confidence) >= float(self.config.semantic_min_confidence)
