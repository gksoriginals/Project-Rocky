from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from rocky.conversation import HistoryEntry, PromptContext, user_message
from rocky.llm import LLM
from rocky.memory.db import MemoryDB
from rocky.memory.policy import (
    EpisodicCandidate,
    MemoryPolicy,
    MemoryWritePlan,
    MemoryWriteResult,
    MemoryWriteCandidateSet,
    SemanticCandidate,
)
from rocky.utils import load_text_file, parse_json_object, unique_strings

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

SUMMARY_SYSTEM_PROMPT = load_text_file(os.path.join(PROMPTS_DIR, "summary_system.txt")).strip()
EXTRACTION_SYSTEM_PROMPT = load_text_file(os.path.join(PROMPTS_DIR, "extraction_system.txt")).strip()
ROUTER_SYSTEM_PROMPT = load_text_file(os.path.join(PROMPTS_DIR, "memory_router_system.txt")).strip()
RECALL_SEMANTIC_SYSTEM_PROMPT = load_text_file(
    os.path.join(PROMPTS_DIR, "recall_semantic_system.txt")
).strip()
RECALL_EPISODIC_SYSTEM_PROMPT = load_text_file(
    os.path.join(PROMPTS_DIR, "recall_episodic_system.txt")
).strip()


@dataclass(slots=True)
class EpisodicEntry:
    summary: str
    excerpt: str
    importance: int = 5
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    episode_type: str = "conversation"
    emotion: str = ""
    source_session_key: str = ""
    status: str = "resolved"


@dataclass(slots=True)
class SemanticDocument:
    title: str
    content: str
    importance: int = 5
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    confidence: float = 0.5
    source_episode_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WorkingMemory:
    dialogue: list[HistoryEntry] = field(default_factory=list)

    def recent_dialogue(self, limit: int = 6) -> list[HistoryEntry]:
        return self.dialogue[-max(int(limit), 0):]


@dataclass(slots=True)
class MemoryStore:
    working: WorkingMemory = field(default_factory=WorkingMemory)
    episodic: "EpisodicMemory" = field(default_factory=lambda: EpisodicMemory())
    semantic: "SemanticMemory" = field(default_factory=lambda: SemanticMemory())


class EpisodicMemory:
    def __init__(self):
        self.entries: list[EpisodicEntry] = []

    def add(
        self,
        summary: str,
        excerpt: str,
        importance: int = 5,
        tags: list[str] | None = None,
        created_at: str = "",
        episode_type: str = "conversation",
        emotion: str = "",
        source_session_key: str = "",
        status: str = "resolved",
    ) -> EpisodicEntry | None:
        normalized = summary.strip()
        if not normalized:
            return None

        if any(entry.summary.strip().lower() == normalized.lower() for entry in self.entries):
            return None

        entry = EpisodicEntry(
            summary=normalized,
            excerpt=excerpt.strip(),
            importance=max(1, min(int(importance), 10)),
            tags=unique_strings(tags or []),
            created_at=created_at.strip(),
            episode_type=episode_type.strip() or "conversation",
            emotion=emotion.strip(),
            source_session_key=source_session_key.strip(),
            status=status.strip() or "resolved",
        )
        self.entries.append(entry)
        return entry


class SemanticMemory:
    def __init__(self):
        self.entries: list[SemanticDocument] = []

    def add_document(
        self,
        title: str,
        content: str,
        importance: int = 5,
        aliases: list[str] | None = None,
        tags: list[str] | None = None,
        created_at: str = "",
        confidence: float = 0.5,
        source_episode_ids: list[str] | None = None,
    ) -> SemanticDocument | None:
        normalized_title = title.strip()
        if not normalized_title:
            return None

        # Check for existing entry with the same title (case-insensitive)
        existing_index = next(
            (i for i, e in enumerate(self.entries) if e.title.strip().lower() == normalized_title.lower()),
            None
        )

        normalized_content = content.strip() or normalized_title
        entry = SemanticDocument(
            title=normalized_title,
            content=normalized_content,
            importance=max(1, min(int(importance), 10)),
            aliases=unique_strings(aliases or []),
            tags=unique_strings(tags or []),
            created_at=created_at.strip(),
            confidence=max(0.0, min(float(confidence), 1.0)),
            source_episode_ids=unique_strings(source_episode_ids or []),
        )

        if existing_index is not None:
            self.entries[existing_index] = entry
        else:
            self.entries.append(entry)

        return entry


    def index_entries(self, limit: int | None = None) -> list[dict[str, object]]:
        entries = self.entries if limit is None else self.entries[-max(int(limit), 0):]
        return [
            {
                "title": entry.title,
                "aliases": list(entry.aliases),
                "tags": list(entry.tags),
                "importance": entry.importance,
            }
            for entry in entries
        ]


class MemoryManager:
    def __init__(
        self,
        dialogue_window: int = 6,
        llm=None,
        db_path: str | None = None,
        session_key: str = "default",
    ):
        self.dialogue_window = dialogue_window
        self.llm = llm or LLM.build(
            model=os.getenv("MEMORY_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gemma4:e2b"))
        )
        self.db_path = db_path or os.getenv("ROCKY_MEMORY_DB", "rocky_memory.sqlite3")
        self.session_key = session_key
        self.db = MemoryDB(self.db_path)
        self.memory = MemoryStore()
        self.policy = MemoryPolicy()
        self._load_persisted_entries()

    @property
    def working(self) -> WorkingMemory:
        return self.memory.working

    @property
    def episodic(self) -> EpisodicMemory:
        return self.memory.episodic

    @property
    def semantic(self) -> SemanticMemory:
        return self.memory.semantic

    @property
    def dialogue(self) -> list[HistoryEntry]:
        return self.working.dialogue

    @dialogue.setter
    def dialogue(self, value: list[HistoryEntry]) -> None:
        self.working.dialogue = list(value)

    def append_user(self, content: str, **metadata):
        self.working.dialogue.append(HistoryEntry(role="user", content=content, metadata=metadata))

    def append_assistant(self, content: str, **metadata):
        self.working.dialogue.append(HistoryEntry(role="assistant", content=content, metadata=metadata))

    def append_tool(self, tool_name: str, content: str, **metadata):
        self.working.dialogue.append(
            HistoryEntry(role="tool", content=content, tool_name=tool_name, metadata=metadata)
        )

    def trim_dialogue(self):
        self.working.dialogue = self.working.dialogue[-self.dialogue_window:]

    def close(self):
        self.db.close()

    def snapshot(self) -> dict[str, object]:
        working_snapshot = self.working_memory_snapshot()
        return {
            "working": working_snapshot,
            "semantic": self.semantic_documents(),
            "episodic": self.episodic_summaries(),
            "recent_dialogue": working_snapshot["recent_dialogue"],
            "last_summary": self.last_summary(),
            "integrity": self.memory_integrity_score(),
        }

    def working_memory_snapshot(self, limit: int = 6) -> dict[str, object]:
        recent = self.recent_dialogue(limit=limit)
        return {
            "recent_dialogue": recent,
            "dialogue_window": self.dialogue_window,
            "active_items": len(self.dialogue),
        }

    def recent_dialogue(self, limit: int = 6) -> list[dict[str, object]]:
        entries = self.working.recent_dialogue(limit)
        return [
            {
                "role": entry.role,
                "content": entry.content,
                "tool_name": entry.tool_name,
            }
            for entry in entries
        ]

    def semantic_documents(self, limit: int = 12) -> list[dict[str, object]]:
        entries = self.semantic.entries[-max(int(limit), 0):]
        return [
            {
                "title": entry.title,
                "content": entry.content,
                "importance": entry.importance,
                "aliases": list(entry.aliases),
                "tags": list(entry.tags),
                "created_at": entry.created_at,
                "confidence": entry.confidence,
                "source_episode_ids": list(entry.source_episode_ids),
            }
            for entry in entries
        ]

    def episodic_summaries(self, limit: int = 12) -> list[dict[str, object]]:
        entries = self.episodic.entries[-max(int(limit), 0):]
        return [
            {
                "summary": entry.summary,
                "excerpt": entry.excerpt,
                "importance": entry.importance,
                "tags": list(entry.tags),
                "created_at": entry.created_at,
                "episode_type": entry.episode_type,
                "emotion": entry.emotion,
                "source_session_key": entry.source_session_key,
                "status": entry.status,
            }
            for entry in entries
        ]

    def last_summary(self) -> str | None:
        if not self.episodic.entries:
            return None
        return self.episodic.entries[-1].summary

    def memory_integrity_score(self) -> int:
        score = 100
        if not self.semantic.entries:
            score -= 3
        if not self.episodic.entries:
            score -= 3
        if not self.last_summary():
            score -= 2
        return max(0, min(100, score))

    def semantic_index(self) -> list[dict[str, object]]:
        return self.semantic.index_entries()

    def semantic_titles(self, limit: int | None = None) -> list[str]:
        entries = self.semantic.entries if limit is None else self.semantic.entries[-max(int(limit), 0):]
        return [entry.title for entry in entries if entry.title.strip()]

    def get_semantic_memory(self, title: str) -> SemanticDocument | None:
        normalized = title.strip().lower()
        if not normalized:
            return None

        for entry in self.semantic.entries:
            if entry.title.strip().lower() == normalized:
                return entry
            if any(alias.strip().lower() == normalized for alias in entry.aliases):
                return entry
        return None

    def episodic_summaries_text(self, limit: int | None = None) -> list[str]:
        entries = self.episodic.entries if limit is None else self.episodic.entries[-max(int(limit), 0):]
        return [entry.summary for entry in entries if entry.summary.strip()]

    def list_memory_titles(self, kind: str, limit: int | None = None) -> list[str]:
        normalized = kind.strip().lower()
        if normalized == "semantic":
            return self.semantic_titles(limit)
        if normalized == "episodic":
            return self.episodic_summaries_text(limit)
        return []

    def add_semantic_memory(
        self,
        title: str,
        content: str,
        importance: int = 5,
        aliases: list[str] | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.5,
        source_episode_ids: list[str] | None = None,
    ) -> SemanticDocument | None:
        entry = self.semantic.add_document(
            title=title,
            content=content,
            importance=importance,
            aliases=aliases,
            tags=tags,
            confidence=confidence,
            source_episode_ids=source_episode_ids,
        )
        if entry is None:
            return None
        self.db.persist_semantic_document(
            entry.title,
            entry.content,
            entry.importance,
            entry.aliases,
            entry.tags,
            confidence=entry.confidence,
            source_episode_ids=entry.source_episode_ids,
        )
        return entry

    def delete_memory(self, kind: str, selector: str) -> dict[str, object]:
        normalized_kind = kind.strip().lower()
        selector_text = selector.strip()
        if not selector_text:
            return {"deleted": False, "reason": "empty-selector"}

        if normalized_kind == "semantic":
            result = self._delete_semantic_memory(selector_text)
        elif normalized_kind == "episodic":
            result = self._delete_episodic_memory(selector_text)
        else:
            return {"deleted": False, "reason": "unknown-kind"}

        self._load_persisted_entries()
        return result

    def delete_all_memory(self, kind: str) -> dict[str, object]:
        normalized_kind = kind.strip().lower()
        if normalized_kind == "semantic":
            count = self.db.delete_all_semantic_documents()
            self.semantic.entries = []
            return {"deleted": bool(count), "kind": "semantic", "count": count}
        if normalized_kind == "episodic":
            count = self.db.delete_all_episodic_entries()
            self.episodic.entries = []
            return {"deleted": bool(count), "kind": "episodic", "count": count}
        if normalized_kind == "all":
            semantic_count = self.db.delete_all_semantic_documents()
            episodic_count = self.db.delete_all_episodic_entries()
            self.semantic.entries = []
            self.episodic.entries = []
            return {
                "deleted": bool(semantic_count or episodic_count),
                "kind": "all",
                "semantic_count": semantic_count,
                "episodic_count": episodic_count,
            }
        return {"deleted": False, "reason": "unknown-kind"}

    def _delete_semantic_memory(self, selector: str) -> dict[str, object]:
        entry = self._resolve_semantic_selector(selector)
        if entry is None:
            return {"deleted": False, "reason": "not-found", "kind": "semantic"}

        removed = self.db.delete_semantic_document(entry.title)
        if removed:
            self.semantic.entries = [
                item for item in self.semantic.entries if item.title.strip().lower() != entry.title.strip().lower()
            ]
        return {"deleted": bool(removed), "kind": "semantic", "title": entry.title}

    def _delete_episodic_memory(self, selector: str) -> dict[str, object]:
        entry = self._resolve_episodic_selector(selector)
        if entry is None:
            return {"deleted": False, "reason": "not-found", "kind": "episodic"}

        removed = self.db.delete_episodic_entry(entry.summary)
        if removed:
            self.episodic.entries = [
                item for item in self.episodic.entries if item.summary.strip().lower() != entry.summary.strip().lower()
            ]
        return {"deleted": bool(removed), "kind": "episodic", "summary": entry.summary}

    def _resolve_semantic_selector(self, selector: str) -> SemanticDocument | None:
        entries = self.semantic.entries
        resolved = self._resolve_selector_by_index_or_text(
            selector,
            entries,
            lambda entry: entry.title,
            lambda entry: entry.aliases,
        )
        return resolved

    def _resolve_episodic_selector(self, selector: str) -> EpisodicEntry | None:
        entries = self.episodic.entries
        resolved = self._resolve_selector_by_index_or_text(
            selector,
            entries,
            lambda entry: entry.summary,
            lambda entry: entry.tags,
        )
        return resolved

    def _resolve_selector_by_index_or_text(self, selector: str, entries, text_getter, alias_getter):
        cleaned = selector.strip()
        if not cleaned:
            return None

        if cleaned.isdigit():
            index = int(cleaned) - 1
            if 0 <= index < len(entries):
                return entries[index]
            return None

        normalized = cleaned.lower()
        exact_matches = [entry for entry in entries if text_getter(entry).strip().lower() == normalized]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return None

        alias_matches = [
            entry
            for entry in entries
            if normalized in text_getter(entry).strip().lower()
            or any(normalized in str(alias).strip().lower() for alias in alias_getter(entry))
        ]
        if len(alias_matches) == 1:
            return alias_matches[0]
        return None

    def build_memory_sections(
        self,
        query: str,
        routes: dict[str, bool] | None = None,
    ) -> tuple[str, str]:
        routes = routes or self.build_memory_routes(query)
        semantic_titles = self._select_relevant_semantic_titles(query) if routes["semantic"] else []
        episodic_summaries = self._select_relevant_episodic_summaries(query) if routes["episodic"] else []
        semantic_section = self._render_selected_semantic_section(semantic_titles)
        episodic_section = self._render_selected_episodic_section(episodic_summaries)
        return semantic_section, episodic_section

    def build_memory_load_report(
        self,
        query: str,
        routes: dict[str, bool] | None = None,
    ) -> dict[str, list[str]]:
        routes = routes or self.build_memory_routes(query)
        semantic_titles = self._select_relevant_semantic_titles(query) if routes["semantic"] else []
        episodic_summaries = self._select_relevant_episodic_summaries(query) if routes["episodic"] else []

        return {
            "semantic": semantic_titles,
            "episodic": episodic_summaries,
        }

    def build_memory_load_summary(
        self,
        query: str,
        routes: dict[str, bool] | None = None,
    ) -> dict[str, str]:
        report = self.build_memory_load_report(query, routes=routes)
        semantic_titles = report.get("semantic") or []
        episodic_summaries = report.get("episodic") or []
        semantic_text = ", ".join(semantic_titles) if semantic_titles else "none"
        episodic_text = ", ".join(episodic_summaries) if episodic_summaries else "none"
        return {
            "semantic": semantic_text,
            "episodic": episodic_text,
        }

    def build_context(self, query: str, routes: dict[str, bool] | None = None) -> list[str]:
        routes = routes or self.build_memory_routes(query)
        recalled = []
        if routes["semantic"]:
            recalled.extend(self._selected_semantic_blocks(self._select_relevant_semantic_titles(query)))
        if routes["episodic"]:
            recalled.extend(self._selected_episodic_blocks(self._select_relevant_episodic_ids(query)))
        return recalled

    def summarize_dialogue(self, dialogue: list[HistoryEntry]) -> str | None:
        if not dialogue:
            return None

        payload = json.dumps(
            {
                "dialogue": self._serialize_dialogue(dialogue),
            },
            ensure_ascii=False,
            indent=2,
        )
        summary = self._generate(SUMMARY_SYSTEM_PROMPT, payload, response_format="text").strip()
        return summary or None

    def learn(self, dialogue: list[HistoryEntry], summary: str | None = None) -> MemoryWriteResult:
        payload = json.dumps(
            {
                "summary": summary,
                "dialogue": self._serialize_dialogue(dialogue),
            },
            ensure_ascii=False,
            indent=2,
        )
        extracted = self._generate(EXTRACTION_SYSTEM_PROMPT, payload, response_format="json")
        candidates = self._build_write_candidates(extracted, dialogue=dialogue, summary=summary)
        plan = self.policy.evaluate(candidates)
        return self._persist_write_plan(plan)

    def _build_write_candidates(
        self,
        extracted: dict,
        dialogue: list[HistoryEntry],
        summary: str | None = None,
    ) -> MemoryWriteCandidateSet:
        episodic_summary = (extracted.get("episodic_summary") or summary or "").strip()
        semantic_facts = extracted.get("semantic_facts") or []
        importance = extracted.get("importance", 5)
        tags = extracted.get("tags") or []
        emotion = str(extracted.get("emotion") or "").strip()
        episode_type = str(extracted.get("episode_type") or "conversation").strip() or "conversation"
        status = str(extracted.get("status") or "resolved").strip() or "resolved"
        confidence = max(0.0, min(float(importance) / 10.0, 1.0))
        candidates = MemoryWriteCandidateSet()

        if episodic_summary:
            candidates.episodic = EpisodicCandidate(
                summary=episodic_summary,
                excerpt=self._build_excerpt(dialogue),
                importance=importance,
                tags=tags,
                episode_type=episode_type,
                emotion=emotion,
                source_session_key=self.session_key,
                status=status,
            )

        for fact in semantic_facts:
            if isinstance(fact, dict):
                title = str(fact.get("title") or "").strip()
                content = str(fact.get("content") or "").strip()
            else:
                title = str(fact).strip()
                content = (episodic_summary or summary or self._build_excerpt(dialogue)).strip()

            if not title:
                continue

            candidates.semantic.append(
                SemanticCandidate(
                    title=title,
                    content=content or title,
                    importance=importance,
                    tags=tags,
                    confidence=confidence,
                )
            )
        return candidates

    def _persist_write_plan(self, plan: MemoryWritePlan) -> MemoryWriteResult:
        result = MemoryWriteResult()
        if plan.episodic is not None:
            candidate = plan.episodic
            entry = self.episodic.add(
                summary=candidate.summary,
                excerpt=candidate.excerpt,
                importance=candidate.importance,
                tags=candidate.tags,
                episode_type=candidate.episode_type,
                emotion=candidate.emotion,
                source_session_key=candidate.source_session_key,
                status=candidate.status,
            )
            if entry is not None:
                self.db.persist_episodic_entry(
                    entry.summary,
                    entry.excerpt,
                    entry.importance,
                    entry.tags,
                    episode_type=entry.episode_type,
                    emotion=entry.emotion,
                    source_session_key=entry.source_session_key,
                    status=entry.status,
                )
                result.episodic_written += 1

        for candidate in plan.semantic:
            entry = self.semantic.add_document(
                title=candidate.title,
                content=candidate.content,
                importance=candidate.importance,
                aliases=candidate.aliases,
                tags=candidate.tags,
                confidence=candidate.confidence,
                source_episode_ids=candidate.source_episode_ids,
            )
            if entry is not None:
                self.db.persist_semantic_document(
                    entry.title,
                    entry.content,
                    entry.importance,
                    entry.aliases,
                    entry.tags,
                    confidence=entry.confidence,
                    source_episode_ids=entry.source_episode_ids,
                )
                result.semantic_written += 1
        return result

    def import_markdown_path(self, path: str | Path) -> list[SemanticDocument]:
        path = Path(path)
        if path.is_dir():
            imported: list[SemanticDocument] = []
            for file_path in sorted(path.rglob("*")):
                if file_path.suffix.lower() not in {".md", ".markdown"}:
                    continue
                imported.extend(self.import_markdown_file(file_path))
            return imported
        return self.import_markdown_file(path)

    def import_markdown_file(self, path: str | Path) -> list[SemanticDocument]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        entries = self._parse_markdown_documents(text, default_title=path.stem)
        imported: list[SemanticDocument] = []
        for title, content in entries:
            entry = self.semantic.add_document(title=title, content=content)
            if entry is None:
                continue
            self.db.persist_semantic_document(
                entry.title,
                entry.content,
                entry.importance,
                entry.aliases,
                entry.tags,
                confidence=entry.confidence,
                source_episode_ids=entry.source_episode_ids,
            )
            imported.append(entry)
        return imported

    def import_markdown_text(self, text: str, default_title: str = "Untitled") -> list[SemanticDocument]:
        entries = self._parse_markdown_documents(text, default_title=default_title)
        imported: list[SemanticDocument] = []
        for title, content in entries:
            entry = self.semantic.add_document(title=title, content=content)
            if entry is None:
                continue
            self.db.persist_semantic_document(
                entry.title,
                entry.content,
                entry.importance,
                entry.aliases,
                entry.tags,
                confidence=entry.confidence,
                source_episode_ids=entry.source_episode_ids,
            )
            imported.append(entry)
        return imported

    def _select_relevant_semantic_titles(self, query: str) -> list[str]:
        index_block = self.build_semantic_index_block()
        if not index_block.strip():
            return self._fallback_semantic_matches(query)

        # Include minimal context to help the LLM resolve pronouns/references
        recent = self.recent_dialogue(limit=3)
        context_str = "\n".join(f"{d['role']}: {d['content']}" for d in recent)
        payload = f"Context:\n{context_str}\n\nQuery: {query}" if context_str else query

        system_prompt = RECALL_SEMANTIC_SYSTEM_PROMPT.format(index_block=index_block)
        response = self._generate(system_prompt, payload, response_format="json")
        selected_titles = response.get("selected_titles", [])
        if not isinstance(selected_titles, list):
            selected_titles = []

        cleaned = [str(item) for item in selected_titles if str(item).strip()]
        fallback = self._fallback_semantic_matches(query)
        if fallback:
            for title in fallback:
                if title not in cleaned:
                    cleaned.append(title)
        return cleaned

    def build_memory_routes(self, query: str) -> dict[str, bool]:
        # Also include minimal context for routing
        recent = self.recent_dialogue(limit=2)
        context_str = "\n".join(f"{d['role']}: {d['content']}" for d in recent)
        payload = f"Context:\n{context_str}\n\nQuery: {query}" if context_str else query

        response = self._generate(ROUTER_SYSTEM_PROMPT, payload, response_format="json")
        semantic = bool(response.get("semantic"))
        episodic = bool(response.get("episodic"))
        semantic = semantic or bool(self._fallback_semantic_matches(query))
        return {"semantic": semantic, "episodic": episodic}

    def build_semantic_index_block(self) -> str:
        entries = self._build_semantic_index_entries()
        if not entries:
            return ""

        semantic_lines = []
        for entry in entries:
            title = str(entry.get("title") or "").strip()
            if not title:
                continue
            aliases = ", ".join(entry.get("aliases") or [])
            tags = ", ".join(entry.get("tags") or [])

            line = f"- title: {title}"
            if aliases:
                line += f" | aliases: {aliases}"

            if tags:
                line += f" | tags: {tags}"
            semantic_lines.append(line)

        return "Semantic index:\n" + "\n".join(semantic_lines)



    def _build_semantic_index_entries(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []

        for entry in self.semantic.entries:
            entries.append(
                {
                    "kind": "semantic",
                    "title": entry.title,
                    "content": entry.content,
                    "aliases": list(entry.aliases),
                    "tags": list(entry.tags),
                }
            )

        return entries

    def _build_semantic_topic_lookup(self) -> dict[str, dict[str, object]]:
        return {
            self._normalize_topic(entry["title"]): entry
            for entry in self._build_semantic_index_entries()
            if str(entry.get("title") or "").strip()
        }

    def _fallback_semantic_matches(self, query: str) -> list[str]:
        normalized_query = self._normalize_topic(query)
        if not normalized_query:
            return []

        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9']+", normalized_query)
            if len(token) > 3
        }
        if not query_tokens:
            return []

        matches: list[str] = []
        for entry in self.semantic.entries:
            candidate_texts = [entry.title, *entry.aliases, *entry.tags]
            for candidate in candidate_texts:
                normalized_candidate = self._normalize_topic(candidate)
                if not normalized_candidate:
                    continue
                candidate_tokens = {
                    token
                    for token in re.findall(r"[a-z0-9']+", normalized_candidate)
                    if len(token) > 3
                }
                if normalized_candidate in normalized_query or normalized_query in normalized_candidate:
                    matches.append(entry.title)
                    break
                if candidate_tokens & query_tokens:
                    matches.append(entry.title)
                    break
        return unique_strings(matches)

    def _select_relevant_episodic_ids(self, query: str) -> list[str]:
        candidate_block, candidate_lookup = self.build_episodic_candidate_block()
        if not candidate_block.strip():
            return []

        system_prompt = RECALL_EPISODIC_SYSTEM_PROMPT.format(candidate_block=candidate_block)
        response = self._generate(system_prompt, query, response_format="json")
        selected_ids = response.get("selected_ids", [])
        if not isinstance(selected_ids, list):
            return []

        cleaned_ids = []
        for candidate_id in selected_ids:
            normalized_id = str(candidate_id).strip()
            if not normalized_id or normalized_id not in candidate_lookup:
                continue
            cleaned_ids.append(normalized_id)
        return cleaned_ids

    def _normalize_topic(self, value: str) -> str:
        return value.strip().lower()

    def _render_selected_semantic_section(self, selected_topics: list[str]) -> str:
        selected = []
        topic_map = self._build_semantic_topic_lookup()
        for topic in selected_topics:
            entry = topic_map.get(self._normalize_topic(topic))
            if entry is None or entry["kind"] != "semantic":
                continue
            cleaned = self._clean_prompt_content(str(entry.get("content") or ""))
            if cleaned:
                selected.append(cleaned)
        return "\n\n".join(selected).strip()

    def build_episodic_candidate_block(self, limit: int = 6) -> tuple[str, dict[str, EpisodicEntry]]:
        entries = self._build_episodic_candidate_entries(limit=limit)
        if not entries:
            return "", {}

        candidate_lookup: dict[str, EpisodicEntry] = {}
        lines = []
        for entry in entries:
            candidate_id = str(entry.get("id") or "").strip()
            summary = str(entry.get("summary") or "").strip()
            excerpt = str(entry.get("excerpt") or "").strip()
            importance = entry.get("importance")
            tags = ", ".join(entry.get("tags") or [])
            if not candidate_id or not summary:
                continue
            candidate_lookup[candidate_id] = entry["entry"]  # type: ignore[index]
            line = f"- {candidate_id}: {summary}"
            if excerpt and excerpt.lower() != summary.lower():
                line += f" | {excerpt}"
            lines.append(line)

        if not lines:
            return "", {}
        return "Episodic candidate block:\n" + "\n".join(lines), candidate_lookup

    def _build_episodic_candidate_entries(self, limit: int = 12) -> list[dict[str, object]]:
        if not self.episodic.entries:
            return []

        max_candidates = max(int(limit), 0)
        recent_entries = list(enumerate(self.episodic.entries[-max_candidates:]))
        recent_offset = max(len(self.episodic.entries) - len(recent_entries), 0)
        recent_entries = [(index + recent_offset, entry) for index, entry in recent_entries]

        important_entries = sorted(
            enumerate(self.episodic.entries),
            key=lambda pair: (-pair[1].importance, pair[0]),
        )[:max_candidates]

        combined: list[tuple[int, EpisodicEntry]] = []
        seen: set[int] = set()
        for index, entry in recent_entries + important_entries:
            if index in seen:
                continue
            seen.add(index)
            combined.append((index, entry))

        candidates: list[dict[str, object]] = []
        for candidate_index, (_, entry) in enumerate(combined, start=1):
            candidates.append(
                {
                    "id": f"E{candidate_index}",
                    "entry": entry,
                    "summary": entry.summary,
                    "excerpt": entry.excerpt,
                    "importance": entry.importance,
                    "tags": list(entry.tags),
                }
            )
        return candidates

    def _selected_episodic_blocks(self, selected_ids: list[str]) -> list[str]:
        blocks = []
        if not selected_ids:
            return blocks

        candidate_lookup = self.build_episodic_candidate_block()[1]
        for candidate_id in selected_ids:
            entry = candidate_lookup.get(candidate_id)
            if entry is None:
                continue
            formatted = self._format_episodic_prompt_text(entry)
            if formatted:
                blocks.append(formatted)
        return blocks

    def _select_relevant_episodic_summaries(self, query: str) -> list[str]:
        candidate_block, candidate_lookup = self.build_episodic_candidate_block()
        if not candidate_block.strip():
            return []

        # Include minimal context
        recent = self.recent_dialogue(limit=3)
        context_str = "\n".join(f"{d['role']}: {d['content']}" for d in recent)
        payload = f"Context:\n{context_str}\n\nQuery: {query}" if context_str else query

        system_prompt = RECALL_EPISODIC_SYSTEM_PROMPT.format(candidate_block=candidate_block)
        response = self._generate(system_prompt, payload, response_format="json")

        selected_ids = response.get("selected_ids", [])
        if not isinstance(selected_ids, list):
            return []

        summaries = []
        for candidate_id in selected_ids:
            normalized_id = str(candidate_id).strip()
            entry = candidate_lookup.get(normalized_id)
            if entry is None:
                continue
            summary = entry.summary.strip()
            if summary:
                summaries.append(summary)
        return summaries

    def _render_selected_episodic_section(self, summaries: list[str]) -> str:
        selected = []
        for summary in summaries:
            entry = next(
                (item for item in self.episodic.entries if item.summary.strip().lower() == summary.strip().lower()),
                None,
            )
            if entry is None:
                continue
            formatted = self._format_episodic_prompt_text(entry)
            if formatted:
                selected.append(formatted)
        return "\n\n".join(selected).strip()

    def _selected_semantic_blocks(self, selected_topics: list[str]) -> list[str]:
        blocks = []
        topic_map = self._build_semantic_topic_lookup()
        for topic in selected_topics:
            entry = topic_map.get(self._normalize_topic(topic))
            if entry is None or entry["kind"] != "semantic":
                continue
            cleaned = self._clean_prompt_content(str(entry.get("content") or ""))
            if cleaned:
                blocks.append(cleaned)
        return blocks

    def _format_episodic_prompt_text(self, entry: EpisodicEntry) -> str:
        parts = [entry.summary.strip()]
        excerpt = entry.excerpt.strip()
        if excerpt and excerpt.lower() != entry.summary.strip().lower():
            parts.append(excerpt)
        return "\n\n".join(part for part in parts if part).strip()

    def _clean_prompt_content(self, content: str) -> str:
        lines = []
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            if stripped.startswith("#"):
                continue
            lines.append(stripped)
        while lines and lines[0] == "":
            lines.pop(0)
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines).strip()

    def _parse_markdown_documents(self, text: str, default_title: str) -> list[tuple[str, str]]:
        lines = text.splitlines()
        docs: list[tuple[str, str]] = []
        title = ""
        body_lines: list[str] = []

        def flush_current() -> None:
            nonlocal title, body_lines
            if title.strip():
                docs.append((title.strip(), "\n".join(body_lines).strip()))
            title = ""
            body_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if title:
                    body_lines.append("")
                continue

            if stripped.startswith("#") and not stripped.startswith("##"):
                flush_current()
                candidate = stripped.lstrip("#").strip()
                if candidate:
                    title = candidate
                continue

            if title:
                body_lines.append(line.rstrip())

        flush_current()

        if not docs and text.strip():
            docs.append((default_title, text.strip()))

        return docs

    def _generate(self, system_prompt: str, user_payload: str, response_format: str = "text") -> str | dict:
        prompt = PromptContext(
            system_prompt=system_prompt,
            context=[],
            dialogue=[user_message(user_payload)],
        )
        raw_response = self.llm.generate_raw(prompt)
        raw_text = str(raw_response.get("text") or "")

        if response_format == "json":
            return parse_json_object(raw_text)
        return raw_text

    def _serialize_dialogue(self, dialogue: list[HistoryEntry]) -> list[dict[str, str]]:
        serialized = []
        for entry in dialogue:
            item = {"role": entry.role, "content": entry.content}
            if entry.tool_name:
                item["tool_name"] = entry.tool_name
            serialized.append(item)
        return serialized

    def _build_excerpt(self, dialogue: list[HistoryEntry]) -> str:
        recent = dialogue[-min(len(dialogue), self.dialogue_window):]
        lines = []
        for entry in recent:
            if entry.role == "tool" and entry.tool_name:
                lines.append(f"tool {entry.tool_name}: {entry.content}")
            else:
                lines.append(f"{entry.role}: {entry.content}")
        return " | ".join(lines)

    def clear_memory_cache(self):
        """Clears in-memory cache and reloads from the database."""
        self.episodic.entries = []
        self.semantic.entries = []
        self._load_persisted_entries()

    def _load_persisted_entries(self):
        self.episodic.entries = []
        self.semantic.entries = []

        episodic_rows = self.db.load_episodic_entries()
        for row in episodic_rows:
            self.episodic.entries.append(
                EpisodicEntry(
                    summary=row["summary"],
                    excerpt=row["excerpt"],
                    importance=int(row["importance"]),
                    tags=self._parse_tags(row["tags_json"]),
                    created_at=str(row.get("created_at") or ""),
                    episode_type=str(row.get("episode_type") or "conversation"),
                    emotion=str(row.get("emotion") or ""),
                    source_session_key=str(row.get("source_session_key") or ""),
                    status=str(row.get("status") or "resolved"),
                )
            )

        semantic_rows = self.db.load_semantic_documents()
        seen_titles = set()
        for row in semantic_rows:
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            self.semantic.entries.append(
                SemanticDocument(
                    title=title,
                    content=str(row.get("content") or ""),
                    importance=int(row.get("importance") or 5),
                    aliases=self._parse_tags(row.get("aliases_json") or "[]"),
                    tags=self._parse_tags(row.get("tags_json") or "[]"),
                    created_at=str(row.get("created_at") or ""),
                    confidence=self._parse_confidence(row.get("confidence")),
                    source_episode_ids=self._parse_tags(row.get("source_episode_ids_json") or "[]"),
                )
            )


    def _parse_tags(self, raw: str) -> list[str]:
        try:
            parsed = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return unique_strings([str(item) for item in parsed])

    def _parse_confidence(self, raw: object) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(value, 1.0))
