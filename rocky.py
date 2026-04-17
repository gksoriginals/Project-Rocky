from __future__ import annotations

import os
import sys

from rocky.memory.db import MemoryDB


def _memory_db_path() -> str:
    return os.getenv("ROCKY_MEMORY_DB", "rocky_memory.sqlite3")


def _load_semantic_titles(limit: int | None = None) -> list[str]:
    db = MemoryDB(_memory_db_path())
    try:
        rows = db.load_semantic_documents()
    finally:
        db.close()

    titles = [str(row.get("title", "")).strip() for row in rows if str(row.get("title", "")).strip()]
    if limit is not None:
        return titles[: max(int(limit), 0)]
    return titles


def _load_semantic_memory(title: str) -> dict | None:
    db = MemoryDB(_memory_db_path())
    try:
        return db.load_semantic_document_by_title(title)
    finally:
        db.close()


def _run_ingest(paths: list[str]) -> int:
    from rocky.agent import RockyAgent
    from rocky.config import MODEL_NAME

    agent = RockyAgent(model=MODEL_NAME)
    imported = []
    for path in paths:
        imported.extend(agent.memory_manager.import_markdown_path(path))
    agent.memory_manager.close()

    if imported:
        print(f"Imported {len(imported)} semantic memory document(s).")
        for item in imported:
            print(f"- {item.title}")
    else:
        print("No semantic memory documents were imported.")
    return 0


def _run_memory_list(args: list[str]) -> int:
    limit: int | None = None
    if args:
        try:
            limit = int(args[0])
        except ValueError:
            print("Usage: python rocky.py memory list [limit]")
            return 1

    titles = _load_semantic_titles(limit)
    if not titles:
        print("No semantic memories found.")
        return 0

    print(f"Semantic memories ({len(titles)}):")
    for index, title in enumerate(titles, 1):
        print(f"{index}. {title}")
    return 0


def _run_memory_search(args: list[str]) -> int:
    if not args:
        print("Usage: python rocky.py memory search <title>")
        return 1

    title = " ".join(args).strip()
    record = _load_semantic_memory(title)
    if record is None:
        print(f"No semantic memory found for: {title}")
        return 0

    aliases = ", ".join(_normalize_list(record.get("aliases_json"))) or "none"
    tags = ", ".join(_normalize_list(record.get("tags_json"))) or "none"
    print(f"Title: {record.get('title', '').strip()}")
    print(f"Content: {record.get('content', '').strip()}")
    print(f"Importance: {record.get('importance', '')}")
    print(f"Aliases: {aliases}")
    print(f"Tags: {tags}")
    return 0


def _normalize_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if not isinstance(raw, str) or not raw.strip():
        return []
    import json

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "ingest":
        if len(args) < 2:
            print("Usage: python rocky.py ingest <markdown-file-or-directory> [more paths...]")
            return 1
        return _run_ingest(args[1:])

    if args and args[0] == "memory" and len(args) > 1 and args[1] == "list":
        return _run_memory_list(args[2:])

    if args and args[0] == "memory" and len(args) > 1 and args[1] == "search":
        return _run_memory_search(args[2:])

    from rocky.agent import RockyAgent
    from rocky.config import MODEL_NAME
    from rocky.tui.app import RockyTUI

    agent = RockyAgent(model=MODEL_NAME)
    if sys.stdin.isatty() and sys.stdout.isatty():
        RockyTUI(agent).run()
    else:
        agent.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
