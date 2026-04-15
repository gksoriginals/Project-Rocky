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


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "ingest":
        if len(args) < 2:
            print("Usage: python rocky.py ingest <markdown-file-or-directory> [more paths...]")
            return 1
        return _run_ingest(args[1:])

    if args and args[0] == "memory" and len(args) > 1 and args[1] == "list":
        return _run_memory_list(args[2:])

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
