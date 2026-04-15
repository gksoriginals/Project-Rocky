import json
import sqlite3


class MemoryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(database=db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                summary_lower TEXT NOT NULL UNIQUE,
                excerpt TEXT NOT NULL,
                importance INTEGER NOT NULL,
                tags_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                title_lower TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                importance INTEGER NOT NULL,
                aliases_json TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                state_json TEXT NOT NULL,
                transcript_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def close(self):
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None

    def load_episodic_entries(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT summary, excerpt, importance, tags_json
            FROM episodic_records
            ORDER BY id ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def load_semantic_entries(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT title, content, importance, aliases_json, tags_json
            FROM semantic_documents
            ORDER BY id ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def load_semantic_documents(self) -> list[dict]:
        return self.load_semantic_entries()

    def persist_episodic_entry(self, summary: str, excerpt: str, importance: int, tags: list[str]):
        self._conn.execute(
            """
            INSERT OR IGNORE INTO episodic_records (
                summary,
                summary_lower,
                excerpt,
                importance,
                tags_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                summary,
                summary.strip().lower(),
                excerpt,
                int(importance),
                json.dumps(tags, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def persist_semantic_document(
        self,
        title: str,
        content: str,
        importance: int,
        aliases: list[str],
        tags: list[str],
    ):
        self._conn.execute(
            """
            INSERT OR REPLACE INTO semantic_documents (
                title,
                title_lower,
                content,
                importance,
                aliases_json,
                tags_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                title.strip().lower(),
                content,
                int(importance),
                json.dumps(aliases, ensure_ascii=False),
                json.dumps(tags, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def delete_episodic_entry(self, summary: str) -> int:
        cursor = self._conn.execute(
            """
            DELETE FROM episodic_records
            WHERE summary_lower = ?
            """,
            (summary.strip().lower(),),
        )
        self._conn.commit()
        return cursor.rowcount or 0

    def delete_semantic_document(self, title: str) -> int:
        title_lower = title.strip().lower()
        cursor = self._conn.execute(
            """
            DELETE FROM semantic_documents
            WHERE title_lower = ?
            """,
            (title_lower,),
        )
        self._conn.commit()
        return cursor.rowcount or 0

    def delete_all_episodic_entries(self) -> int:
        cursor = self._conn.execute("DELETE FROM episodic_records")
        self._conn.commit()
        return cursor.rowcount or 0

    def delete_all_semantic_documents(self) -> int:
        semantic_cursor = self._conn.execute("DELETE FROM semantic_documents")
        self._conn.commit()
        return semantic_cursor.rowcount or 0

    def delete_all_session_snapshots(self) -> int:
        cursor = self._conn.execute("DELETE FROM session_snapshots")
        self._conn.commit()
        return cursor.rowcount or 0

    def persist_session_snapshot(self, session_key: str, state: dict, transcript: list[dict]):
        self._conn.execute(
            """
            INSERT INTO session_snapshots (
                session_key,
                state_json,
                transcript_json
            ) VALUES (?, ?, ?)
            """,
            (
                session_key,
                json.dumps(state, ensure_ascii=False),
                json.dumps(transcript, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def load_latest_session_snapshot(self, session_key: str | None = None) -> dict | None:
        query = """
            SELECT session_key, state_json, transcript_json, created_at
            FROM session_snapshots
        """
        params: tuple[str, ...] = ()
        if session_key:
            query += " WHERE session_key = ?"
            params = (session_key,)
        query += " ORDER BY id DESC LIMIT 1"
        row = self._conn.execute(query, params).fetchone()
        if row is None:
            return None
        try:
            state = json.loads(row["state_json"] or "{}")
        except json.JSONDecodeError:
            state = {}
        try:
            transcript = json.loads(row["transcript_json"] or "[]")
        except json.JSONDecodeError:
            transcript = []
        return {
            "session_key": row["session_key"],
            "state": state if isinstance(state, dict) else {},
            "transcript": transcript if isinstance(transcript, list) else [],
            "created_at": row["created_at"],
        }
