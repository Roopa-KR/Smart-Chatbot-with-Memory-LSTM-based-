"""SQLite-backed memory and conversation storage."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chatbot_memory.db"


@contextmanager
def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database() -> None:
    """Create the database and required tables if they do not exist."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_user_time ON conversation_log(user_id, created_at)"
        )


def store_memory(user_id: str, key: str, value: str) -> None:
    """Insert or update a memory value for a user."""

    initialize_database()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO user_memory (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (user_id, key, value, datetime.now(timezone.utc).isoformat()),
        )


def retrieve_memory(user_id: str, key: str) -> Optional[str]:
    """Fetch a stored memory value for a user."""

    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            "SELECT value FROM user_memory WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        row = cursor.fetchone()
        return row["value"] if row else None


def log_conversation(user_id: str, role: str, message: str) -> None:
    """Store a chat message for short-term context and auditing."""

    initialize_database()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversation_log (user_id, role, message, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, role, message, datetime.now(timezone.utc).isoformat()),
        )


def get_recent_messages(user_id: str, limit: int = 2) -> List[dict]:
    """Return the most recent messages for a user in chronological order."""

    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            SELECT role, message, created_at
            FROM conversation_log
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()

    return [
        {"role": row["role"], "message": row["message"], "created_at": row["created_at"]}
        for row in reversed(rows)
    ]
