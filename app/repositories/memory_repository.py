from __future__ import annotations

from app.models.domain import MemoryItem
from app.repositories.database import Database


class MemoryRepository:
    def __init__(self, database: Database):
        self.database = database

    def add_session_message(
        self, session_id: str, customer_id: str, role: str, content: str
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO session_messages(session_id, customer_id, role, content)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, customer_id, role, content),
            )

    def get_session_messages(
        self, session_id: str, customer_id: str, limit: int
    ) -> list[dict[str, str]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM session_messages
                WHERE session_id = ? AND customer_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, customer_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def add_customer_memory(self, item: MemoryItem) -> int:
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO customer_memories(customer_id, memory_type, content, importance)
                VALUES (?, ?, ?, ?)
                """,
                (item.customer_id, item.memory_type, item.content, item.importance),
            )
            return int(cursor.lastrowid)

    def get_customer_memories(self, customer_id: str, limit: int) -> list[MemoryItem]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id AS memory_id, customer_id, memory_type, content, importance, created_at
                FROM customer_memories
                WHERE customer_id = ?
                ORDER BY importance DESC, id DESC LIMIT ?
                """,
                (customer_id, limit),
            ).fetchall()
        return [MemoryItem.model_validate(dict(row)) for row in rows]
