"""Initialize the SQLite database for the RAG application."""

from __future__ import annotations

from pathlib import Path
import sqlite3


SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DATABASE_PATH = Path(__file__).with_name("rag.db")


def initialize_database(database_path: Path = DATABASE_PATH) -> None:
    """Create the SQLite database schema."""

    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    finally:
        connection.close()


if __name__ == "__main__":  # pragma: no cover - manual bootstrap
    initialize_database()

