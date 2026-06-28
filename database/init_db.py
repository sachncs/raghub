"""Initialise the SQLite database for the RAG application.

Run directly to bootstrap a fresh database:

    python database/init_db.py

By default the database is created at ``database/rag.db`` next to
this file. Override the location by passing a path:

    python -c "from database.init_db import initialize_database; from pathlib import Path; initialize_database(Path('/tmp/rag.db'))"
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DATABASE_PATH = Path(__file__).with_name("rag.db")


def initialize_database(database_path: Path = DATABASE_PATH) -> None:
    """Create the SQLite database schema.

    Args:
        database_path: Destination file. Parent directories are
            created on demand. Default: ``database/rag.db``.
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        # ``executescript`` allows multiple statements (the schema
        # defines three tables in one file).
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    finally:
        connection.close()


if __name__ == "__main__":  # pragma: no cover - manual bootstrap
    initialize_database()