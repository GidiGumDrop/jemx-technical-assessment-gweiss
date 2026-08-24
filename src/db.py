"""SQLite connection handling and the atomic build-and-swap used by ingestion."""

from __future__ import annotations

import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .config import DEFAULT_DB_PATH, SCHEMA_SQL


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection with foreign keys on and rows accessible by name."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables, indexes and views. Safe to run against an existing database."""
    conn.executescript(SCHEMA_SQL.read_text())
    conn.commit()


@contextmanager
def staged_database(target: Path, *, commit: bool = True) -> Iterator[Path]:
    """Build into a scratch file, then replace `target` only once the block succeeds.

    An export that fails validation part-way therefore leaves the live database
    exactly as it was, rather than half-overwritten. When `target` already exists
    it is copied first, so an export containing only a new week merges into what
    we already hold instead of replacing it.

    With ``commit=False`` the scratch file is always discarded, which is how a
    dry-run validates a new export against the data we already hold without
    touching it.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.name}.staging-{os.getpid()}")
    staging.unlink(missing_ok=True)

    if target.exists():
        shutil.copy2(target, staging)

    try:
        yield staging
    except BaseException:
        staging.unlink(missing_ok=True)
        raise

    if commit:
        os.replace(staging, target)
    else:
        staging.unlink(missing_ok=True)


def table_counts(conn: sqlite3.Connection, tables: list[str]) -> dict[str, int]:
    """Row count per table, for reporting after a load."""
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
        for table in tables
    }


def data_range(conn: sqlite3.Connection) -> dict[str, object]:
    """The reporting window derived from the data itself (see v_data_range)."""
    row = conn.execute("SELECT * FROM v_data_range").fetchone()
    return dict(row) if row is not None else {}
