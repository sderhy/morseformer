"""SQLite QSO logbook — a self-contained service seam.

This is a fully working store, not a stub, so the future "log this
callsign" / "export logbook" features only need UI wiring, not new
persistence logic. It deliberately has no Qt dependency: it can be
unit-tested and reused from the CLI or a script.

Schema (table ``qso``)::

    id          INTEGER PK
    logged_at   TEXT     ISO-8601 UTC timestamp
    callsign    TEXT     normalised upper-case call
    country     TEXT     DXCC entity guess, nullable
    excerpt     TEXT     surrounding transcript text, nullable
    notes       TEXT     free-form, nullable
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qso (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    callsign  TEXT NOT NULL,
    country   TEXT,
    excerpt   TEXT,
    notes     TEXT
);
CREATE INDEX IF NOT EXISTS idx_qso_callsign ON qso(callsign);
"""


@dataclass(frozen=True)
class QsoEntry:
    id: int
    logged_at: str
    callsign: str
    country: str | None
    excerpt: str | None
    notes: str | None


class Logbook:
    """A QSO logbook persisted to an SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(
        self,
        callsign: str,
        *,
        country: str | None = None,
        excerpt: str | None = None,
        notes: str | None = None,
        logged_at: str | None = None,
    ) -> int:
        ts = logged_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = self._conn.execute(
            "INSERT INTO qso (logged_at, callsign, country, excerpt, notes)"
            " VALUES (?, ?, ?, ?, ?)",
            (ts, callsign.upper(), country, excerpt, notes),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def all(self, *, limit: int | None = None) -> list[QsoEntry]:
        sql = "SELECT * FROM qso ORDER BY id DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql).fetchall()
        return [
            QsoEntry(
                id=r["id"],
                logged_at=r["logged_at"],
                callsign=r["callsign"],
                country=r["country"],
                excerpt=r["excerpt"],
                notes=r["notes"],
            )
            for r in rows
        ]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM qso").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
