"""SQLite is the source of truth. The FAISS index is a derived artifact that can
always be rebuilt from here, which is what makes interrupted runs cheap to resume."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    path      TEXT NOT NULL UNIQUE,
    mtime     REAL NOT NULL,
    size      INTEGER NOT NULL,
    -- embedding stored as raw float32 bytes; NULL means "discovered, not yet encoded"
    embedding BLOB,
    encoded_at REAL
);

CREATE INDEX IF NOT EXISTS idx_files_pending
    ON files(id) WHERE embedding IS NULL;
"""


class Db:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        # WAL lets a search run while an index is still writing.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def check_model_compatibility(self, model: str, dim: int) -> None:
        """Vectors from different encoders are not comparable. Refuse to mix them."""
        seen = self.get_meta("model")
        if seen is None:
            self.set_meta("model", model)
            self.set_meta("dim", str(dim))
            return
        if seen != model:
            raise RuntimeError(
                f"Index was built with {seen!r} but you are using {model!r}. "
                "Vectors from different models are not comparable. "
                "Run `vgrep reset` and re-index."
            )

    # -- files --------------------------------------------------------------

    def upsert_file(self, path: str, mtime: float, size: int) -> bool:
        """Register a file. Returns True if it needs (re-)encoding.

        A file needs work if it is new, or if mtime/size changed since we last saw it.
        Unchanged files are skipped entirely -- this is what makes re-indexing fast.
        """
        row = self.conn.execute(
            "SELECT mtime, size, embedding FROM files WHERE path=?", (path,)
        ).fetchone()

        if row is None:
            self.conn.execute(
                "INSERT INTO files(path, mtime, size) VALUES(?,?,?)", (path, mtime, size)
            )
            return True

        if row["mtime"] != mtime or row["size"] != size:
            self.conn.execute(
                "UPDATE files SET mtime=?, size=?, embedding=NULL, encoded_at=NULL WHERE path=?",
                (mtime, size, path),
            )
            return True

        return row["embedding"] is None

    def pending(self) -> Iterator[sqlite3.Row]:
        """Files discovered but not yet encoded."""
        yield from self.conn.execute(
            "SELECT id, path FROM files WHERE embedding IS NULL ORDER BY id"
        )

    def count_pending(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) c FROM files WHERE embedding IS NULL"
        ).fetchone()["c"]

    def save_embeddings(self, items: Iterable[tuple[int, np.ndarray]], now: float) -> None:
        self.conn.executemany(
            "UPDATE files SET embedding=?, encoded_at=? WHERE id=?",
            [(v.astype(np.float32).tobytes(), now, fid) for fid, v in items],
        )
        self.conn.commit()

    def drop_missing(self) -> int:
        """Remove rows whose file no longer exists on disk."""
        gone = [
            r["id"] for r in self.conn.execute("SELECT id, path FROM files")
            if not Path(r["path"]).exists()
        ]
        if gone:
            self.conn.executemany("DELETE FROM files WHERE id=?", [(i,) for i in gone])
            self.conn.commit()
        return len(gone)

    def all_embeddings(self, dim: int) -> tuple[np.ndarray, list[int]]:
        """Every stored vector, for rebuilding the FAISS index."""
        rows = self.conn.execute(
            "SELECT id, embedding FROM files WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
        if not rows:
            return np.zeros((0, dim), dtype=np.float32), []
        vecs = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return vecs, [r["id"] for r in rows]

    def paths_for(self, ids: Iterable[int]) -> dict[int, str]:
        ids = list(ids)
        if not ids:
            return {}
        q = f"SELECT id, path FROM files WHERE id IN ({','.join('?' * len(ids))})"
        return {r["id"]: r["path"] for r in self.conn.execute(q, ids)}

    def stats(self) -> dict[str, int]:
        c = self.conn.execute(
            "SELECT COUNT(*) total, COUNT(embedding) encoded FROM files"
        ).fetchone()
        return {"total": c["total"], "encoded": c["encoded"]}

    def close(self) -> None:
        self.conn.close()
