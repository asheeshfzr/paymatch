# app/db_index.py
import sqlite3
import os
import re
from typing import List, Dict, Tuple

TRIGRAM_PATTERN = re.compile(r'(?=(.{3}))')  # sliding window for 3-grams


def _normalize_for_index(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


class SimpleTrigramIndex:
    """
    Simple SQLite-backed trigram-like index:
    - For each name, we store its trigrams into a table (name_id, trigram)
    - To query, compute query trigrams and count hits per name_id, returning top scorers.
    This simulates pg_trgm candidate generation without requiring Postgres.
    """

    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS names (id TEXT PRIMARY KEY, name TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS trigrams (name_id TEXT, trigram TEXT)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trigrams_trigram ON trigrams(trigram)")
        self.conn.commit()

    def add(self, name_id: str, name: str):
        norm = _normalize_for_index(name)
        trigrams = self._trigrams(norm)
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO names (id, name) VALUES (?, ?)", (name_id, name))
        # delete old trigrams
        c.execute("DELETE FROM trigrams WHERE name_id = ?", (name_id,))
        rows = [(name_id, tg) for tg in trigrams]
        if rows:
            c.executemany("INSERT INTO trigrams (name_id, trigram) VALUES (?, ?)", rows)
        self.conn.commit()

    def _trigrams(self, s: str) -> List[str]:
        s2 = f"  {s}  "  # pad to capture start/end
        return [s2[i:i+3] for i in range(len(s2)-2)]

    def query(self, q: str, top_k: int = 50) -> List[Tuple[str, int]]:
        """Return list of (name_id, score) ordered by matching trigram counts."""
        q_norm = _normalize_for_index(q)
        q_trigrams = self._trigrams(q_norm)
        if not q_trigrams:
            return []
        placeholders = ",".join("?" for _ in q_trigrams)
        sql = f"""
            SELECT name_id, COUNT(*) as cnt
            FROM trigrams
            WHERE trigram IN ({placeholders})
            GROUP BY name_id
            ORDER BY cnt DESC
            LIMIT ?
        """
        params = q_trigrams + [top_k]
        c = self.conn.cursor()
        c.execute(sql, params)
        return c.fetchall()

    def name_for_id(self, name_id: str) -> str:
        c = self.conn.cursor()
        c.execute("SELECT name FROM names WHERE id = ?", (name_id,))
        r = c.fetchone()
        return r[0] if r else ""
