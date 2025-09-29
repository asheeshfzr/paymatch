# app/trigram_index.py
import sqlite3
import re
from typing import List, Tuple

def _normalize_for_index(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

class SimpleTrigramIndex:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS trigrams (user_id TEXT, trigram TEXT)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trigrams_trigram ON trigrams(trigram)")
        self.conn.commit()

    def _trigrams(self, s: str) -> List[str]:
        s2 = f"  {s}  "
        return [s2[i:i+3] for i in range(len(s2)-2)]

    def add(self, user_id: str, name: str):
        norm = _normalize_for_index(name)
        trigrams = self._trigrams(norm)
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO users (id, name) VALUES (?, ?)", (user_id, name))
        c.execute("DELETE FROM trigrams WHERE user_id = ?", (user_id,))
        rows = [(user_id, tg) for tg in trigrams]
        if rows:
            c.executemany("INSERT INTO trigrams (user_id, trigram) VALUES (?, ?)", rows)
        self.conn.commit()

    def query(self, q: str, top_k: int = 50) -> List[Tuple[str, int]]:
        q_norm = _normalize_for_index(q)
        q_trigrams = self._trigrams(q_norm)
        if not q_trigrams:
            return []
        placeholders = ",".join("?" for _ in q_trigrams)
        sql = f"""
            SELECT user_id, COUNT(*) as cnt
            FROM trigrams
            WHERE trigram IN ({placeholders})
            GROUP BY user_id
            ORDER BY cnt DESC
            LIMIT ?
        """
        params = q_trigrams + [top_k]
        c = self.conn.cursor()
        c.execute(sql, params)
        return c.fetchall()

    def get_name(self, user_id: str) -> str:
        c = self.conn.cursor()
        c.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        r = c.fetchone()
        return r[0] if r else ""
