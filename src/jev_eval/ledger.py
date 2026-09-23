"""Single-process SQLite request ledger and shared, conservative spend accounting."""

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .io import read_json


def now():
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS attempts (
          id INTEGER PRIMARY KEY, request_key TEXT, started_at TEXT,
          status TEXT, reserved_usd TEXT, actual_usd TEXT,
          reserved_tokens INTEGER, tokens INTEGER, raw_path TEXT, error_code TEXT);
        CREATE TABLE IF NOT EXISTS success (
          request_key TEXT PRIMARY KEY, raw_path TEXT, model_resolved TEXT);
        """)

    def cached(self, key):
        row = self.db.execute(
            "SELECT raw_path FROM success WHERE request_key=?", (key,)
        ).fetchone()
        return read_json(row[0])["response"] if row else None

    def totals(self):
        rows = self.db.execute(
            "SELECT reserved_usd, actual_usd, reserved_tokens, tokens FROM attempts"
        ).fetchall()
        known = sum((Decimal(a) for _, a, _, _ in rows if a is not None), Decimal(0))
        reserved = sum((Decimal(r) for r, a, _, _ in rows if a is None), Decimal(0))
        return {
            "attempts": len(rows),
            "cost_usd": str(known + reserved),
            "reported_cost_usd": str(known),
            "uncertain_reserved_usd": str(reserved),
            "tokens": sum(t if t is not None else r for _, _, r, t in rows),
        }

    def reserve(self, key, usd, tokens, max_usd, max_tokens, max_requests):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            totals = self.totals()
            if Decimal(totals["cost_usd"]) + usd > max_usd:
                raise ValueError("Monetary cap reached; checkpoint retained")
            if (
                totals["tokens"] + tokens > max_tokens
                or totals["attempts"] >= max_requests
            ):
                raise ValueError("Token/request cap reached; checkpoint retained")
            cursor = self.db.execute(
                "INSERT INTO attempts(request_key,started_at,status,reserved_usd,reserved_tokens) VALUES(?,?,?,?,?)",
                (key, now(), "pending", str(usd), tokens),
            )
            self.db.commit()
            return cursor.lastrowid
        except Exception:
            self.db.rollback()
            raise

    def finish(self, attempt, status, raw_path, cost=None, tokens=None, error=None):
        self.db.execute(
            "UPDATE attempts SET status=?,raw_path=?,actual_usd=?,tokens=?,error_code=? WHERE id=?",
            (
                status,
                str(raw_path),
                str(cost) if cost is not None else None,
                tokens,
                error,
                attempt,
            ),
        )
        self.db.commit()

    def success(self, key, raw_path, model):
        self.db.execute(
            "INSERT OR REPLACE INTO success VALUES(?,?,?)", (key, str(raw_path), model)
        )
        self.db.commit()
