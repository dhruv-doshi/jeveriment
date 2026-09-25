"""Single-process SQLite request ledger and shared, conservative spend accounting."""

import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .io import read_json


def now():
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        for attempt in range(10):
            try:
                self.db.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower() or attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS attempts (
          id INTEGER PRIMARY KEY, request_key TEXT, started_at TEXT,
          status TEXT, reserved_usd TEXT, actual_usd TEXT,
          reserved_tokens INTEGER, tokens INTEGER, raw_path TEXT, error_code TEXT);
        CREATE TABLE IF NOT EXISTS success (
          request_key TEXT PRIMARY KEY, raw_path TEXT, model_resolved TEXT);
        CREATE TABLE IF NOT EXISTS budget_state (
          id INTEGER PRIMARY KEY CHECK (id = 1), attempts INTEGER NOT NULL,
          cost_usd TEXT NOT NULL, reported_cost_usd TEXT NOT NULL,
          uncertain_reserved_usd TEXT NOT NULL, tokens INTEGER NOT NULL);
        """)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            # Reconcile records written by an older client before this process
            # starts. This scan runs once per connection, not once per request.
            rows = self.db.execute(
                "SELECT reserved_usd, actual_usd, reserved_tokens, tokens "
                "FROM attempts"
            ).fetchall()
            known = sum(
                (Decimal(a) for _, a, _, _ in rows if a is not None),
                Decimal(0),
            )
            reserved = sum(
                (Decimal(r) for r, a, _, _ in rows if a is None),
                Decimal(0),
            )
            tokens = sum(t if t is not None else r for _, _, r, t in rows)
            self.db.execute(
                "INSERT OR REPLACE INTO budget_state VALUES(1,?,?,?,?,?)",
                (len(rows), str(known + reserved), str(known), str(reserved), tokens),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def cached(self, key):
        row = self.db.execute(
            "SELECT raw_path FROM success WHERE request_key=?", (key,)
        ).fetchone()
        return read_json(row[0])["response"] if row else None

    def totals(self):
        attempts, cost, known, reserved, tokens = self.db.execute(
            "SELECT attempts,cost_usd,reported_cost_usd,"
            "uncertain_reserved_usd,tokens FROM budget_state WHERE id=1"
        ).fetchone()
        return {
            "attempts": attempts,
            "cost_usd": cost,
            "reported_cost_usd": known,
            "uncertain_reserved_usd": reserved,
            "tokens": tokens,
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
            self.db.execute(
                "UPDATE budget_state SET attempts=?,cost_usd=?,"
                "uncertain_reserved_usd=?,tokens=? WHERE id=1",
                (
                    totals["attempts"] + 1,
                    str(Decimal(totals["cost_usd"]) + usd),
                    str(Decimal(totals["uncertain_reserved_usd"]) + usd),
                    totals["tokens"] + tokens,
                ),
            )
            self.db.commit()
            return cursor.lastrowid
        except Exception:
            self.db.rollback()
            raise

    def finish(self, attempt, status, raw_path, cost=None, tokens=None, error=None):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT reserved_usd,actual_usd,reserved_tokens,tokens "
                "FROM attempts WHERE id=?", (attempt,)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown budget attempt")
            reserved_usd, previous_cost, reserved_tokens, previous_tokens = row
            totals = self.totals()
            old_cost = Decimal(previous_cost or reserved_usd)
            new_cost = Decimal(str(cost)) if cost is not None else Decimal(reserved_usd)
            old_reported = Decimal(previous_cost) if previous_cost is not None else Decimal(0)
            new_reported = Decimal(str(cost)) if cost is not None else Decimal(0)
            old_reserved = Decimal(reserved_usd) if previous_cost is None else Decimal(0)
            new_reserved = Decimal(reserved_usd) if cost is None else Decimal(0)
            self.db.execute(
                "UPDATE attempts SET status=?,raw_path=?,actual_usd=?,tokens=?,"
                "error_code=? WHERE id=?",
                (
                    status,
                    str(raw_path),
                    str(cost) if cost is not None else None,
                    tokens,
                    error,
                    attempt,
                ),
            )
            self.db.execute(
                "UPDATE budget_state SET cost_usd=?,reported_cost_usd=?,"
                "uncertain_reserved_usd=?,tokens=? WHERE id=1",
                (
                    str(Decimal(totals["cost_usd"]) - old_cost + new_cost),
                    str(Decimal(totals["reported_cost_usd"]) - old_reported + new_reported),
                    str(Decimal(totals["uncertain_reserved_usd"]) - old_reserved + new_reserved),
                    totals["tokens"] - (
                        previous_tokens if previous_tokens is not None else reserved_tokens
                    ) + (tokens if tokens is not None else reserved_tokens),
                ),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def success(self, key, raw_path, model):
        self.db.execute(
            "INSERT OR REPLACE INTO success VALUES(?,?,?)", (key, str(raw_path), model)
        )
        self.db.commit()
