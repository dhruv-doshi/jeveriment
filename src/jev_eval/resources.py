import math
import platform
import sqlite3
import statistics
import threading
import time
from decimal import Decimal
from pathlib import Path

import psutil

from .io import read_json, write_json


class ResourceMonitor:
    def __init__(self, destination):
        self.destination = Path(destination)
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_cpu_percent = 0.0
        self.process = psutil.Process()
        self.api_attempt_start = 0

    @staticmethod
    def _swap_used():
        try:
            return psutil.swap_memory().used
        except (OSError, RuntimeError):
            # macOS can expose swap counters conditionally; this is optional
            # telemetry and must not prevent a phase from running.
            return None

    def __enter__(self):
        self.started = time.monotonic()
        self.cpu_start = self.process.cpu_times()
        ledger_path = Path("runs/budget_typesafe.sqlite")
        if ledger_path.exists():
            with sqlite3.connect(ledger_path, timeout=30) as db:
                self.api_attempt_start = db.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM attempts"
                ).fetchone()[0]
        self.swap_start = self._swap_used()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        last_wall = time.monotonic()
        last_cpu = self.process.cpu_times()
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.stop.wait(0.1)
            current_wall = time.monotonic()
            current_cpu = self.process.cpu_times()
            elapsed = current_wall - last_wall
            if elapsed > 0:
                used = (current_cpu.user + current_cpu.system) - (
                    last_cpu.user + last_cpu.system
                )
                self.peak_cpu_percent = max(
                    self.peak_cpu_percent, 100 * used / elapsed
                )
            last_wall, last_cpu = current_wall, current_cpu

    def _api_usage(self):
        ledger_path = Path("runs/budget_typesafe.sqlite")
        if not ledger_path.exists():
            return None
        run_prefix = str(self.destination.parent) + "/"
        with sqlite3.connect(ledger_path, timeout=30) as db:
            rows = db.execute(
                "SELECT id,status,actual_usd,reserved_usd,tokens,reserved_tokens,raw_path "
                "FROM attempts WHERE id>? ORDER BY id",
                (self.api_attempt_start,),
            ).fetchall()
        rows = [r for r in rows if r[6] and r[6].startswith(run_prefix)]
        latencies = []
        for row in rows:
            try:
                elapsed = read_json(row[6]).get("elapsed_seconds")
            except (OSError, ValueError, TypeError):
                elapsed = None
            if isinstance(elapsed, (int, float)) and elapsed >= 0:
                latencies.append(float(elapsed))
        ordered = sorted(latencies)
        return {
            "attempts": len(rows),
            "successful_attempts": sum(row[1] == "success" for row in rows),
            "reported_cost_usd": str(
                sum((Decimal(row[2]) for row in rows if row[2] is not None), Decimal(0))
            ),
            "uncertain_reserved_cost_usd": str(
                sum((Decimal(row[3]) for row in rows if row[2] is None), Decimal(0))
            ),
            "reported_tokens": sum(row[4] or 0 for row in rows),
            "latency_samples": len(ordered),
            "latency_seconds": {
                "mean": statistics.mean(ordered) if ordered else None,
                "median": statistics.median(ordered) if ordered else None,
                "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
                if ordered
                else None,
                "max": max(ordered) if ordered else None,
            },
        }

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        elapsed = time.monotonic() - self.started
        cpu_end = self.process.cpu_times()
        cpu_seconds = (cpu_end.user + cpu_end.system) - (
            self.cpu_start.user + self.cpu_start.system
        )
        write_json(
            self.destination,
            {
                "elapsed_seconds": elapsed,
                "process_cpu_seconds": cpu_seconds,
                "average_cpu_percent": 100 * cpu_seconds / elapsed if elapsed else 0,
                "peak_cpu_percent": self.peak_cpu_percent,
                "peak_rss_bytes": self.peak_rss,
                "api_usage": self._api_usage(),
                "swap_used_start": self.swap_start,
                "swap_used_end": self._swap_used(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "machine": platform.machine(),
                "status": "failed" if kind else "complete",
                "thermal_throttling": "not_measured",
                "mps_peak_memory": "not_measured",
            },
        )
