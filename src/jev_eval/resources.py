import platform
import threading
import time
from pathlib import Path

import psutil

from .io import write_json


class ResourceMonitor:
    def __init__(self, destination):
        self.destination = Path(destination)
        self.stop = threading.Event()
        self.peak_rss = 0

    def __enter__(self):
        self.started = time.monotonic()
        self.swap_start = psutil.swap_memory().used
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            self.stop.wait(0.1)

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        write_json(
            self.destination,
            {
                "elapsed_seconds": time.monotonic() - self.started,
                "peak_rss_bytes": self.peak_rss,
                "swap_used_start": self.swap_start,
                "swap_used_end": psutil.swap_memory().used,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "machine": platform.machine(),
                "status": "failed" if kind else "complete",
                "thermal_throttling": "not_measured",
                "mps_peak_memory": "not_measured",
            },
        )
