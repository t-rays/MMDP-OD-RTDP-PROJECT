from __future__ import annotations

"""Peak additional process-memory measurement."""

from dataclasses import dataclass
import os
import threading

import psutil

MIB = 1024 * 1024


@dataclass(frozen=True)
class ResourceSnapshot:
    peak_rss_delta_mb: float


class ResourceMonitor:
    """Sample process RSS until ``stop`` is called."""

    def __init__(self, sample_interval_seconds: float = 0.05) -> None:
        if sample_interval_seconds <= 0.0:
            raise ValueError("sample_interval_seconds must be positive")

        self.sample_interval_seconds = sample_interval_seconds
        self._process = psutil.Process(os.getpid())
        self._baseline_bytes = int(self._process.memory_info().rss)
        self._peak_bytes = self._baseline_bytes
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        try:
            rss = int(self._process.memory_info().rss)
        except psutil.Error:
            return
        self._peak_bytes = max(self._peak_bytes, rss)

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample_once()

    def start(self) -> "ResourceMonitor":
        if self._thread is None:
            self._sample_once()
            self._thread = threading.Thread(
                target=self._run,
                name="resource-monitor",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> ResourceSnapshot:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4 * self.sample_interval_seconds))
        self._sample_once()
        return ResourceSnapshot(
            peak_rss_delta_mb=max(
                0.0,
                (self._peak_bytes - self._baseline_bytes) / MIB,
            )
        )
