from __future__ import annotations

"""Low-overhead process-memory measurement for isolated experiment runs."""

from dataclasses import dataclass
import os
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover - clear runtime message on user machines
    psutil = None  # type: ignore[assignment]


MIB = 1024 * 1024


@dataclass(frozen=True)
class ResourceSnapshot:
    baseline_rss_mb: float
    peak_rss_mb: float
    peak_rss_delta_mb: float
    memory_limit_mb: float | None
    memory_limit_reached: bool


class ResourceMonitor:
    """Sample process RSS in a daemon thread.

    ``memory_limit_mb`` is interpreted as additional RSS above the value at the
    start of the measured phase.  This makes sequential interactive runs less
    sensitive to the Python interpreter's already-loaded modules.  For final
    benchmarks, each run should still be isolated in a fresh subprocess.
    """

    def __init__(
        self,
        *,
        memory_limit_mb: float | None = None,
        sample_interval_seconds: float = 0.05,
    ) -> None:
        if memory_limit_mb is not None and memory_limit_mb <= 0.0:
            raise ValueError("memory_limit_mb must be positive or None")
        if sample_interval_seconds <= 0.0:
            raise ValueError("sample_interval_seconds must be positive")
        if psutil is None:
            raise RuntimeError(
                "Memory measurement requires psutil. Install it with: "
                "python -m pip install psutil"
            )

        self.memory_limit_mb = memory_limit_mb
        self.sample_interval_seconds = sample_interval_seconds
        self._process = psutil.Process(os.getpid())
        self._baseline_bytes = int(self._process.memory_info().rss)
        self._peak_bytes = self._baseline_bytes
        self._limit_reached = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        try:
            rss = int(self._process.memory_info().rss)
        except Exception:
            return
        if rss > self._peak_bytes:
            self._peak_bytes = rss
        if self.memory_limit_mb is not None:
            delta_mb = (rss - self._baseline_bytes) / MIB
            if delta_mb >= self.memory_limit_mb:
                self._limit_reached.set()

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample_once()

    def start(self) -> "ResourceMonitor":
        if self._thread is not None:
            return self
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
        return self.snapshot()

    def limit_reached(self) -> bool:
        return self._limit_reached.is_set()

    def snapshot(self) -> ResourceSnapshot:
        baseline_mb = self._baseline_bytes / MIB
        peak_mb = self._peak_bytes / MIB
        return ResourceSnapshot(
            baseline_rss_mb=baseline_mb,
            peak_rss_mb=peak_mb,
            peak_rss_delta_mb=max(0.0, peak_mb - baseline_mb),
            memory_limit_mb=self.memory_limit_mb,
            memory_limit_reached=self.limit_reached(),
        )

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
