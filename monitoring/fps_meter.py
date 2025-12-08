# monitoring/fps_meter.py
import time
from collections import deque
from typing import Deque


class FPSMeter:
    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._timestamps: Deque[float] = deque(maxlen=window_size)
        self._last_time: float | None = None

    def tick(self) -> None:
        """Call this on every processed frame."""
        now = time.time()
        if self._last_time is not None:
            self._timestamps.append(now)
        self._last_time = now

    @property
    def fps(self) -> float:
        """Current FPS computed over the sliding window."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return float((len(self._timestamps) - 1) / elapsed)

    def reset(self) -> None:
        self._timestamps.clear()
        self._last_time = None
