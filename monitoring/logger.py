# monitoring/logger.py
from __future__ import annotations

import json
import time
from collections import deque
from statistics import mean
from typing import Deque, Dict, Any, Optional


# ------------------ GPU STATS ------------------


def get_gpu_stats() -> Dict[str, Optional[float | str]]:
    """
    Returns basic GPU metrics using pynvml if available.
    If no GPU / pynvml, returns None fields.
    """
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        used_mb = mem.used / (1024 * 1024)
        total_mb = mem.total / (1024 * 1024)

        return {
            "gpu_name": name,
            "gpu_memory_used_mb": float(used_mb),
            "gpu_memory_total_mb": float(total_mb),
            "gpu_utilization": float(util.gpu),
        }
    except Exception:
        return {
            "gpu_name": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_utilization": None,
        }


# ------------------ LATENCY / HISTOGRAM ------------------


class LatencyMeter:
    """
    Tracks request latencies and exposes:
      - global average
      - moving average (window_size)
      - p50 / p90 / p95
      - FPS estimate (requests/second)
      - total request count

    All units are milliseconds for latencies.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._latencies: Deque[float] = deque(maxlen=window_size)
        self._total_count: int = 0
        self._total_sum: float = 0.0
        self._start_time: float = time.time()

    def record_latency(self, ms: float) -> None:
        self._latencies.append(ms)
        self._total_count += 1
        self._total_sum += ms

    @staticmethod
    def _percentile(sorted_vals, p: float) -> float:
        if not sorted_vals:
            return 0.0
        n = len(sorted_vals)
        if n == 1:
            return float(sorted_vals[0])
        k = int(round((p / 100.0) * (n - 1)))
        return float(sorted_vals[k])

    def get_stats(self) -> Dict[str, float | int]:
        """
        Returns JSON-friendly stats dict:
          - avg_latency_ms (global)
          - moving_avg_latency_ms (window)
          - p50_latency_ms
          - p90_latency_ms
          - p95_latency_ms
          - fps
          - total_requests
        """
        lat_sorted = sorted(self._latencies)
        if lat_sorted:
            window_avg = float(mean(lat_sorted))
            p50 = self._percentile(lat_sorted, 50)
            p90 = self._percentile(lat_sorted, 90)
            p95 = self._percentile(lat_sorted, 95)
        else:
            window_avg = p50 = p90 = p95 = 0.0

        if self._total_count > 0:
            global_avg = self._total_sum / self._total_count
        else:
            global_avg = 0.0

        elapsed = max(time.time() - self._start_time, 1e-6)
        fps = float(self._total_count / elapsed)

        return {
            "avg_latency_ms": float(global_avg),
            "moving_avg_latency_ms": float(window_avg),
            "p50_latency_ms": float(p50),
            "p90_latency_ms": float(p90),
            "p95_latency_ms": float(p95),
            "fps": fps,
            "total_requests": self._total_count,
        }


# ------------------ JSON LOGGER ------------------


class JsonLogger:
    """
    Simple line-delimited JSON logger.

    Usage:
        logger = JsonLogger("logs/api_events.jsonl")
        logger.log("detect", {"latency_ms": 12.3, "backend": "onnx"})
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        # line-buffered text mode
        self._fh = open(self.log_path, "a", encoding="utf-8", buffering=1)

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        record: Dict[str, Any] = {
            "ts": time.time(),
            "event": event_type,
            **payload,
        }
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
