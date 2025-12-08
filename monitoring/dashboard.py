# monitoring/dashboard.py
from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import List


def load_latencies(path: str) -> List[float]:
    latencies: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "detect":
                continue
            if "latency_ms" in rec:
                latencies.append(float(rec["latency_ms"]))
    return latencies


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    n = len(vals)
    if n == 1:
        return float(vals[0])
    k = int(round((p / 100.0) * (n - 1)))
    return float(vals[k])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple CLI dashboard for API latency logs."
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to JSONL log file (from JsonLogger).",
    )
    args = parser.parse_args()

    latencies = load_latencies(args.log)
    if not latencies:
        print("No latency records found in log.")
        return

    print(f"Total requests: {len(latencies)}")
    print(f"Average latency (ms): {mean(latencies):.2f}")
    print(f"p50 latency (ms): {percentile(latencies, 50):.2f}")
    print(f"p90 latency (ms): {percentile(latencies, 90):.2f}")
    print(f"p95 latency (ms): {percentile(latencies, 95):.2f}")


if __name__ == "__main__":
    main()
