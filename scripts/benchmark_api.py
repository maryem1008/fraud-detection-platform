"""Asynchronously benchmark the local FastAPI prediction endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import median

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def request_payloads(count: int) -> list[dict[str, float]]:
    path = ROOT / "data" / "processed" / "test.csv"
    if not path.exists():
        raise FileNotFoundError("Run scripts/prepare_data.py before benchmarking")
    frame = pd.read_csv(path, usecols=FEATURES)
    if frame.empty:
        raise ValueError("The held-out test split is empty")
    return [
        {
            "time": float(row["Time"]),
            "amount": float(row["Amount"]),
            **{f"v{i}": float(row[f"V{i}"]) for i in range(1, 29)},
        }
        for _, row in frame.iloc[:count].iterrows()
    ]


async def benchmark(base_url: str, count: int, concurrency: int) -> dict[str, float | int | str]:
    payloads = request_payloads(count)
    limiter = asyncio.Semaphore(concurrency)
    observed: list[float] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        async def send(payload: dict[str, float]) -> None:
            async with limiter:
                response = await client.post("/predict", json=payload)
                response.raise_for_status()
                body = response.json()
                observed.append(float(body["latency_ms"]))

        await asyncio.gather(*(send(payloads[i % len(payloads)]) for i in range(count)))
        health = await client.get("/health")
        health.raise_for_status()

    ordered = sorted(observed)
    p95 = ordered[min(int(0.95 * (len(ordered) - 1)), len(ordered) - 1)]
    return {
        "base_url": base_url,
        "requests": len(observed),
        "concurrency": concurrency,
        "median_inference_latency_ms": median(observed),
        "p95_inference_latency_ms": p95,
        "max_inference_latency_ms": max(observed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=250)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON result")
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("--requests and --concurrency must be positive")
    result = asyncio.run(benchmark(args.url, args.requests, args.concurrency))
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
