"""Lightweight observability and run metrics exporter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


def main() -> None:
    output_dir = Path("output")
    files = list(output_dir.glob("*"))
    run_metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_file_count": len(files),
    }
    Path("output/run_metrics.json").write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")

    # Optional Pushgateway support.
    gateway = None
    try:
        import os

        gateway = os.getenv("PUSHGATEWAY_URL")
    except Exception:
        gateway = None

    if gateway:
        registry = CollectorRegistry()
        gauge = Gauge("github_trend_output_files", "Number of output files", registry=registry)
        gauge.set(len(files))
        push_to_gateway(gateway, job="github_trend_pipeline", registry=registry)

    print("Observability metrics saved to output/run_metrics.json")


if __name__ == "__main__":
    main()
