"""Prefect workflow orchestration for full BDA pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from prefect import flow, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@task(retries=2, retry_delay_seconds=15, log_prints=True)
def run_step(command: list[str]) -> None:
    print(f"Running step: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


@flow(name="github-trend-etl-flow", log_prints=True)
def github_trend_pipeline():
    run_step([sys.executable, "scripts/fetch_data.py"])
    run_step([sys.executable, "scripts/store_mongo.py"])
    run_step([sys.executable, "scripts/spark_processing.py"])
    run_step([sys.executable, "scripts/analysis.py"])
    run_step([sys.executable, "scripts/data_quality.py"])
    run_step([sys.executable, "scripts/nlp_topic_intelligence.py"])
    run_step([sys.executable, "scripts/graph_analytics.py"])
    run_step([sys.executable, "scripts/visualization.py"])
    run_step([sys.executable, "scripts/observability.py"])


if __name__ == "__main__":
    github_trend_pipeline()
