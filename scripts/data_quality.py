"""Data quality checks and schema drift report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {
    "id",
    "name",
    "full_name",
    "language",
    "stars",
    "forks",
    "created_at",
    "topics",
    "html_url",
}


def main() -> None:
    data_path = Path("data/raw_repos.json")
    output_path = Path("output/data_quality_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        output_path.write_text(json.dumps({"error": "raw_repos.json missing"}, indent=2), encoding="utf-8")
        return

    df = pd.read_json(data_path)
    report = {
        "row_count": int(len(df)),
        "duplicate_repo_ids": int(df["id"].duplicated().sum()) if "id" in df.columns else None,
        "null_language_pct": round(float(df["language"].isna().mean() * 100), 2) if "language" in df.columns else None,
        "schema_missing_columns": sorted(list(EXPECTED_COLUMNS - set(df.columns))),
        "schema_extra_columns": sorted(list(set(df.columns) - EXPECTED_COLUMNS)),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Data quality report generated: output/data_quality_report.json")


if __name__ == "__main__":
    main()
