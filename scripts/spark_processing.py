"""
Distributed processing with PySpark using both RDD and DataFrame APIs.

Outputs:
- output/processed_language_year.csv
- output/processed_language_month.csv
- output/rdd_language_stars.json
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, month, to_timestamp, year

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def get_spark() -> SparkSession:
    python_exec = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec
    return (
        SparkSession.builder.appName("GitHubRepoTrendAnalyzer")
        .master("local[*]")
        .config("spark.pyspark.python", python_exec)
        .config("spark.pyspark.driver.python", python_exec)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_df(spark: SparkSession, source: str = "json") -> DataFrame:
    if source == "mongo":
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("MONGODB_URI is required for mongo source.")
        return (
            spark.read.format("mongodb")
            .option("spark.mongodb.read.connection.uri", mongo_uri)
            .option("spark.mongodb.read.database", "github_trends")
            .option("spark.mongodb.read.collection", "repos")
            .load()
        )

    # raw_repos.json is a JSON array, so Spark must parse in multiline mode.
    return spark.read.option("multiLine", True).json("data/raw_repos.json")


def run_rdd_analysis(df) -> None:
    rdd = (
        df.select("language", "stars")
        .where(col("language").isNotNull())
        .rdd.map(lambda r: (r["language"], int(r["stars"] or 0)))
    )

    # RDD operations required by assignment: map + reduceByKey
    reduced = rdd.reduceByKey(lambda a, b: a + b)
    result = [{"language": x[0], "total_stars": x[1]} for x in reduced.collect()]
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/rdd_language_stars.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("RDD analysis complete: output/rdd_language_stars.json")


def run_dataframe_analysis(df) -> None:
    clean_df = (
        df.where(col("language").isNotNull())
        .withColumn("created_ts", to_timestamp(col("created_at")))
        .withColumn("year", year(col("created_ts")))
        .withColumn("month", month(col("created_ts")))
        .cache()
    )

    # Additional engineered features for final-year complexity.
    featured_pdf = clean_df.toPandas()
    if not featured_pdf.empty:
        now = pd.Timestamp.utcnow().tz_localize(None)
        featured_pdf["created_ts"] = pd.to_datetime(featured_pdf["created_ts"], errors="coerce", utc=True).dt.tz_localize(None)
        featured_pdf["repo_age_days"] = (now - featured_pdf["created_ts"]).dt.days.clip(lower=1)
        featured_pdf["stars_per_day"] = (featured_pdf["stars"] / featured_pdf["repo_age_days"]).round(4)
        featured_pdf["fork_star_ratio"] = (featured_pdf["forks"] / featured_pdf["stars"].replace(0, 1)).round(4)
        featured_pdf["repo_age_norm"] = (
            (featured_pdf["repo_age_days"] - featured_pdf["repo_age_days"].min())
            / (featured_pdf["repo_age_days"].max() - featured_pdf["repo_age_days"].min() + 1e-9)
        ).round(4)

    lang_year_df = clean_df.groupBy("language", "year").agg(
        {"*": "count", "stars": "sum"}
    ).withColumnRenamed("count(1)", "repo_count").withColumnRenamed("sum(stars)", "total_stars")

    lang_month_df = clean_df.groupBy("language", "month").agg(
        {"*": "count", "stars": "sum"}
    ).withColumnRenamed("count(1)", "repo_count").withColumnRenamed("sum(stars)", "total_stars")

    Path("output").mkdir(parents=True, exist_ok=True)
    for target in ("output/processed_language_year.csv", "output/processed_language_month.csv"):
        target_path = Path(target)
        if target_path.is_dir():
            shutil.rmtree(target_path, ignore_errors=False)
        elif target_path.exists():
            target_path.unlink()

    def _safe_write_csv(pandas_df: pd.DataFrame, target: str) -> None:
        target_path = Path(target)
        try:
            pandas_df.to_csv(target_path, index=False)
        except PermissionError:
            fallback = target_path.with_name(
                f"{target_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target_path.suffix}"
            )
            pandas_df.to_csv(fallback, index=False)
            LOGGER.warning("File locked: %s. Wrote fallback: %s", target_path, fallback)

    _safe_write_csv(lang_year_df.toPandas(), "output/processed_language_year.csv")
    _safe_write_csv(lang_month_df.toPandas(), "output/processed_language_month.csv")
    if not featured_pdf.empty:
        _safe_write_csv(featured_pdf, "output/feature_engineering.csv")

    # Bronze/Silver/Gold architecture
    Path("data/bronze").mkdir(parents=True, exist_ok=True)
    Path("data/silver").mkdir(parents=True, exist_ok=True)
    Path("data/gold").mkdir(parents=True, exist_ok=True)
    df.toPandas().to_parquet("data/bronze/repos_raw.parquet", index=False)
    clean_df.toPandas().to_parquet("data/silver/repos_clean.parquet", index=False)
    lang_year_df.toPandas().to_parquet("data/gold/lang_year_agg.parquet", index=False)

    LOGGER.info("DataFrame analysis complete: processed csv outputs saved.")


def main(source: Optional[str] = None) -> None:
    source = source or os.getenv("DATA_SOURCE", "json")
    spark = get_spark()
    try:
        df = load_df(spark, source=source)
        run_rdd_analysis(df)
        run_dataframe_analysis(df)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
