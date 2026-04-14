"""
Spark SQL analytics and optional ML forecasting.

Outputs:
- output/trends.csv
- output/growth.csv
- output/avg_stars.csv
- output/top_topics.csv
- output/advanced_analytics.csv
- output/predicted_growth.csv
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    explode,
    lit,
    month,
    to_timestamp,
    when,
    year,
)
from pyspark.sql.window import Window
from pyspark.sql.functions import lag, round as spark_round
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def get_base_df(spark):
    return (
        spark.read.option("multiLine", True).json("data/raw_repos.json")
        .where(col("language").isNotNull())
        .withColumn("created_ts", to_timestamp(col("created_at")))
        .withColumn("year", year(col("created_ts")))
        .withColumn("month", month(col("created_ts")))
    )


def save_single_csv(df, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=False)
    elif target.exists():
        target.unlink()

    pandas_df = df.toPandas()
    try:
        pandas_df.to_csv(target, index=False)
    except PermissionError:
        fallback = target.with_name(f"{target.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target.suffix}")
        pandas_df.to_csv(fallback, index=False)
        LOGGER.warning("File locked: %s. Wrote fallback: %s", target, fallback)


def run_sql_analysis(spark, base_df):
    base_df.createOrReplaceTempView("repos")

    growth_query = """
        WITH yearly AS (
            SELECT language, year, COUNT(*) AS repo_count
            FROM repos
            GROUP BY language, year
        ),
        lagged AS (
            SELECT
                language,
                year,
                repo_count,
                LAG(repo_count) OVER (PARTITION BY language ORDER BY year) AS prev_count
            FROM yearly
        )
        SELECT
            language,
            year,
            repo_count,
            prev_count,
            CASE
                WHEN prev_count IS NULL OR prev_count = 0 THEN NULL
                ELSE ROUND(((repo_count - prev_count) * 100.0) / prev_count, 2)
            END AS growth_rate_pct
        FROM lagged
    """
    growth_df = spark.sql(growth_query)
    save_single_csv(growth_df, "output/growth.csv")

    avg_stars_df = spark.sql(
        """
        SELECT language, ROUND(AVG(stars), 2) AS avg_stars_per_repo
        FROM repos
        GROUP BY language
        ORDER BY avg_stars_per_repo DESC
        """
    )
    save_single_csv(avg_stars_df, "output/avg_stars.csv")

    trending_df = spark.sql(
        """
        SELECT language, COUNT(*) AS repo_count, SUM(stars) AS total_stars
        FROM repos
        GROUP BY language
        ORDER BY total_stars DESC
        LIMIT 10
        """
    )
    save_single_csv(trending_df, "output/trends.csv")

    topic_df = (
        base_df.select(explode(col("topics")).alias("topic"), "stars")
        .groupBy("topic")
        .agg({"*": "count", "stars": "sum"})
        .withColumnRenamed("count(1)", "repo_count")
        .withColumnRenamed("sum(stars)", "total_stars")
        .orderBy(col("repo_count").desc())
    )
    save_single_csv(topic_df, "output/top_topics.csv")

    LOGGER.info("Spark SQL analysis outputs saved.")


def run_advanced_analytics(base_df):
    topic_targets = ["ai", "web", "blockchain"]
    topic_flat = base_df.select("year", "month", "language", "stars", explode(col("topics")).alias("topic"))
    filtered_topic = topic_flat.where(col("topic").isin(topic_targets))

    month_trend = (
        base_df.groupBy("year", "month", "language")
        .agg({"*": "count", "stars": "sum"})
        .withColumnRenamed("count(1)", "repo_count")
        .withColumnRenamed("sum(stars)", "total_stars")
    )

    topic_trend = (
        filtered_topic.groupBy("year", "month", "topic")
        .agg({"*": "count", "stars": "sum"})
        .withColumnRenamed("count(1)", "repo_count")
        .withColumnRenamed("sum(stars)", "total_stars")
    )

    star_filtered = base_df.where(col("stars") >= 500).select("language", "year", "month", "stars")

    lang_compare = base_df.where(col("language").isin("Python", "JavaScript")).groupBy("language").agg(
        {"*": "count", "stars": "sum"}
    ).withColumnRenamed("count(1)", "repo_count").withColumnRenamed("sum(stars)", "total_stars")

    productivity = (
        base_df.groupBy("language", "year")
        .agg({"*": "count", "stars": "sum", "forks": "sum", "open_issues": "sum"})
        .withColumnRenamed("count(1)", "repo_count")
        .withColumnRenamed("sum(stars)", "total_stars")
        .withColumnRenamed("sum(forks)", "total_forks")
        .withColumnRenamed("sum(open_issues)", "total_open_issues")
        .withColumn("productivity_score", spark_round(col("total_stars") / col("repo_count"), 2))
        .withColumn(
            "activity_proxy_score",
            spark_round((col("total_stars") + (col("total_forks") * 2) + col("total_open_issues")) / col("repo_count"), 2),
        )
    )

    advanced = (
        month_trend.withColumn("metric", lit("month_wise_trend"))
        .select("metric", "language", "year", "month", "repo_count", "total_stars")
        .unionByName(
            topic_trend.withColumn("metric", lit("topic_trend"))
            .withColumnRenamed("topic", "language")
            .select("metric", "language", "year", "month", "repo_count", "total_stars")
        )
        .unionByName(
            star_filtered.withColumn("metric", lit("stars_ge_500"))
            .withColumn("repo_count", lit(1))
            .withColumn("total_stars", col("stars"))
            .select("metric", "language", "year", "month", "repo_count", "total_stars")
        )
    )
    save_single_csv(advanced, "output/advanced_analytics.csv")
    save_single_csv(lang_compare, "output/language_comparison.csv")
    save_single_csv(productivity, "output/developer_productivity.csv")
    LOGGER.info("Advanced analytics outputs saved.")


def run_ml_prediction(spark, base_df):
    yearly = base_df.groupBy("language", "year").agg({"*": "count"}).withColumnRenamed("count(1)", "repo_count")
    languages = [row["language"] for row in yearly.select("language").distinct().collect()]

    predictions = None
    for language in languages:
        lang_df = yearly.where(col("language") == language).orderBy("year")
        if lang_df.count() < 3:
            continue

        feature_df = lang_df.withColumn("year_num", col("year").cast("double")).withColumn(
            "label", col("repo_count").cast("double")
        )
        assembler = VectorAssembler(inputCols=["year_num"], outputCol="features")
        training = assembler.transform(feature_df).select("features", "label")
        model = LinearRegression(featuresCol="features", labelCol="label").fit(training)

        future_year_df = spark.createDataFrame([(2025.0,), (2026.0,)], ["year_num"])
        future_features = assembler.transform(future_year_df)
        forecast = model.transform(future_features).select(
            lit(language).alias("language"),
            col("year_num").cast("int").alias("year"),
            spark_round(col("prediction"), 2).alias("predicted_repo_count"),
        )

        predictions = forecast if predictions is None else predictions.unionByName(forecast)

    if predictions is None:
        predictions = spark.createDataFrame([], "language string, year int, predicted_repo_count double")

    save_single_csv(predictions, "output/predicted_growth.csv")

    yearly_pdf = yearly.toPandas()
    model_report_rows = []
    for language, lang_group in yearly_pdf.groupby("language"):
        lang_group = lang_group.sort_values("year")
        if len(lang_group) < 4:
            continue

        train = lang_group.iloc[:-1]
        test = lang_group.iloc[-1:]
        y_train = train["repo_count"].astype(float).values
        y_test = test["repo_count"].astype(float).values

        # Linear trend model
        x_train = train["year"].astype(float).values
        coeff = np.polyfit(x_train, y_train, 1)
        linear_pred = np.polyval(coeff, test["year"].astype(float).values)

        # ARIMA model
        try:
            arima_model = ARIMA(y_train, order=(1, 1, 1)).fit()
            arima_pred = arima_model.forecast(steps=1)
        except Exception:
            arima_pred = linear_pred

        linear_mae = float(mean_absolute_error(y_test, linear_pred))
        linear_rmse = float(np.sqrt(mean_squared_error(y_test, linear_pred)))
        arima_mae = float(mean_absolute_error(y_test, arima_pred))
        arima_rmse = float(np.sqrt(mean_squared_error(y_test, arima_pred)))

        best_model = "ARIMA" if arima_rmse < linear_rmse else "LinearTrend"
        model_report_rows.append(
            {
                "language": language,
                "linear_mae": round(linear_mae, 4),
                "linear_rmse": round(linear_rmse, 4),
                "arima_mae": round(arima_mae, 4),
                "arima_rmse": round(arima_rmse, 4),
                "selected_model": best_model,
            }
        )

    report_df = pd.DataFrame(model_report_rows)
    report_df.to_csv("output/model_selection_report.csv", index=False)

    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name="language_growth_model_comparison"):
        if not report_df.empty:
            mlflow.log_metric("avg_linear_rmse", float(report_df["linear_rmse"].mean()))
            mlflow.log_metric("avg_arima_rmse", float(report_df["arima_rmse"].mean()))
            mlflow.log_artifact("output/model_selection_report.csv")

    LOGGER.info("ML prediction and model selection report saved.")


def main():
    python_exec = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec
    spark = (
        SparkSession.builder.appName("GitHubTrendSQLAnalysis")
        .master("local[*]")
        .config("spark.pyspark.python", python_exec)
        .config("spark.pyspark.driver.python", python_exec)
        .getOrCreate()
    )
    Path("output").mkdir(parents=True, exist_ok=True)

    try:
        base_df = get_base_df(spark)
        run_sql_analysis(spark, base_df)
        run_advanced_analytics(base_df)
        run_ml_prediction(spark, base_df)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
