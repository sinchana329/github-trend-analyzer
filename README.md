# GitHub Repository Trend Analyzer using Apache Spark and MongoDB

Final upgraded version for academic final-year complexity.

## Upgraded Capabilities

- Incremental GitHub ingestion with MongoDB checkpoint (`pipeline_meta.last_fetched_at`)
- Resumable ETL with full backfill + daily incremental mode
- MongoDB Atlas NoSQL storage with dedupe/indexing
- Spark RDD + DataFrame + Spark SQL analytics
- Advanced feature engineering (`stars_per_day`, `fork_star_ratio`, `repo_age_norm`)
- Bronze / Silver / Gold lakehouse layers in Parquet
- Advanced ML model comparison (Linear Trend vs ARIMA) with MAE/RMSE
- NLP topic intelligence using SentenceTransformer + clustering
- Graph analytics using NetworkX (centrality)
- Data quality validation report
- Observability output metrics + optional Prometheus Pushgateway
- Prefect orchestration flow with retries
- Enhanced Streamlit dashboard filters, anomalies, downloads, confidence proxy

## Project Structure

```text
bda_project/
├── app/
│   └── dashboard.py
├── data/
│   ├── raw_repos.json
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── output/
├── pipelines/
│   └── prefect_flow.py
├── scripts/
│   ├── fetch_data.py
│   ├── store_mongo.py
│   ├── spark_processing.py
│   ├── analysis.py
│   ├── visualization.py
│   ├── data_quality.py
│   ├── nlp_topic_intelligence.py
│   ├── graph_analytics.py
│   └── observability.py
├── .env.example
├── requirements.txt
└── README.md
```

## Environment Setup

Create `.env` from `.env.example`, then set:

```env
GITHUB_TOKEN=your_github_pat
MONGODB_URI=your_mongodb_atlas_uri
DATA_SOURCE=json
INCREMENTAL_MODE=false

# FINAL RUN: 3-year window
START_YEAR=2022
END_YEAR=2024
MIN_STARS=50
PER_PAGE=100
MAX_PAGES_PER_YEAR=10
SLEEP_SECONDS=1.0
```

## What is saved in MongoDB Atlas

- DB: `github_trends`
- Collection: `repos` (repository records + `ingested_at`)
- Collection: `pipeline_meta` (checkpoint: `last_fetched_at`, `record_count`)

## Final Run Commands (no-error sequence)

Run from project root:

```bash
pip install -r requirements.txt
python scripts/fetch_data.py
python scripts/store_mongo.py
python scripts/spark_processing.py
python scripts/analysis.py
python scripts/data_quality.py
python scripts/nlp_topic_intelligence.py
python scripts/graph_analytics.py
python scripts/visualization.py
python scripts/observability.py
streamlit run app/dashboard.py
```

## Orchestration Run (Prefect)

```bash
python pipelines/prefect_flow.py
```

This executes fetch -> store -> spark -> analysis -> quality -> NLP -> graph -> viz -> observability with retries.

## Key Outputs

- Core analytics:
  - `output/trends.csv`
  - `output/growth.csv`
  - `output/top_topics.csv`
  - `output/advanced_analytics.csv`
- ML:
  - `output/predicted_growth.csv`
  - `output/model_selection_report.csv`
- Feature engineering:
  - `output/feature_engineering.csv`
- Quality/NLP/Graph/Observability:
  - `output/data_quality_report.json`
  - `output/nlp_topic_clusters.csv`
  - `output/graph_metrics.csv`
  - `output/run_metrics.json`
- Visuals:
  - `output/charts_language_growth.png`
  - `output/charts_top_languages.png`
  - `output/charts_topic_trends.png`

## Incremental Daily Mode

Set:

```env
INCREMENTAL_MODE=true
```

Then run:

```bash
python scripts/fetch_data.py
python scripts/store_mongo.py
```

Only newly updated repos since last checkpoint are fetched.
