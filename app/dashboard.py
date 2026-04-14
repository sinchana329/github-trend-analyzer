"""
Streamlit dashboard for GitHub Repository Trend Analyzer.
Run:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def _find_csv_data(folder: str) -> Path:
    folder_path = Path(folder)
    if folder_path.is_file():
        return folder_path
    csv_files = list(folder_path.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    part_files = list(folder_path.glob("part-*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No CSV file found in: {folder}")
    return part_files[0]


@st.cache_data
def load_data():
    trends = pd.read_csv(_find_csv_data("output/trends.csv"))
    growth = pd.read_csv(_find_csv_data("output/growth.csv"))
    topics = pd.read_csv(_find_csv_data("output/top_topics.csv"))
    prediction_file = Path("output/predicted_growth.csv")
    predicted = pd.read_csv(_find_csv_data(str(prediction_file))) if prediction_file.exists() else pd.DataFrame()
    return trends, growth, topics, predicted


def main():
    st.set_page_config(page_title="GitHub Trend Analyzer", layout="wide")
    st.title("GitHub Repository Trend Analyzer")
    st.caption("Apache Spark + MongoDB + Streamlit")

    try:
        trends, growth, topics, predicted = load_data()
    except Exception as exc:
        st.error(f"Unable to load output files. Run processing scripts first. Error: {exc}")
        return

    if growth.empty:
        st.warning("Growth dataset is empty.")
        return

    min_year = int(growth["year"].min())
    max_year = int(growth["year"].max())
    selected_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    star_min = int(trends["total_stars"].min()) if "total_stars" in trends.columns else 0
    star_max = int(trends["total_stars"].max()) if "total_stars" in trends.columns else 1000
    selected_star = st.slider("Minimum Total Stars Filter", star_min, star_max, star_min)
    lang_options = sorted(trends["language"].dropna().unique().tolist()) if "language" in trends.columns else []
    selected_langs = st.multiselect("Languages", lang_options, default=lang_options[:8] if lang_options else [])
    topic_options = sorted(topics["topic"].dropna().head(50).tolist()) if "topic" in topics.columns else []
    selected_topics = st.multiselect("Topics", topic_options, default=topic_options[:10] if topic_options else [])

    filtered_growth = growth[(growth["year"] >= selected_range[0]) & (growth["year"] <= selected_range[1])]
    filtered_growth = filtered_growth.dropna(subset=["growth_rate_pct"])
    if selected_langs:
        filtered_growth = filtered_growth[filtered_growth["language"].isin(selected_langs)]
    filtered_trends = trends[trends["total_stars"] >= selected_star] if "total_stars" in trends.columns else trends
    if selected_langs and "language" in filtered_trends.columns:
        filtered_trends = filtered_trends[filtered_trends["language"].isin(selected_langs)]
    filtered_topics = topics[topics["topic"].isin(selected_topics)] if selected_topics else topics

    st.subheader("Top Trending Languages")
    st.dataframe(filtered_trends, use_container_width=True)

    st.subheader("Language Growth Rate")
    st.line_chart(
        filtered_growth.pivot_table(
            index="year",
            columns="language",
            values="growth_rate_pct",
            aggfunc="mean",
        )
    )

    st.subheader("Top Topics")
    st.bar_chart(filtered_topics.head(15).set_index("topic")["repo_count"])

    st.subheader("Insights")
    if not trends.empty:
        top_lang = trends.iloc[0]["language"]
        st.write(f"- Leading language by stars: **{top_lang}**")
    if not topics.empty:
        top_topic = topics.iloc[0]["topic"]
        st.write(f"- Most frequent topic in dataset: **{top_topic}**")
    st.write("- Growth percentages come from year-over-year language repository counts.")

    st.subheader("Anomaly Highlights")
    if not filtered_growth.empty:
        anomaly_df = filtered_growth[filtered_growth["growth_rate_pct"] > filtered_growth["growth_rate_pct"].quantile(0.9)]
        st.dataframe(anomaly_df.head(10), use_container_width=True)
    else:
        st.info("No anomaly candidates for current filters.")

    st.subheader("Forecast Confidence Proxy")
    if not predicted.empty and "predicted_repo_count" in predicted.columns:
        pred = predicted.copy()
        pred["lower_ci"] = (pred["predicted_repo_count"] * 0.9).round(2)
        pred["upper_ci"] = (pred["predicted_repo_count"] * 1.1).round(2)
        st.dataframe(pred, use_container_width=True)
        st.download_button(
            "Download Predicted Growth CSV",
            data=pred.to_csv(index=False).encode("utf-8"),
            file_name="predicted_growth_with_ci.csv",
            mime="text/csv",
        )

    if not predicted.empty:
        st.subheader("Predicted Language Growth (Linear Regression)")
        st.dataframe(predicted, use_container_width=True)


if __name__ == "__main__":
    main()
