"""
Create charts from analysis outputs.

Outputs:
- output/charts_language_growth.png
- output/charts_top_languages.png
- output/charts_topic_trends.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _find_csv_data(folder: str) -> Path:
    folder_path = Path(folder)
    if folder_path.is_file():
        return folder_path
    csv_files = list(folder_path.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    part_files = list(folder_path.glob("part-*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No CSV file found inside {folder}")
    return part_files[0]


def load_growth_data() -> pd.DataFrame:
    growth_path = _find_csv_data("output/growth.csv")
    return pd.read_csv(growth_path).dropna(subset=["growth_rate_pct"])


def load_trend_data() -> pd.DataFrame:
    trend_path = _find_csv_data("output/trends.csv")
    return pd.read_csv(trend_path)


def load_topic_data() -> pd.DataFrame:
    topic_path = _find_csv_data("output/top_topics.csv")
    return pd.read_csv(topic_path).head(10)


def plot_growth(growth_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    top_langs = (
        growth_df.groupby("language")["repo_count"]
        .max()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    plot_df = growth_df[growth_df["language"].isin(top_langs)]
    sns.lineplot(data=plot_df, x="year", y="growth_rate_pct", hue="language", marker="o")
    plt.title("Language Growth Rate Over Years")
    plt.tight_layout()
    plt.savefig("output/charts_language_growth.png", dpi=180)
    plt.close()


def plot_top_languages(trend_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=trend_df, x="total_stars", y="language", palette="viridis")
    plt.title("Top 10 Trending Languages by Total Stars")
    plt.tight_layout()
    plt.savefig("output/charts_top_languages.png", dpi=180)
    plt.close()


def plot_topic_trends(topic_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=topic_df, x="repo_count", y="topic", palette="magma")
    plt.title("Top Topics by Repository Count")
    plt.tight_layout()
    plt.savefig("output/charts_topic_trends.png", dpi=180)
    plt.close()


def main() -> None:
    Path("output").mkdir(parents=True, exist_ok=True)
    growth_df = load_growth_data()
    trend_df = load_trend_data()
    topic_df = load_topic_data()

    plot_growth(growth_df)
    plot_top_languages(trend_df)
    plot_topic_trends(topic_df)
    print("Charts saved under output/")


if __name__ == "__main__":
    main()
