"""NLP topic intelligence using sentence embeddings + clustering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def main() -> None:
    input_path = Path("data/raw_repos.json")
    output_path = Path("output/nlp_topic_clusters.csv")
    if not input_path.exists():
        raise FileNotFoundError("data/raw_repos.json not found")

    df = pd.read_json(input_path)
    text_col = (
        df["name"].fillna("")
        + " "
        + df.get("description", pd.Series([""] * len(df))).fillna("")
        + " "
        + df["topics"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_col.tolist(), show_progress_bar=False)

    kmeans = KMeans(n_clusters=min(8, max(2, len(df) // 10)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    out_df = df[["id", "name", "language", "created_at", "stars"]].copy()
    out_df["cluster_id"] = labels
    out_df["year"] = pd.to_datetime(out_df["created_at"], errors="coerce").dt.year
    out_df.to_csv(output_path, index=False)
    print("NLP clusters saved to output/nlp_topic_clusters.csv")


if __name__ == "__main__":
    main()
