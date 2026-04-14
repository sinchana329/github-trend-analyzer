"""Graph analytics for language-topic-repo relationships."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def main() -> None:
    input_path = Path("data/raw_repos.json")
    output_path = Path("output/graph_metrics.csv")
    if not input_path.exists():
        raise FileNotFoundError("data/raw_repos.json not found")

    df = pd.read_json(input_path)
    graph = nx.Graph()

    for _, row in df.iterrows():
        repo_node = f"repo:{row.get('id')}"
        lang_node = f"lang:{row.get('language') or 'Unknown'}"
        graph.add_edge(repo_node, lang_node, edge_type="repo-language")
        topics = row.get("topics") if isinstance(row.get("topics"), list) else []
        for topic in topics:
            topic_node = f"topic:{topic}"
            graph.add_edge(repo_node, topic_node, edge_type="repo-topic")

    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, k=min(100, len(graph)), seed=42)

    metrics = pd.DataFrame(
        [{"node": node, "degree_centrality": degree[node], "betweenness_centrality": betweenness[node]} for node in graph.nodes]
    ).sort_values("degree_centrality", ascending=False)
    metrics.to_csv(output_path, index=False)
    print("Graph analytics saved to output/graph_metrics.csv")


if __name__ == "__main__":
    main()
