"""Graph visualization helpers for interactive and static outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pyvis.network import Network


def visualize_graph_pyvis(G: nx.Graph, output_path: str) -> None:
    """Create an interactive PyVis HTML visualization."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="750px", width="100%", directed=G.is_directed(), notebook=True)
    net.barnes_hut()

    for node, attrs in G.nodes(data=True):
        title = str(attrs.get("title", node))
        label = title[:80] + ("..." if len(title) > 80 else "")
        net.add_node(node, label=label, title=title)

    for source, target, attrs in G.edges(data=True):
        relation = str(attrs.get("relation", "RELATED"))
        weight = float(attrs.get("weight", 1.0))
        net.add_edge(
            source,
            target,
            label=relation,
            title=f"{relation}, weight={weight:.3f}",
            value=max(weight, 0.1),
        )

    net.write_html(str(path), notebook=True)


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    metric_name: str,
    output_path: str,
) -> None:
    """Save a bar chart comparing one scalar graph metric."""
    if metric_name not in metrics_df.columns:
        raise ValueError(f"Metric '{metric_name}' is not present in metrics_df")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(metrics_df["graph"], metrics_df[metric_name], color="#3A7CA5")
    plt.title(f"Graph comparison by {metric_name}")
    plt.xlabel("Graph")
    plt.ylabel(metric_name)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
