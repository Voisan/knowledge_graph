"""Graph metric, community, centrality, and path explanation utilities."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd


def _simple_graph_for_metrics(G: nx.Graph) -> nx.Graph:
    """Convert graph variants to a simple graph for clustering/community metrics."""
    if G.is_directed():
        graph = nx.Graph(G)
    else:
        graph = nx.Graph(G)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _top_nodes(scores: dict[Any, float], limit: int = 10) -> list[dict[str, Any]]:
    """Return top scored nodes as serializable dictionaries."""
    return [
        {"paper_id": node, "score": float(score)}
        for node, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def _communities(G: nx.Graph) -> list[set[Any]]:
    """Return communities, falling back to singleton communities when needed."""
    graph = _simple_graph_for_metrics(G)
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [{node} for node in graph.nodes]
    try:
        return [set(community) for community in nx.community.greedy_modularity_communities(graph, weight="weight")]
    except Exception:
        return [{node} for node in graph.nodes]


def calculate_graph_metrics(G: nx.Graph, graph_name: str = "") -> dict[str, Any]:
    """Calculate scalar structural metrics for a graph."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    undirected = _simple_graph_for_metrics(G)

    degrees = dict(G.degree())
    average_degree = float(sum(degrees.values()) / n_nodes) if n_nodes else 0.0

    if undirected.number_of_nodes() > 0:
        components = nx.number_connected_components(undirected)
        average_clustering = nx.average_clustering(undirected, weight="weight") if n_edges else 0.0
    else:
        components = 0
        average_clustering = 0.0

    communities = _communities(undirected)

    return {
        "graph": graph_name,
        "number_of_nodes": n_nodes,
        "number_of_edges": n_edges,
        "density": float(nx.density(G)) if n_nodes > 1 else 0.0,
        "number_connected_components": components,
        "average_degree": average_degree,
        "average_clustering": float(average_clustering),
        "number_of_communities": len(communities),
        "community_sizes": sorted([len(community) for community in communities], reverse=True),
    }


def compare_graphs(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    """Calculate comparable scalar metrics for several graphs."""
    rows: list[dict[str, Any]] = []
    for graph_name, graph in graphs.items():
        metrics = calculate_graph_metrics(graph, graph_name)
        rows.append(
            {
                "graph": graph_name,
                "number_of_nodes": metrics["number_of_nodes"],
                "number_of_edges": metrics["number_of_edges"],
                "density": metrics["density"],
                "number_connected_components": metrics["number_connected_components"],
                "average_degree": metrics["average_degree"],
                "average_clustering": metrics["average_clustering"],
                "number_of_communities": metrics["number_of_communities"],
            }
        )
    return pd.DataFrame(rows)


def get_communities_df(G: nx.Graph) -> pd.DataFrame:
    """Return paper-to-community assignments for a graph."""
    rows: list[dict[str, Any]] = []
    for community_id, community in enumerate(_communities(G)):
        for node in sorted(community):
            attrs = G.nodes[node]
            rows.append(
                {
                    "community_id": community_id,
                    "paper_id": int(node),
                    "title": str(attrs.get("title", "")),
                    "filename": str(attrs.get("filename", "")),
                }
            )
    return pd.DataFrame(rows, columns=["community_id", "paper_id", "title", "filename"])


def get_community_summary_df(communities_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate community assignments into compact readable summaries."""
    columns = ["community_id", "papers_count", "papers"]
    if communities_df is None or communities_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for community_id, group in communities_df.groupby("community_id", sort=True):
        labels = group["title"].where(group["title"].astype(str).str.strip() != "", group["filename"])
        rows.append(
            {
                "community_id": int(community_id),
                "papers_count": int(len(group)),
                "papers": ", ".join(labels.astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def calculate_centrality_df(G: nx.Graph) -> pd.DataFrame:
    """Calculate degree centrality, betweenness centrality, and PageRank."""
    columns = [
        "paper_id",
        "title",
        "filename",
        "degree_centrality",
        "betweenness_centrality",
        "pagerank",
    ]
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    degree = nx.degree_centrality(G) if G.number_of_nodes() > 1 else {node: 0.0 for node in G.nodes}
    betweenness = (
        nx.betweenness_centrality(G, weight="weight")
        if G.number_of_nodes() > 1
        else {node: 0.0 for node in G.nodes}
    )
    try:
        pagerank = nx.pagerank(G, weight="weight")
    except Exception:
        pagerank = {node: 0.0 for node in G.nodes}

    rows = []
    for node, attrs in G.nodes(data=True):
        rows.append(
            {
                "paper_id": int(node),
                "title": str(attrs.get("title", "")),
                "filename": str(attrs.get("filename", "")),
                "degree_centrality": float(degree.get(node, 0.0)),
                "betweenness_centrality": float(betweenness.get(node, 0.0)),
                "pagerank": float(pagerank.get(node, 0.0)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("pagerank", ascending=False).reset_index(drop=True)


def explain_path(G: nx.Graph, source: int, target: int) -> pd.DataFrame:
    """Explain the shortest path between two papers as edge-by-edge steps."""
    columns = [
        "step",
        "source_id",
        "source_title",
        "relation",
        "target_id",
        "target_title",
        "weight",
        "source_method",
        "reason",
    ]
    try:
        path = nx.shortest_path(G, source=int(source), target=int(target))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for step, (source_id, target_id) in enumerate(zip(path[:-1], path[1:]), start=1):
        attrs = G.get_edge_data(source_id, target_id, default={})
        rows.append(
            {
                "step": step,
                "source_id": int(source_id),
                "source_title": str(G.nodes[source_id].get("title", "")),
                "relation": str(attrs.get("relation", "")),
                "target_id": int(target_id),
                "target_title": str(G.nodes[target_id].get("title", "")),
                "weight": float(attrs.get("weight", 0.0) or 0.0),
                "source_method": str(attrs.get("source_method", "")),
                "reason": str(attrs.get("reason", "")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def find_paper_id_by_filename(papers_df: pd.DataFrame, filename_part: str) -> int:
    """Find the first paper id whose filename contains the given text."""
    if papers_df is None or papers_df.empty:
        raise ValueError("papers_df is empty")
    mask = papers_df["filename"].astype(str).str.contains(filename_part, case=False, na=False)
    matches = papers_df[mask]
    if matches.empty:
        raise ValueError(f"No paper filename contains: {filename_part}")
    return int(matches.iloc[0]["paper_id"])


def find_paper_id_by_title(papers_df: pd.DataFrame, title_part: str) -> int:
    """Find the first paper id whose title contains the given text."""
    if papers_df is None or papers_df.empty:
        raise ValueError("papers_df is empty")
    mask = papers_df["title"].astype(str).str.contains(title_part, case=False, na=False)
    matches = papers_df[mask]
    if matches.empty:
        raise ValueError(f"No paper title contains: {title_part}")
    return int(matches.iloc[0]["paper_id"])
