"""Graph metric calculation and comparison utilities."""

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


def calculate_graph_metrics(G: nx.Graph) -> dict[str, Any]:
    """Calculate structural metrics for a graph."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    undirected = _simple_graph_for_metrics(G)

    degrees = dict(G.degree())
    average_degree = float(sum(degrees.values()) / n_nodes) if n_nodes else 0.0
    degree_centrality = nx.degree_centrality(G) if n_nodes > 1 else {node: 0.0 for node in G.nodes}

    if G.is_directed() and n_nodes > 0:
        try:
            pagerank = nx.pagerank(G, weight="weight")
        except Exception:
            pagerank = {node: 0.0 for node in G.nodes}
    else:
        pagerank = {node: 0.0 for node in G.nodes}

    if undirected.number_of_nodes() > 0:
        components = nx.number_connected_components(undirected)
        average_clustering = nx.average_clustering(undirected, weight="weight")
    else:
        components = 0
        average_clustering = 0.0

    try:
        communities = list(nx.community.greedy_modularity_communities(undirected, weight="weight"))
        community_sizes = sorted([len(community) for community in communities], reverse=True)
    except Exception:
        communities = []
        community_sizes = []

    return {
        "number_of_nodes": n_nodes,
        "number_of_edges": n_edges,
        "density": float(nx.density(G)) if n_nodes > 1 else 0.0,
        "number_connected_components": components,
        "average_degree": average_degree,
        "average_clustering": float(average_clustering),
        "degree_centrality": {node: float(score) for node, score in degree_centrality.items()},
        "pagerank": {node: float(score) for node, score in pagerank.items()},
        "top_10_nodes_by_degree": _top_nodes({node: float(degree) for node, degree in degrees.items()}),
        "top_10_nodes_by_pagerank": _top_nodes(pagerank),
        "number_of_communities": len(communities),
        "community_sizes": community_sizes,
    }


def compare_graphs(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    """Calculate comparable scalar metrics for several graphs."""
    rows: list[dict[str, Any]] = []
    for graph_name, graph in graphs.items():
        metrics = calculate_graph_metrics(graph)
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

