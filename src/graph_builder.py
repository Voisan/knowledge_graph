"""Builders for similarity, citation, LLM, and hybrid paper graphs."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import pandas as pd


SYMMETRIC_LLM_RELATIONS = {"SIMILAR_TO", "SAME_TOPIC", "COMPARES_WITH"}


def _add_paper_nodes(G: nx.Graph, papers_df: pd.DataFrame) -> None:
    """Add paper nodes with common attributes."""
    for _, row in papers_df.iterrows():
        G.add_node(
            int(row["paper_id"]),
            title=str(row.get("title", "")),
            filename=str(row.get("filename", "")),
        )


def build_similarity_graph(
    papers_df: pd.DataFrame,
    candidate_pairs: Iterable[dict],
) -> nx.Graph:
    """Build an undirected graph from embedding similarity candidate pairs."""
    G = nx.Graph()
    _add_paper_nodes(G, papers_df)
    for edge in candidate_pairs:
        G.add_edge(
            int(edge["source"]),
            int(edge["target"]),
            relation="SIMILAR_TO",
            weight=float(edge.get("similarity", edge.get("weight", 1.0))),
            source_method="embedding_similarity",
        )
    return G


def build_citation_graph(
    papers_df: pd.DataFrame,
    citation_edges: Iterable[dict],
) -> nx.DiGraph:
    """Build a directed citation graph."""
    G = nx.DiGraph()
    _add_paper_nodes(G, papers_df)
    for edge in citation_edges:
        G.add_edge(
            int(edge["source"]),
            int(edge["target"]),
            relation=edge.get("relation", "CITES"),
            weight=float(edge.get("weight", 1.0)),
            source_method=edge.get("source_method", "citation"),
        )
    return G


def build_llm_graph(papers_df: pd.DataFrame, llm_edges: Iterable[dict]) -> nx.DiGraph:
    """Build a directed graph from LLM-classified semantic relations."""
    G = nx.DiGraph()
    _add_paper_nodes(G, papers_df)
    for edge in llm_edges:
        relation = edge.get("relation", "NO_RELATION")
        if relation == "NO_RELATION":
            continue
        source = int(edge["source"])
        target = int(edge["target"])
        edge_attrs = dict(
            relation=relation,
            weight=float(edge.get("confidence", edge.get("weight", 1.0))),
            source_method=edge.get("source_method", "llm"),
        )
        G.add_edge(source, target, **edge_attrs)
        if relation in SYMMETRIC_LLM_RELATIONS:
            G.add_edge(target, source, **edge_attrs)
    return G


def build_hybrid_graph(
    papers_df: pd.DataFrame,
    similarity_edges: Iterable[dict],
    citation_edges: Iterable[dict],
    llm_edges: Iterable[dict],
) -> nx.MultiDiGraph:
    """Build a hybrid multidigraph with all edge construction methods."""
    G = nx.MultiDiGraph()
    _add_paper_nodes(G, papers_df)

    for edge in similarity_edges:
        G.add_edge(
            int(edge["source"]),
            int(edge["target"]),
            relation="SIMILAR_TO",
            weight=float(edge.get("similarity", edge.get("weight", 1.0))),
            source_method="embedding_similarity",
        )
        G.add_edge(
            int(edge["target"]),
            int(edge["source"]),
            relation="SIMILAR_TO",
            weight=float(edge.get("similarity", edge.get("weight", 1.0))),
            source_method="embedding_similarity",
        )

    for edge in citation_edges:
        G.add_edge(
            int(edge["source"]),
            int(edge["target"]),
            relation=edge.get("relation", "CITES"),
            weight=float(edge.get("weight", 1.0)),
            source_method=edge.get("source_method", "citation"),
        )

    for edge in llm_edges:
        relation = edge.get("relation", "NO_RELATION")
        if relation == "NO_RELATION":
            continue
        source = int(edge["source"])
        target = int(edge["target"])
        edge_attrs = dict(
            relation=relation,
            weight=float(edge.get("confidence", edge.get("weight", 1.0))),
            source_method=edge.get("source_method", "llm"),
        )
        G.add_edge(source, target, **edge_attrs)
        if relation in SYMMETRIC_LLM_RELATIONS:
            G.add_edge(target, source, **edge_attrs)

    return G
