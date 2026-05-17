"""Builders for embedding, LLM relation, and hybrid paper knowledge graphs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx
import pandas as pd


VALID_LLM_RELATIONS = {
    "SIMILAR_TO",
    "BASED_ON",
    "EXTENDS",
    "COMPARES_WITH",
    "SAME_TOPIC",
}
STRICT_CONFIDENCE_RELATIONS = {"SAME_TOPIC", "SIMILAR_TO"}
RELAXED_CONFIDENCE_RELATIONS = {"BASED_ON", "EXTENDS", "COMPARES_WITH"}


def _add_paper_nodes(G: nx.Graph, papers_df: pd.DataFrame) -> None:
    """Add paper nodes with common attributes."""
    if papers_df is None or papers_df.empty:
        return

    for _, row in papers_df.iterrows():
        G.add_node(
            int(row["paper_id"]),
            title=str(row.get("title", "")),
            filename=str(row.get("filename", "")),
            type="Paper",
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert scalar values to float while tolerating missing cells."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
            confidence=float(edge.get("similarity", edge.get("weight", 1.0))),
            candidate_similarity=float(edge.get("similarity", edge.get("weight", 1.0))),
            reason="Edge added by embedding similarity.",
            source_method="embedding",
        )
    return G


def filter_llm_relations(llm_relations_df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw LLM relation rows into reliable typed semantic edges.

    Rules:
    - ``NO_RELATION`` and ``ERROR`` are always removed.
    - ``SAME_TOPIC`` and ``SIMILAR_TO`` require confidence >= 0.7.
    - ``BASED_ON``, ``EXTENDS``, and ``COMPARES_WITH`` require confidence >= 0.5.
    """
    columns = [
        "source",
        "target",
        "relation",
        "confidence",
        "reason",
        "candidate_similarity",
        "source_method",
    ]
    if llm_relations_df is None or llm_relations_df.empty:
        return pd.DataFrame(columns=columns)

    df = llm_relations_df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = "" if column in {"relation", "reason", "source_method"} else 0.0

    df["relation"] = df["relation"].fillna("").astype(str).str.upper().str.strip()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["candidate_similarity"] = pd.to_numeric(
        df["candidate_similarity"],
        errors="coerce",
    ).fillna(0.0)

    valid_mask = df["relation"].isin(VALID_LLM_RELATIONS)
    strict_mask = df["relation"].isin(STRICT_CONFIDENCE_RELATIONS) & (df["confidence"] >= 0.7)
    relaxed_mask = df["relation"].isin(RELAXED_CONFIDENCE_RELATIONS) & (df["confidence"] >= 0.5)
    filtered = df[valid_mask & (strict_mask | relaxed_mask)].copy()
    return filtered[columns].reset_index(drop=True)


def build_llm_relation_graph(
    papers_df: pd.DataFrame,
    llm_relations_df: pd.DataFrame,
) -> nx.Graph:
    """Build an undirected LLM relation graph from filtered relation rows."""
    G = nx.Graph()
    _add_paper_nodes(G, papers_df)

    filtered = filter_llm_relations(llm_relations_df)
    for _, edge in filtered.iterrows():
        relation = str(edge.get("relation", "")).upper()
        if relation not in VALID_LLM_RELATIONS:
            continue
        source = int(edge["source"])
        target = int(edge["target"])
        confidence = _safe_float(edge.get("confidence"), 0.0)
        candidate_similarity = _safe_float(edge.get("candidate_similarity"), 0.0)
        G.add_edge(
            source,
            target,
            relation=relation,
            confidence=confidence,
            candidate_similarity=candidate_similarity,
            weight=confidence,
            reason=str(edge.get("reason", "")),
            source_method="llm",
        )
    return G


def build_weighted_hybrid_graph(
    papers_df: pd.DataFrame,
    embedding_graph: nx.Graph,
    llm_relations_df: pd.DataFrame,
    alpha: float = 0.5,
) -> nx.Graph:
    """Build the final weighted hybrid graph.

    LLM edges have priority over embedding-only edges. For LLM edges the weight
    is calculated as:

    ``w_ij = alpha * sim_ij + (1 - alpha) * conf_ij``
    """
    G = nx.Graph()
    _add_paper_nodes(G, papers_df)
    alpha = max(0.0, min(1.0, float(alpha)))

    filtered = filter_llm_relations(llm_relations_df)
    for _, edge in filtered.iterrows():
        source = int(edge["source"])
        target = int(edge["target"])
        confidence = _safe_float(edge.get("confidence"), 0.0)
        candidate_similarity = _safe_float(edge.get("candidate_similarity"), 0.0)
        weight = alpha * candidate_similarity + (1.0 - alpha) * confidence
        G.add_edge(
            source,
            target,
            relation=str(edge.get("relation", "")).upper(),
            confidence=confidence,
            candidate_similarity=candidate_similarity,
            weight=weight,
            reason=str(edge.get("reason", "")),
            source_method="llm",
        )

    if embedding_graph is None:
        return G

    for source, target, attrs in embedding_graph.edges(data=True):
        if G.has_edge(source, target):
            continue
        similarity = _safe_float(attrs.get("candidate_similarity", attrs.get("weight")), 0.0)
        G.add_edge(
            int(source),
            int(target),
            relation="SIMILAR_TO",
            confidence=similarity,
            candidate_similarity=similarity,
            weight=similarity,
            source_method="embedding",
            reason="Edge added by embedding similarity.",
        )

    return G


def graph_to_nodes_df(G: nx.Graph) -> pd.DataFrame:
    """Export graph nodes to a paper-level DataFrame."""
    rows = [
        {
            "paper_id": int(node),
            "title": str(attrs.get("title", "")),
            "filename": str(attrs.get("filename", "")),
        }
        for node, attrs in G.nodes(data=True)
    ]
    return pd.DataFrame(rows, columns=["paper_id", "title", "filename"])


def graph_to_edges_df(G: nx.Graph) -> pd.DataFrame:
    """Export graph edges to a DataFrame with semantic edge attributes."""
    rows = [
        {
            "source": int(source),
            "target": int(target),
            "relation": str(attrs.get("relation", "")),
            "weight": _safe_float(attrs.get("weight"), 0.0),
            "confidence": _safe_float(attrs.get("confidence"), 0.0),
            "candidate_similarity": _safe_float(attrs.get("candidate_similarity"), 0.0),
            "source_method": str(attrs.get("source_method", "")),
            "reason": str(attrs.get("reason", "")),
        }
        for source, target, attrs in G.edges(data=True)
    ]
    columns = [
        "source",
        "target",
        "relation",
        "weight",
        "confidence",
        "candidate_similarity",
        "source_method",
        "reason",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_llm_graph(papers_df: pd.DataFrame, llm_edges: Iterable[dict]) -> nx.Graph:
    """Backward-compatible wrapper for the LLM relation graph builder."""
    return build_llm_relation_graph(papers_df, pd.DataFrame(llm_edges))
