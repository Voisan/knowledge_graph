"""Rule-based extraction of citation edges from references sections."""

from __future__ import annotations

import re

import pandas as pd

from .preprocessing import extract_references_section, normalize_title


def extract_references(text: str) -> list[str]:
    """Split a raw text or references section into individual references."""
    if not isinstance(text, str) or not text.strip():
        return []

    references_text = extract_references_section(text) or text
    references_text = references_text.strip()
    chunks = re.split(r"\n\s*(?:\[\d+\]|\d+\.|\d+\))\s*", "\n" + references_text)
    references = [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks]
    references = [ref for ref in references if len(ref) >= 20]

    if len(references) <= 1:
        references = [
            re.sub(r"\s+", " ", ref).strip()
            for ref in re.split(r"\n{1,}", references_text)
            if len(ref.strip()) >= 20
        ]

    return references


def match_references_to_papers(
    references: list[str],
    papers_df: pd.DataFrame,
) -> list[dict[str, int | str | float]]:
    """Match references against known paper titles and create CITES edges.

    To keep the required function signature simple, the citing paper can be
    passed through papers_df.attrs["source_paper_id"]. If it is absent and the
    DataFrame contains a single unique paper_id, that paper is treated as the
    citation source. Otherwise source is set to -1.
    """
    if papers_df.empty or not references:
        return []

    source_id = int(papers_df.attrs.get("source_paper_id", -1))
    unique_ids = papers_df["paper_id"].dropna().unique() if "paper_id" in papers_df else []
    if source_id == -1 and len(unique_ids) == 1:
        source_id = int(unique_ids[0])

    normalized_refs = [normalize_title(ref) for ref in references]
    edges: dict[tuple[int, int], dict[str, int | str | float]] = {}

    for _, row in papers_df.iterrows():
        target_id = int(row["paper_id"])
        if source_id == target_id:
            continue

        title = str(row.get("title", ""))
        normalized_title = normalize_title(title)
        if len(normalized_title) < 10:
            continue

        matched = any(normalized_title in ref for ref in normalized_refs)
        if matched:
            edges[(source_id, target_id)] = {
                "source": source_id,
                "target": target_id,
                "relation": "CITES",
                "weight": 1.0,
                "source_method": "citation",
            }

    return list(edges.values())
