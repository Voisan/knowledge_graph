"""Text cleaning and section extraction utilities for scientific papers."""

from __future__ import annotations

import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize whitespace and remove common PDF extraction artifacts."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """Return at most max_chars characters from text."""
    if not isinstance(text, str):
        return ""
    if max_chars <= 0:
        return ""
    return text[:max_chars].strip()


def extract_abstract(text: str) -> str:
    """Extract an abstract section using simple heading-based heuristics."""
    if not isinstance(text, str) or not text.strip():
        return ""

    pattern = re.compile(
        r"(?is)\babstract\b\s*[:.\-]?\s*(.+?)(?=\n\s*(?:1\.?\s*)?"
        r"(?:introduction|keywords|index terms|background|related work)\b)"
    )
    match = pattern.search(text)
    if match:
        return clean_text(match.group(1))[:3000]

    short_pattern = re.compile(r"(?is)\babstract\b\s*[:.\-]?\s*(.{200,2500})")
    match = short_pattern.search(text)
    return clean_text(match.group(1))[:3000] if match else ""


def extract_references_section(text: str) -> str:
    """Extract the references or bibliography section from paper text."""
    if not isinstance(text, str) or not text.strip():
        return ""

    pattern = re.compile(
        r"(?is)\n\s*(references|bibliography|литература|список литературы)\s*\n(.+)$"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return ""

    return clean_text(matches[-1].group(2))


def normalize_title(title: str) -> str:
    """Normalize a title for fuzzy matching against references."""
    if not isinstance(title, str):
        return ""

    title = Path(title).stem
    title = title.lower()
    title = re.sub(r"[_\-]+", " ", title)
    title = re.sub(r"[^a-zа-яё0-9\s]", " ", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title)
    return title.strip()
