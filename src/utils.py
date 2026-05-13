"""Utility helpers for paths, persistence, and logging."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV, creating parent directories when needed."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def save_json(data: dict[str, Any] | list[Any], path: str | Path) -> None:
    """Save JSON data using UTF-8 encoding."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_graph_gexf(G: nx.Graph, path: str | Path) -> None:
    """Save a graph in GEXF format for external graph tools."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    nx.write_gexf(G, output_path)


def is_kaggle_environment() -> bool:
    """Return True when the code is likely running inside a Kaggle notebook."""
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

