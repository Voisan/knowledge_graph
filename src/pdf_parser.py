"""PDF parsing utilities for building a paper-level dataset."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import fitz
except RuntimeError as exc:
    if "static/" in str(exc):
        raise RuntimeError(
            "Detected the wrong 'fitz' package. PyMuPDF is imported as 'fitz', "
            "but a different PyPI package named 'fitz' is installed in this "
            "environment. Run: python -m pip uninstall -y fitz frontend && "
            "python -m pip install -U PyMuPDF"
        ) from exc
    raise
import pandas as pd
from tqdm.auto import tqdm

from .preprocessing import clean_text, extract_abstract, extract_references_section


PAPER_COLUMNS = [
    "paper_id",
    "filename",
    "title",
    "text",
    "abstract",
    "references_raw",
]


def extract_text_from_pdf(path: str) -> str:
    """Extract plain text from a PDF file with PyMuPDF."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages: list[str] = []
    try:
        with fitz.open(pdf_path) as document:
            for page in document:
                pages.append(page.get_text("text"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF {pdf_path}: {exc}") from exc

    return clean_text("\n".join(pages))


def _extract_title(text: str, filename: str) -> str:
    """Infer a paper title from the first non-empty lines of extracted text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates: list[str] = []

    for line in lines[:20]:
        if len(line) < 8 or len(line) > 220:
            continue
        if re.search(r"\b(abstract|keywords|arxiv|doi|copyright)\b", line, re.I):
            continue
        if len(line.split()) >= 3:
            candidates.append(line)

    return candidates[0] if candidates else Path(filename).stem


def parse_papers_from_folder(folder_path: str) -> pd.DataFrame:
    """Parse all PDF files in a folder into a paper DataFrame.

    The returned DataFrame contains paper_id, filename, title, text, abstract,
    and references_raw. Files that cannot be parsed are skipped with an error
    message stored in the text field.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"Warning: no PDF files found in {folder}")
        return pd.DataFrame(columns=PAPER_COLUMNS)

    records: list[dict[str, str | int]] = []

    for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Parsing PDFs")):
        try:
            text = extract_text_from_pdf(str(pdf_path))
            title = _extract_title(text, pdf_path.name)
            abstract = extract_abstract(text)
            references_raw = extract_references_section(text)
        except Exception as exc:
            text = ""
            title = pdf_path.stem
            abstract = ""
            references_raw = ""
            print(f"Warning: failed to parse {pdf_path.name}: {exc}")

        records.append(
            {
                "paper_id": idx,
                "filename": pdf_path.name,
                "title": title,
                "text": text,
                "abstract": abstract,
                "references_raw": references_raw,
            }
        )

    return pd.DataFrame.from_records(records, columns=PAPER_COLUMNS)
