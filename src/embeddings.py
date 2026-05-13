"""Embedding computation and candidate pair selection."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """Load a SentenceTransformer embedding model.

    The explicit CPU device keeps local notebook runs predictable on machines
    without CUDA and avoids slow device auto-detection surprises.
    """
    return SentenceTransformer(
        model_name,
        device="cpu",
        model_kwargs={"use_safetensors": False},
    )


def compute_embeddings(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Compute dense embeddings for a list of texts."""
    safe_texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
    if not safe_texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = model.encode(
        safe_texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute a cosine similarity matrix from embeddings."""
    if embeddings.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return cosine_similarity(embeddings).astype(np.float32)


def get_candidate_pairs(
    similarity_matrix: np.ndarray,
    threshold: float = 0.55,
    top_k: int = 5,
) -> list[dict[str, int | float]]:
    """Return unique top-k similar paper pairs above a similarity threshold."""
    n_items = similarity_matrix.shape[0]
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be a square matrix")

    pairs: dict[tuple[int, int], float] = {}
    for source_idx in range(n_items):
        similarities = similarity_matrix[source_idx].copy()
        similarities[source_idx] = -np.inf
        candidate_indices = np.argsort(similarities)[::-1][:top_k]

        for target_idx in candidate_indices:
            score = float(similarities[target_idx])
            if score < threshold:
                continue
            pair_key = tuple(sorted((source_idx, int(target_idx))))
            pairs[pair_key] = max(score, pairs.get(pair_key, float("-inf")))

    return [
        {"source": source, "target": target, "similarity": score}
        for (source, target), score in sorted(pairs.items())
    ]
