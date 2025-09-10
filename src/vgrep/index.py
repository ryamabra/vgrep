"""Vector index: flat, exact, numpy-only.

We originally used FAISS here, but faiss-cpu and torch each bundle their own
copy of libomp.dylib, and loading both into one process aborts on macOS. For a
flat exact index FAISS was doing nothing numpy cannot: search is a single
matrix-vector product against L2-normalised rows, i.e. exact cosine ranking.

A 100k x 768 float32 matrix is ~300MB and scores in well under 100ms, so the
approximate structures (IVF, HNSW) that FAISS exists to provide are not needed
at this scale anyway. Dropping the dependency removes the crash and a
significant chunk of install weight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FlatIndex:
    """Exact inner-product search over normalised vectors."""

    def __init__(self, vectors: np.ndarray, ids: list[int]):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.ids = np.asarray(ids, dtype=np.int64)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def save(self, path: Path) -> None:
        np.savez(path, vectors=self.vectors, ids=self.ids)

    @classmethod
    def load(cls, path: Path) -> "FlatIndex | None":
        if not path.exists():
            return None
        data = np.load(path)
        return cls(data["vectors"], data["ids"].tolist())


def build(vectors: np.ndarray, ids: list[int], path: Path, dim: int) -> FlatIndex:
    if len(ids) == 0:
        vectors = np.zeros((0, dim), dtype=np.float32)
    idx = FlatIndex(vectors, ids)
    idx.save(path)
    return idx


def load(path: Path) -> FlatIndex | None:
    return FlatIndex.load(path)


def search(index: FlatIndex | None, query: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Returns (file_id, similarity) pairs, best first."""
    if index is None or index.ntotal == 0:
        return []

    q = np.asarray(query, dtype=np.float32).ravel()
    scores = index.vectors @ q

    k = min(k, index.ntotal)
    # argpartition finds the top k without fully sorting the rest.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]

    return [(int(index.ids[i]), float(scores[i])) for i in top]
