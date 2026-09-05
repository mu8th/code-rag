"""Vector retrieval over an embedded chunk index.

Pure numpy: L2-normalize the stored matrix once, then cosine-similarity a query
vector against it and return the top-k. Keeping this dependency-free (numpy is
the only requirement) means it is trivially testable and fast for the index
sizes a codebase produces.
"""

from __future__ import annotations

import numpy as np

from ..models import Chunk


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows of ``matrix`` (zero rows stay zero)."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(matrix / norms, dtype=np.float32)


class VectorIndex:
    """An in-memory cosine index over a set of chunks and their embeddings."""

    def __init__(self, chunks: list[Chunk], matrix: np.ndarray) -> None:
        if len(chunks) != matrix.shape[0]:
            raise ValueError(
                f"chunk count {len(chunks)} != matrix rows {matrix.shape[0]}"
            )
        self._chunks = chunks
        self._matrix = _normalize(matrix)

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    def query(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return the top-``k`` ``(chunk_position, score)`` pairs for ``vector``.

        Args:
            vector: A 1-D embedding of the same dimension as the index.
            k: How many results to return.

        Returns:
            Up to ``k`` ``(position, cosine_score)`` tuples, highest score first.
            Positions index into the original chunk list.
        """
        if self._matrix.size == 0:
            return []
        q = _normalize(vector.reshape(1, -1))[0]
        scores = self._matrix @ q
        k = max(0, min(k, len(scores)))
        if k == 0:
            return []
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(int(pos), float(scores[pos])) for pos in top]

    def chunks_at(self, positions: list[int]) -> list[Chunk]:
        """Return the :class:`Chunk` objects at the given positions."""
        return [self._chunks[p] for p in positions]
