"""Vector storage and brute-force search.

A few thousand chunks, so an exhaustive dot product over one numpy matrix
answers in low milliseconds. A vector server would buy nothing at this size.
"""

import sqlite3

import numpy as np

from ncert_rag.core.models import ChunkSource
from ncert_rag.services.embedder import DIMS


def save(conn: sqlite3.Connection, ids: list[int], vecs: np.ndarray) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (chunk_id, vec) VALUES (?, ?)",
        [(cid, vec.tobytes()) for cid, vec in zip(ids, vecs, strict=True)],
    )


def load(conn: sqlite3.Connection, source: ChunkSource) -> tuple[list[int], np.ndarray]:
    """Every stored vector for one chunk set, as an (n, DIMS) matrix."""
    rows = conn.execute(
        "SELECT e.chunk_id, e.vec FROM embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id WHERE c.source = ? ORDER BY e.chunk_id",
        (source,),
    ).fetchall()
    if not rows:
        return [], np.empty((0, DIMS), dtype=np.float32)

    ids = [row["chunk_id"] for row in rows]
    matrix = np.frombuffer(b"".join(row["vec"] for row in rows), dtype=np.float32)
    return ids, matrix.reshape(len(ids), DIMS)


def top_k(
    matrix: np.ndarray, ids: list[int], query: np.ndarray, k: int
) -> list[tuple[int, float]]:
    """Highest cosine similarity first. Both sides are already normalized."""
    if not ids:
        return []
    scores = matrix @ query
    best = np.argpartition(-scores, min(k, len(ids) - 1))[:k]
    best = best[np.argsort(-scores[best])]
    return [(ids[i], float(scores[i])) for i in best]
