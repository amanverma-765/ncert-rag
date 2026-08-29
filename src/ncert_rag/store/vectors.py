"""Vector storage and search, backed by a persistent Chroma index.

One collection per chunk set, rather than one shared collection filtered by
`source`. Filtering an HNSW index loses recall by an amount that depends on
which subset is asked for, and `vector` against `vector_raw` exists precisely
to measure one difference between two chunk sets. A shared index would add a
second, uncontrolled one.
"""

import sqlite3
from collections.abc import Sequence
from functools import cache

import numpy as np

from ncert_rag.core.models import ChunkSource
from ncert_rag.core.paths import CHROMA_DIR

_NAMES: dict[str, str] = {"parsed": "chunks_parsed", "raw": "chunks_raw"}


@cache
def _client():
    # imported lazily: chromadb drags in onnxruntime and costs seconds to
    # import, while the lexical arms, the CLI and most of the tests never
    # touch a vector.
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@cache
def collection(source: ChunkSource):
    """The collection holding one chunk set's vectors.

    `embedding_function=None` because we hand Chroma vectors we encoded
    ourselves; letting it construct a default would pull an ONNX model down at
    query time and break the offline guarantee. The space has to be named:
    Chroma defaults to l2, not cosine.
    """
    return _client().get_or_create_collection(
        name=_NAMES[source],
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def save(source: ChunkSource, ids: Sequence[int], vecs: np.ndarray) -> None:
    """Upsert rather than add, so re-running a half-written batch is not an error."""
    collection(source).upsert(
        ids=[str(cid) for cid in ids],
        embeddings=[vec.tolist() for vec in vecs],
    )


def stored_ids(source: ChunkSource) -> set[int]:
    """Chunk ids that already have a vector. There is no join to do this with."""
    return {int(cid) for cid in collection(source).get(include=[])["ids"]}


def count(source: ChunkSource) -> int:
    return collection(source).count()


def drop_book(conn: sqlite3.Connection, slug: str) -> None:
    """Delete one book's vectors. Must run *before* its chunk rows go.

    `chunks.id` is a plain INTEGER PRIMARY KEY, so SQLite hands ids freed by a
    delete to whichever book is inserted next. A vector left behind under a
    reused id gets served for the wrong chapter, which is worse than a missing
    one, and nothing downstream would report it.
    """
    for source in _NAMES:
        ids = [
            str(row["id"])
            for row in conn.execute(
                "SELECT id FROM chunks WHERE book = ? AND source = ?", (slug, source)
            )
        ]
        if ids:
            collection(source).delete(ids=ids)


def top_k(source: ChunkSource, query: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Highest cosine similarity first, ids back as SQLite integers.

    Chroma reports cosine *distance*. Both sides are L2-normalized, so
    1 - distance is exactly the dot product the numpy matrix used to return,
    which keeps every arm higher-is-better and the printed scores comparable.

    The int() is load-bearing: Chroma ids are strings, `db.hits` matches them
    against INTEGER primary keys, and TEXT never equals INTEGER in SQLite. A
    missed cast is silent zero recall, not an error.
    """
    if k <= 0:
        return []
    result = collection(source).query(
        query_embeddings=[query.tolist()], n_results=k, include=["distances"]
    )
    return [
        (int(cid), 1.0 - float(distance))
        for cid, distance in zip(result["ids"][0], result["distances"][0], strict=True)
    ]
