"""Dense retrieval over one chunk set.

`vector` searches the section-aware chunks and `vector_raw` the fixed windows.
Same model and same cleaned text, so the gap between them measures chunking.
"""

import sqlite3

from ncert_rag.core.models import ChunkSource, RetrievalHit
from ncert_rag.services import embedder
from ncert_rag.store import db, vectors


class Vector:
    def __init__(self, conn: sqlite3.Connection, source: ChunkSource = "parsed"):
        self.conn = conn
        self.source = source
        self.name = "vector" if source == "parsed" else "vector_raw"
        # the whole matrix is a few MB; load once and reuse across queries
        self.ids, self.matrix = vectors.load(conn, source)

    def search(self, question: str, k: int) -> list[tuple[int, float]]:
        return vectors.top_k(self.matrix, self.ids, embedder.encode_query(question), k)

    def retrieve(self, question: str, k: int) -> list[RetrievalHit]:
        return db.hits(self.conn, self.search(question, k))
