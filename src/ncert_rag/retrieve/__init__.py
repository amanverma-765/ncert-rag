import sqlite3
from collections.abc import Callable
from functools import partial

from ncert_rag.retrieve.base import Retriever
from ncert_rag.retrieve.bm25 import Bm25
from ncert_rag.retrieve.expansion import BASES, Expansion
from ncert_rag.retrieve.hybrid import Hybrid
from ncert_rag.retrieve.vector import Vector

ARMS: dict[str, Callable[[sqlite3.Connection], Retriever]] = {
    "bm25": Bm25,
    "vector": Vector,
    "vector_raw": partial(Vector, source="raw"),
    "hybrid": Hybrid,
    **{f"expansion_{base}": partial(Expansion, base=base) for base in BASES},
}

EXPANSION_ARMS = [name for name in ARMS if name.startswith("expansion_")]

# the arms that need no model, for running without a proxy up
OFFLINE = [name for name in ARMS if name not in EXPANSION_ARMS]

__all__ = ["ARMS", "EXPANSION_ARMS", "OFFLINE", "Retriever"]
