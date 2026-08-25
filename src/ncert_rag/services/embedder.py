"""Local sentence embeddings for the vector arms.

Runs on CPU, so re-running the eval costs nothing. Vectors are L2-normalized
at encode time, making cosine similarity a plain dot product later.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims
DIMS = 384

# BGE is trained with an asymmetric prefix: queries get it, passages do not
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def _load() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)
    return _model


def encode_documents(texts: list[str], batch: int = 64) -> np.ndarray:
    return (
        _load()
        .encode(
            texts,
            batch_size=batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        .astype(np.float32)
    )


def encode_query(text: str) -> np.ndarray:
    return (
        _load()
        .encode(
            _QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        .astype(np.float32)
    )
