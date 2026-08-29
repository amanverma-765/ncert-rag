"""Chroma-backed vector search, driven against an in-memory collection.

The id coercion is the reason this file exists: Chroma returns string ids,
`db.hits` matches them against INTEGER primary keys, and TEXT never equals
INTEGER in SQLite. Getting it wrong is not an error, it is every vector arm
quietly reporting zero recall.
"""

import chromadb
import numpy as np
import pytest

from ncert_rag.store import vectors


@pytest.fixture
def collection(monkeypatch, request):
    # EphemeralClient shares one in-memory system across instantiations, so a
    # fixed collection name would carry data between tests. Name it per test.
    client = chromadb.EphemeralClient()
    name = f"chunks-{abs(hash(request.node.name)):x}"[:60]
    made = client.create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    monkeypatch.setattr(vectors, "collection", lambda source: made)
    yield made
    client.delete_collection(name)


def unit(rows: int, dims: int = 8) -> np.ndarray:
    matrix = np.random.default_rng(0).random((rows, dims), dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_ids_come_back_as_integers_not_strings(collection):
    vectors.save("parsed", [11, 22, 33], unit(3))
    found = vectors.top_k("parsed", unit(3)[0], k=3)
    assert all(isinstance(cid, int) for cid, _ in found)
    assert {cid for cid, _ in found} == {11, 22, 33}


def test_score_is_the_similarity_not_the_distance(collection):
    matrix = unit(4)
    vectors.save("parsed", [1, 2, 3, 4], matrix)
    query = matrix[2]
    found = vectors.top_k("parsed", query, k=4)
    # 1 - cosine distance is the dot product, because both sides are normalized
    for cid, score in found:
        assert score == pytest.approx(float(matrix[cid - 1] @ query), abs=1e-5)


def test_best_match_ranks_first(collection):
    matrix = unit(5)
    vectors.save("parsed", [1, 2, 3, 4, 5], matrix)
    found = vectors.top_k("parsed", matrix[3], k=5)
    assert found[0][0] == 4
    assert [s for _, s in found] == sorted((s for _, s in found), reverse=True)


def test_asking_for_more_than_exists_returns_what_exists(collection):
    vectors.save("parsed", [1, 2], unit(2))
    assert len(vectors.top_k("parsed", unit(2)[0], k=50)) == 2


def test_empty_collection_returns_nothing(collection):
    assert vectors.top_k("parsed", unit(1)[0], k=5) == []


def test_zero_k_never_reaches_the_index(collection):
    vectors.save("parsed", [1], unit(1))
    assert vectors.top_k("parsed", unit(1)[0], k=0) == []


def test_saving_the_same_batch_twice_is_not_an_error(collection):
    # a build killed mid-batch re-runs it; upsert makes that idempotent
    vectors.save("parsed", [1, 2], unit(2))
    vectors.save("parsed", [1, 2], unit(2))
    assert vectors.count("parsed") == 2


def test_stored_ids_reports_what_is_already_embedded(collection):
    vectors.save("parsed", [7, 8, 9], unit(3))
    assert vectors.stored_ids("parsed") == {7, 8, 9}
