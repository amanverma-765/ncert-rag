from ncert_rag.retrieve import ARMS, EXPANSION_ARMS, OFFLINE, REWRITE_BASE
from ncert_rag.retrieve.expansion import BASES


def test_every_base_gets_an_expansion_arm():
    # adding a base should be enough to get it into the eval
    assert len(EXPANSION_ARMS) == len(BASES)
    assert set(EXPANSION_ARMS) <= set(ARMS)


def test_every_expansion_arm_names_the_plain_arm_it_fronts():
    # both evals compare an expansion arm against its own base; a base added
    # without a mapping would silently drop out of those comparisons
    assert set(REWRITE_BASE) == set(EXPANSION_ARMS)
    assert set(REWRITE_BASE.values()) <= set(ARMS) - set(EXPANSION_ARMS)


def test_offline_arms_need_no_model():
    assert set(OFFLINE) == set(ARMS) - set(EXPANSION_ARMS)
    assert "bm25" in OFFLINE and "vector" in OFFLINE


def test_expansion_covers_lexical_dense_and_fused_bases():
    # the point of these arms is what the rewrite feeds, so all three must run
    assert {"expansion_bm25", "expansion_vector", "expansion_hybrid"} <= set(ARMS)


def test_arms_are_constructible_without_a_connection_for_offline():
    # constructing must not require credentials for the offline arms
    assert callable(ARMS["bm25"]) and callable(ARMS["vector_raw"])
