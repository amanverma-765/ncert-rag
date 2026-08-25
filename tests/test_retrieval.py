from ncert_rag.retrieve.base import RRF_K, rrf
from ncert_rag.retrieve.bm25 import fts_query


def test_rrf_rewards_agreement_between_arms():
    # 2 ranks high in both lists; 1 tops one but trails the other
    fused = rrf([[1, 2, 3], [2, 3, 1]], k=3)
    assert fused[0][0] == 2


def test_rrf_keeps_a_chunk_only_one_arm_found():
    assert {cid for cid, _ in rrf([[1], [2]], k=5)} == {1, 2}


def test_rrf_scores_by_rank_not_by_score():
    [(_, score)] = rrf([[9]], k=1)
    assert score == 1 / (RRF_K + 1)


def test_rrf_truncates_to_k():
    assert len(rrf([[1, 2, 3, 4]], k=2)) == 2


def test_fts_query_quotes_terms_and_drops_stopwords():
    assert fts_query("What is phyllotaxy?") == '"phyllotaxy"'


def test_fts_query_survives_punctuation_that_would_break_match():
    # quotes, parens and bare FTS5 operators must not reach MATCH as syntax
    assert fts_query('Explain "SN2" reaction (i) NOT AND') == '"SN2" OR "reaction"'


def test_fts_query_is_empty_when_nothing_survives():
    assert fts_query("what is the of") == ""
