import pytest

from evals.metrics import first_hit, hits_at, mcnemar, paired_margin, summarize

GOLD = ("class_11_biology", 5)


def test_first_hit_reports_one_based_rank():
    ranked = [("class_11_biology", 4), GOLD, ("class_11_biology", 9)]
    assert first_hit(ranked, GOLD) == 2


def test_first_hit_returns_none_when_absent():
    assert first_hit([("class_10_science", 1)], GOLD) is None


def test_first_hit_takes_the_earliest_occurrence():
    assert first_hit([GOLD, GOLD], GOLD) == 1


def test_summarize_counts_recall_at_each_cutoff():
    score = summarize([1, 3, 11, None])
    assert score.n == 4
    assert score.recall[1] == 0.25
    assert score.recall[5] == 0.5  # ranks 1 and 3
    assert score.recall[10] == 0.5  # rank 11 falls outside
    assert score.mrr == (1 + 1 / 3) / 4


def test_summarize_handles_no_questions():
    score = summarize([])
    assert score.n == 0 and score.mrr == 0.0


def test_hits_at_marks_ranks_within_the_cutoff():
    assert hits_at([1, 5, 6, None], k=5) == [True, True, False, False]


def test_mcnemar_ignores_the_questions_both_arms_agree_on():
    # padding 200 questions both arms get right must not dilute the evidence
    a = [True, False] + [True] * 200
    b = [False, True] + [True] * 200
    only_a, only_b, p = mcnemar(a, b)
    assert (only_a, only_b) == (1, 1)
    assert p == 1.0  # one win each is no evidence at all


def test_mcnemar_calls_a_lopsided_split_significant():
    # 30 questions only b finds against 5 only a finds
    a = [True] * 5 + [False] * 30
    b = [False] * 5 + [True] * 30
    only_a, only_b, p = mcnemar(a, b)
    assert (only_a, only_b) == (5, 30)
    assert p < 0.01


def test_mcnemar_survives_two_identical_arms():
    assert mcnemar([True, False], [True, False]) == (0, 0, 1.0)


def test_paired_margin_widens_when_arms_disagree_more():
    # same +1 difference, but the second pair disagrees on far more questions,
    # so it pins the true difference down less well
    narrow = paired_margin(1, 2, 282)
    wide = paired_margin(40, 41, 282)
    assert wide > narrow


def test_paired_margin_shows_what_a_p_of_one_still_permits():
    # 12 vs 11 discordant is p=1.0, which alone says nothing about equivalence
    assert mcnemar([True] * 12 + [False] * 11, [False] * 12 + [True] * 11)[2] == 1.0
    assert paired_margin(11, 12, 282) == pytest.approx(3.33, abs=0.05)


def test_paired_margin_handles_no_questions():
    assert paired_margin(0, 0, 0) == 0.0
