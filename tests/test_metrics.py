from evals.metrics import first_hit, summarize

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
