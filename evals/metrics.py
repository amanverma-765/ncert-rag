"""Scoring. No I/O, so it can be tested on hand-written rankings.

Grading is at chapter level: an exercise question is labelled only with its
chapter, and raw windows have no section to compare against.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Label = tuple[str, int]  # (book slug, chapter number)

KS = (1, 5, 10)

# Reported next to R@5 in every table. At n=282 the interval is around six
# points wide, wider than most differences between arms here.
CONFIDENCE = 1.96  # two-sided 95%


def first_hit(ranked: Sequence[Label], gold: Label) -> int | None:
    """1-based rank of the first chunk from the gold chapter, if any."""
    for position, label in enumerate(ranked, start=1):
        if label == gold:
            return position
    return None


@dataclass(frozen=True, slots=True)
class Score:
    n: int
    recall: dict[int, float]
    mrr: float

    @property
    def margin(self) -> float:
        """Half-width of the 95% interval on R@5, in points.

        Two arms whose R@5 differ by less than roughly the sum of their
        margins are not distinguishable by this eval.
        """
        if not self.n:
            return 0.0
        p = self.recall[5]
        return CONFIDENCE * math.sqrt(p * (1 - p) / self.n) * 100

    def row(self) -> str:
        cells = " ".join(f"{self.recall[k]:>7.1%}" for k in KS)
        return f"{cells} {self.mrr:>7.3f} {self.margin:>7.1f} {self.n:>6}"


def summarize(ranks: Iterable[int | None]) -> Score:
    ranks = list(ranks)
    if not ranks:
        return Score(0, dict.fromkeys(KS, 0.0), 0.0)

    return Score(
        n=len(ranks),
        recall={
            k: sum(1 for r in ranks if r is not None and r <= k) / len(ranks)
            for k in KS
        },
        # truncated at the deepest cutoff, so MRR and recall@10 agree about
        # what counts as found
        mrr=sum(1.0 / r for r in ranks if r is not None and r <= max(KS)) / len(ranks),
    )


HEADER = f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'+-R@5':>7} {'n':>6}"
