"""Run every arm over every question and write the report.

Each arm sees the same questions and the same corpus; only retrieval differs.
"""

import sqlite3
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.metrics import (
    HEADER,
    KS,
    Score,
    first_hit,
    hits_at,
    mcnemar,
    paired_margin,
    summarize,
)
from evals.questions import load
from ncert_rag.core.registry import BOOKS
from ncert_rag.retrieve import ARMS
from ncert_rag.retrieve.expansion import Expansion
from ncert_rag.store import db

REPORT_PATH = Path(__file__).parent / "REPORT.md"
K = max(KS)
WORKERS = 8


def _ranks(retriever, questions: Sequence[dict]) -> dict[str, int | None]:
    out = {}
    for question in questions:
        hits = retriever.retrieve(question["question"], K)
        ranked = [(hit.book, hit.chapter) for hit in hits]
        out[question["id"]] = first_hit(ranked, (question["book"], question["chapter"]))
    return out


def _warm_expansion(arm: Expansion, questions: Sequence[dict]) -> None:
    """Pay the rewrite calls concurrently instead of one at a time."""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda q: arm.expand(q["question"]), questions))


def _table(title: str, rows: list[tuple[str, Score]]) -> str:
    header = " | ".join(f"R@{k}" for k in KS)
    lines = [
        f"### {title}",
        "",
        f"| arm | {header} | MRR | n |",
        "|---" * (len(KS) + 3) + "|",
    ]
    for name, score in rows:
        cells = " | ".join(f"{score.recall[k]:.1%}" for k in KS)
        lines.append(f"| {name} | {cells} | {score.mrr:.3f} | {score.n} |")
    lines.append(
        "\nDifferences between rows are not tested here. Every arm answers the "
        "same questions, so comparing two of them calls for a paired test; "
        "those are under Paired comparisons, over the whole question set only. "
        "Nothing in this table is tested per tier.\n"
    )
    return "\n".join(lines) + "\n"


def _scores(
    results: dict[str, dict[str, int | None]],
    questions: Sequence[dict],
    tier: str | None = None,
) -> list[tuple[str, Score]]:
    ids = [q["id"] for q in questions if tier is None or q["tier"] == tier]
    return [
        (name, summarize(ranks[qid] for qid in ids)) for name, ranks in results.items()
    ]


# The pairs the eval was built to answer, named up front. Picking each family's
# winner and then testing it on the same data flatters that winner and inflates
# every p below; fixing the pairs makes that impossible instead of apologising
# for it in a footnote.
CLAIMS = [
    ("bm25", "vector"),
    ("bm25", "hybrid"),
    ("vector_raw", "vector"),
    ("vector", "hybrid"),
    ("hybrid", "expansion_hybrid"),
]


def _compare(results, questions, a: str, b: str):
    """Paired R@5 comparison: gap in points, both discordant counts, p, margin.

    The gap alone cannot say whether an arm is better. Two arms can differ by
    three points because one genuinely retrieves more, or because a handful of
    questions fell the other way; only the questions they disagree on tell them
    apart, which is what the paired test counts. The margin comes along because
    a high p on its own does not mean the arms are equal.
    """
    if a not in results or b not in results:
        return None
    ids = [q["id"] for q in questions]
    hits_a = hits_at(results[a][qid] for qid in ids)
    hits_b = hits_at(results[b][qid] for qid in ids)
    only_a, only_b, p = mcnemar(hits_a, hits_b)
    gap = (sum(hits_b) - sum(hits_a)) / len(ids) * 100
    return gap, only_a, only_b, p, paired_margin(only_a, only_b, len(ids))


def _format_p(p: float) -> str:
    """Never print 0.0000. A p that small is a bound, not an exact zero."""
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def _claims(results, questions) -> str:
    """The paired tests, as numbers.

    What they mean is written by hand in EVALUATION.md. An earlier version of
    this file generated the reading too, in interpolated English -- which went
    stale and started contradicting the table above it, because nobody proof-
    reads a format string.
    """
    scores = dict(_scores(results, questions))
    lines = [
        "## Paired comparisons",
        "",
        "Every arm answers the same questions, so each pair is tested with "
        "McNemar over the questions the two disagree about -- the only ones "
        "carrying evidence. `disagree` is those counts, b's wins first.",
        "",
        "| comparison | R@5 | gap | disagree | p | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for a, b in CLAIMS:
        result = _compare(results, questions, a, b)
        if result is None:
            continue
        gap, only_a, only_b, p, margin = result
        lines.append(
            f"| `{b}` over `{a}` | {scores[a].recall[5]:.1%} -> "
            f"{scores[b].recall[5]:.1%} | {gap:+.1f} | {only_b}/{only_a} | "
            f"{_format_p(p)} | [{gap - margin:+.1f}, {gap + margin:+.1f}] |"
        )
    lines.append(
        "\nFive pre-specified tests sharing no multiple-comparison correction, "
        "so a Bonferroni floor would be p<0.01 here. A high p is not evidence "
        "that two arms are equal: read the interval, which is what it can "
        "still hide.\n"
    )
    return "\n".join(lines) + "\n"


def _corpus_note(conn: sqlite3.Connection) -> str:
    counts = {
        source: conn.execute(
            "SELECT COUNT(*) n FROM chunks WHERE source = ?", (source,)
        ).fetchone()["n"]
        for source in ("parsed", "raw")
    }
    chapters = conn.execute(
        "SELECT COUNT(DISTINCT book || ':' || chapter) n FROM chunks"
    ).fetchone()["n"]
    return (
        f"{len(BOOKS)} books, {chapters} chapters, "
        f"{counts['parsed']} section-aware chunks, {counts['raw']} raw windows. "
        f"Embeddings: BAAI/bge-small-en-v1.5, local."
    )


def _check_overwrite(arm_names: Sequence[str], force: bool) -> None:
    """Refuse to quietly drop arms the existing report already holds.

    The report is the only record of the expansion arms, which need a model
    proxy and cannot always be re-run. Writing it is an unconditional
    overwrite, so the documented offline command
    (`--arms bm25,vector,vector_raw,hybrid`) silently deletes those rows and
    the serving-cost section `evals.cost` appends.
    """
    if force or not REPORT_PATH.exists():
        return
    present = {name for name in ARMS if f"| {name} |" in REPORT_PATH.read_text()}
    lost = sorted(present - set(arm_names))
    if lost:
        raise SystemExit(
            f"{REPORT_PATH.name} holds arms this run does not: {', '.join(lost)}.\n"
            "Writing it would drop them, and any serving-cost section with them.\n"
            "Pass --force to overwrite anyway, or re-run every arm with\n"
            f"  --arms {','.join(ARMS)}\n"
            "(which needs the model proxy the expansion arms call)."
        )


def main(arm_names: Sequence[str], force: bool = False) -> None:
    _check_overwrite(arm_names, force)
    conn = db.connect()
    questions = load()
    results: dict[str, dict[str, int | None]] = {}

    for name in arm_names:
        arm = ARMS[name](conn)
        if isinstance(arm, Expansion):
            print(f"{name}: expanding {len(questions)} queries...")
            _warm_expansion(arm, questions)
        print(f"{name}: retrieving...")
        results[name] = _ranks(arm, questions)

    parts = [
        "# NCERT retrieval eval",
        "",
        _corpus_note(conn),
        "",
        f"Arms: {', '.join(arm_names)}. Questions are chapter exercise questions "
        "reworded the way a student asks. Graded at chapter level: a hit means "
        "the arm surfaced a chunk from the chapter the question came from.",
        "",
        _table("All questions", _scores(results, questions)),
    ]
    for tier in ("clean", "fragmented"):
        if any(q["tier"] == tier for q in questions):
            parts.append(
                _table(f"{tier.title()} tier", _scores(results, questions, tier))
            )
    parts.append(_claims(results, questions))

    REPORT_PATH.write_text("\n".join(parts))
    print(f"\nWrote {REPORT_PATH}")

    print(f"\n{'arm':<22}{HEADER}")
    for name, score in _scores(results, questions):
        print(f"{name:<22}{score.row()}")


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    from ncert_rag.retrieve import OFFLINE

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated. Without a model, use: {','.join(OFFLINE)}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the report even if it holds arms this run does not",
    )
    args = parser.parse_args()
    main([name.strip() for name in args.arms.split(",")], force=args.force)
