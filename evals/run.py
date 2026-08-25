"""Run every arm over every question and write the report.

Each arm sees the same questions and the same corpus; only retrieval differs.
"""

import sqlite3
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.metrics import HEADER, KS, Score, first_hit, summarize
from evals.questions import load
from ncert_rag.core.registry import BOOKS
from ncert_rag.retrieve import ARMS, EXPANSION_ARMS
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
        f"| arm | {header} | MRR | ±R@5 | n |",
        "|---" * (len(KS) + 4) + "|",
    ]
    for name, score in rows:
        cells = " | ".join(f"{score.recall[k]:.1%}" for k in KS)
        lines.append(
            f"| {name} | {cells} | {score.mrr:.3f} | ±{score.margin:.1f} | {score.n} |"
        )
    widest = max((score.margin for _n, score in rows), default=0.0)
    lines.append(
        f"\n95% interval on R@5. Two arms differing by less than about "
        f"{2 * widest:.0f} points are not distinguishable here.\n"
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


def _best(scores: dict, names: list[str]) -> tuple[str, float] | None:
    """(name, R@5) of the strongest arm in a family, or None if none ran."""
    ranked = [(name, scores[name].recall[5]) for name in names if name in scores]
    return max(ranked, key=lambda pair: pair[1]) if ranked else None


def _verdict(results, questions) -> str:
    """Summarize what the numbers say.

    Compares the best arm of each family rather than fixed pairs, so a
    dominated arm never speaks for its whole approach.
    """
    scores = dict(_scores(results, questions))

    def recall(arm: str) -> float | None:
        return scores[arm].recall[5] if arm in scores else None

    bm25 = recall("bm25")
    dense = _best(scores, [n for n in scores if n.startswith("vector")])
    rewritten = _best(scores, [n for n in scores if n in EXPANSION_ARMS])

    lines = ["## Verdict", ""]

    if bm25 is not None and dense:
        name, value = dense
        gap = (value - bm25) * 100
        lines.append(
            f"- Do embeddings earn their place: best model-free dense arm "
            f"(`{name}`) {value:.1%} against BM25 {bm25:.1%} at R@5, a gap of "
            f"{gap:+.1f} points. "
            + (
                "Yes. Students do not phrase questions in textbook words, and "
                "that gap is what bridges it."
                if gap > 2
                else "No. Lexical search already covers this corpus."
            )
        )

    if dense and rewritten:
        (_dn, dense_value), (rname, rewrite_value) = dense, rewritten
        lines.append(
            f"- Is an LLM rewrite worth a call per query: `{rname}` "
            f"{rewrite_value:.1%} against {dense_value:.1%} without a model, "
            f"{(rewrite_value - dense_value) * 100:+.1f} points. Price it against "
            "the latency below before taking it."
        )

    vector, raw = recall("vector"), recall("vector_raw")
    if vector is not None and raw is not None:
        gap = (vector - raw) * 100
        lines.append(
            f"- Does the parser help retrieval: section-aware chunks {vector:.1%} "
            f"against raw windows {raw:.1%} ({gap:+.1f} points). "
            + (
                "Not on recall. Its payoff is smaller context and real section "
                "citations, both below."
                if abs(gap) <= 2
                else "Chunking strategy moves retrieval measurably."
            )
        )

    hybrid = recall("hybrid")
    if hybrid is not None and vector is not None and hybrid < vector:
        lines.append(
            f"- Hybrid fusion is dominated here: {hybrid:.1%} against {vector:.1%} "
            "for dense alone. Fusing in BM25's weak ranks costs more than it adds."
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


def main(arm_names: Sequence[str]) -> None:
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
    parts.append(_verdict(results, questions))

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
    main([name.strip() for name in parser.parse_args().arms.split(",")])
