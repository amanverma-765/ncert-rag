"""What each arm costs to serve, alongside what it retrieves.

Expansion pays a model call per query; the vector arms pay once at index time.
Recall alone cannot price that, so latency and tokens are measured here too.

Queries run one at a time: a thread pool would measure pool throughput rather
than the latency one student waits through.
"""

import argparse
import random
import statistics
import time
from dataclasses import dataclass

import tiktoken

from evals.metrics import first_hit
from evals.questions import load
from evals.run import REPORT_PATH
from ncert_rag.core.paths import DB_PATH
from ncert_rag.retrieve import ARMS, EXPANSION_ARMS, expansion
from ncert_rag.retrieve.expansion import Expansion
from ncert_rag.services import embedder
from ncert_rag.store import db

SEED = 20260825
K = 10
CONTEXT_K = 5  # passages actually stuffed into the answer prompt

# cl100k is not this model's tokenizer, but it is a stable offline yardstick and
# the comparison between arms is what matters here, not the absolute figure
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    recall: float
    median: float
    p95: float
    context_tokens: float
    rewrite_tokens: float

    @property
    def total_tokens(self) -> float:
        return self.context_tokens + self.rewrite_tokens


@dataclass(frozen=True, slots=True)
class Measured:
    latencies: list[float]
    recall: float
    context_tokens: float  # mean tokens of the top-CONTEXT_K passages


def _measure(retriever, questions: list[dict]) -> Measured:
    """Time each query, score it, and weigh the context it hands the model.

    Context size varies by arm: fixed windows send more text per question than
    chunks that stop at a section boundary, and that is paid on every query.
    """
    times, ranks, tokens = [], [], []
    for question in questions:
        start = time.perf_counter()
        hits = retriever.retrieve(question["question"], K)
        times.append((time.perf_counter() - start) * 1000)
        ranks.append(
            first_hit(
                [(hit.book, hit.chapter) for hit in hits],
                (question["book"], question["chapter"]),
            )
        )
        context = "\n\n".join(hit.text for hit in hits[:CONTEXT_K])
        tokens.append(len(_ENCODING.encode(context)))

    found = sum(1 for rank in ranks if rank is not None and rank <= 5)
    return Measured(
        latencies=times,
        recall=found / len(ranks),
        context_tokens=statistics.mean(tokens),
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def _index_cost(conn) -> tuple[int, float, float]:
    """Vectors stored, megabytes on disk, and measured encode rate."""
    vectors = conn.execute("SELECT COUNT(*) n FROM embeddings").fetchone()["n"]
    megabytes = DB_PATH.stat().st_size / 1e6

    sample = [
        row["text"]
        for row in conn.execute("SELECT text FROM chunks ORDER BY id LIMIT 128")
    ]
    start = time.perf_counter()
    embedder.encode_documents(sample)
    per_second = len(sample) / (time.perf_counter() - start)
    return vectors, megabytes, per_second


def main(n: int) -> None:
    conn = db.connect()
    questions = load()
    rng = random.Random(SEED)
    sample = rng.sample(questions, min(n, len(questions)))

    print(f"Timing {len(sample)} queries per arm, serially...")
    rows = []
    for name, make in ARMS.items():
        # every arm starts cold, or the second expansion arm measured would
        # report the first one's cached rewrites as free
        expansion.clear_cache()
        arm = make(conn)
        measured = _measure(arm, sample)
        rewrite = 0.0
        if isinstance(arm, Expansion) and arm.calls:
            rewrite = (arm.input_tokens + arm.output_tokens) / arm.calls
        median = statistics.median(measured.latencies)
        rows.append(
            Row(
                name=name,
                recall=measured.recall,
                median=median,
                p95=_percentile(measured.latencies, 0.95),
                context_tokens=measured.context_tokens,
                rewrite_tokens=rewrite,
            )
        )
        print(
            f"  {name}: R@5 {measured.recall:.1%}, median {median:.1f} ms, "
            f"{measured.context_tokens + rewrite:.0f} tokens/query"
        )

    vectors, megabytes, per_second = _index_cost(conn)
    _append_report(rows, len(sample), vectors, megabytes, per_second)


def _append_report(
    rows: list[Row],
    n: int,
    vectors: int,
    megabytes: float,
    per_second: float,
) -> None:
    lines = [
        "",
        "## Serving cost",
        "",
        f"Accuracy and cost from the same {n} queries, run one at a time on CPU "
        "so the timings are what one student waits through rather than pool "
        "throughput.",
        "",
        f"No arm is free. Every one hands its top {CONTEXT_K} passages to the "
        "answering model, and how many tokens that is depends on the retrieval "
        "method: an arm returning fixed windows sends more text per question than "
        "one whose chunks stop at a section boundary. Expansion pays that same "
        "context cost plus a rewrite call on top. Counted with cl100k as a stable "
        "offline yardstick; the comparison between arms is the point, not the "
        "absolute number.",
        "",
        "| arm | R@5 | median | p95 | context tok | rewrite tok | total tok |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        rewrite = f"{row.rewrite_tokens:.0f}" if row.rewrite_tokens else "0"
        lines.append(
            f"| {row.name} | {row.recall:.1%} | {row.median:.1f} ms | "
            f"{row.p95:.1f} ms | {row.context_tokens:.0f} | {rewrite} | "
            f"{row.total_tokens:.0f} |"
        )

    lines += [
        "",
        f"One-time index cost: {vectors} vectors, {megabytes:.0f} MB for the whole "
        f"database, encoding at roughly {per_second:.0f} chunks/second on CPU. "
        "The lexical arms need none of it.",
        "",
        _tradeoff(rows),
    ]

    REPORT_PATH.write_text(REPORT_PATH.read_text() + "\n".join(lines) + "\n")
    print(f"\nAppended serving cost to {REPORT_PATH}")


def _tradeoff(rows: list[Row]) -> str:
    """What the halves say together, which none of them says alone."""
    by_name = {row.name: row for row in rows}
    if not {"bm25", "vector", "vector_raw"} <= by_name.keys():
        return ""

    bm25, vector, raw = by_name["bm25"], by_name["vector"], by_name["vector_raw"]
    best = max(rows, key=lambda row: row.recall)

    return (
        "### Reading these together\n\n"
        f"Dense retrieval buys {(vector.recall - bm25.recall) * 100:+.1f} points "
        f"over BM25 for {vector.median - bm25.median:+.1f} ms, paying once at "
        "index time rather than per query. That is the cheapest large gain in "
        f"the table. The best arm overall is `{best.name}` at {best.recall:.1%}, "
        f"and whether its {best.median:.0f} ms median is affordable is the whole "
        "question. An LLM rewrite sits on the critical path of every query "
        "unless it is cached.\n\n"
        "Context size is where the parser shows up. Section-aware chunks send "
        f"{vector.context_tokens:.0f} tokens per query against "
        f"{raw.context_tokens:.0f} for fixed windows, a difference of "
        f"{(1 - vector.context_tokens / max(raw.context_tokens, 1)) * 100:+.0f}% "
        "paid on every question forever, a cost the recall tables, where the "
        "two are level, never showed.\n\n" + _rewriters(rows)
    )


def _rewriters(rows: list[Row]) -> str:
    """What the rewrite is worth on top of each thing it can search with."""
    expansions = [row for row in rows if row.name in EXPANSION_ARMS]
    plain = {row.name: row for row in rows if row.name not in EXPANSION_ARMS}
    if not expansions or not plain:
        return ""

    # each expansion arm against the same base without the rewrite in front
    pairs = {
        "expansion_bm25": "bm25",
        "expansion_vector": "vector",
        "expansion_raw": "vector_raw",
        "expansion_hybrid": "hybrid",
    }
    lines = [
        "The rewrite is one LLM call in front of an ordinary search, so the "
        "question is what it adds to each search it could front:",
        "",
    ]
    for arm in expansions:
        base = plain.get(pairs.get(arm.name, ""))
        if base is None:
            continue
        gap = (arm.recall - base.recall) * 100
        lines.append(
            f"- `{arm.name}` {arm.recall:.1%} against `{base.name}` "
            f"{base.recall:.1%} alone: {gap:+.1f} points for "
            f"{arm.median / max(base.median, 0.1):.0f}x the latency and "
            f"{arm.total_tokens / max(base.total_tokens, 1):.1f}x the tokens."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=80, help="queries to time")
    main(parser.parse_args().n)
