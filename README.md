# ncert-rag

Retrieval over NCERT science and mathematics textbooks (classes 10-12), built to
answer one question with measurements rather than assumption: **what should
actually serve student questions, and what does it cost?**

The corpus is small, the vocabulary is technical, and every chapter arrives as
its own PDF with its structure intact. Those three facts all favour plain
full-text search, so rather than assume embeddings are required, the repo ships
eight retrieval techniques over one shared corpus and measures them.

## The eight techniques

| arm | what it does |
| --- | --- |
| `bm25` | SQLite FTS5 over section-aware chunks |
| `vector` | cosine over embeddings of the same chunks |
| `vector_raw` | cosine over embeddings of fixed windows cut with **no** parsing |
| `hybrid` | `bm25` + `vector`, fused by reciprocal rank |
| `expansion_bm25` | an LLM rewrites the query into textbook vocabulary, then `bm25` |
| `expansion_vector` | same rewrite → `vector` |
| `expansion_raw` | same rewrite → `vector_raw` |
| `expansion_hybrid` | same rewrite → `hybrid` |

`vector` against `vector_raw` isolates what the structural parser is worth: same
text, same model, only the chunk boundaries differ. The `expansion_*` family all
share one rewrite model, so they isolate what the rewrite adds to each thing it
can search with.

Findings and the full numbers live in [`evals/EVALUATION.md`](evals/EVALUATION.md).

## Use it

Needs [uv](https://docs.astral.sh/uv/) and Python 3.14+.

| Command | |
| --- | --- |
| `uv sync` | create `.venv` and install dependencies |
| `uv run ncert-rag build` | download, parse and index all 17 books |
| `uv run ncert-rag search "..." --arm vector` | query one arm by hand |
| `uv run python -m evals.questions` | build the query set |
| `uv run python -m evals.run` | score every arm → `evals/REPORT.md` |
| `uv run python -m evals.cost` | latency and tokens → appends to the report |
| `uv run pytest` | run the tests |
| `uv run ruff check` | lint |

`build` is idempotent per book: one whose PDFs hash unchanged is skipped. The
`expansion_*` arms and the query-set generator need `NINEROUTER_API_KEY` in
`.env`; everything else runs offline (`--arms bm25,vector,vector_raw,hybrid`).

## Layout

Dependencies flow one way: `core` ← `ingest` → `store` ← `retrieve` ← `evals`.

- `core/`: dataclasses and the book registry, no internal imports
- `ingest/`: PDFs to rows via download, clean, parse, chunk, `pipeline.build()`
  - `parse/profile.py` induces each book's heading font gate; `parse/resolver.py`
    recovers the chapter number a file actually prints
- `store/`: SQLite schema, FTS5 index, and vector search over a numpy matrix
- `retrieve/`: the eight arms behind one `Retriever` protocol
- `services/`: the chat model and the embedding model
- `evals/`: query-set generation, metrics, runner, cost benchmark

## What the corpus looks like

17 books, 169 chapters. NCERT publishes one PDF per chapter, so chapter count,
order and page ranges come free from the file layout. Nothing is merged.

Three things about the source material shape the code:

**Books disagree on how headings look.** Biology sets section numbers two points
above body text; chemistry keeps them inline at body size and only bold; class 10
science sets them large but *not* bold and overprints each one several times. A
single hardcoded font rule cannot read all of them, so the gate is induced per
book from a sample of chapters.

**Part II files lie about their chapter numbers.** `chemistry_ii/chapter_01.pdf`
is really chapter 6, because NCERT numbers Part II on from Part I while the
downloader restarts at 01. The number is recovered from the section marks inside
the file (`6.1`, `6.2` and so on means chapter 6) rather than a lookup table.

**Mathematics does not extract cleanly.** Equations shred into fragments, so
maths books are tagged `fragmented` in the registry and the eval reports that
tier separately instead of averaging the problem away.

## A note on the eval

Two bugs in early versions produced confident, wrong results, and both are worth
knowing about before extending this:

- **Exercise questions are printed inside the chapters.** Leaving them in the
  index let every eval question match its own printed copy. BM25 scored 93.6%
  R@1 against that leak and 72.8% without it, which inverted the ranking of every
  arm. Exercise pages are now excluded from both chunk sets.
- **The query set must not be written by a model any arm uses.** When it was,
  each rewriter scored best on the questions its own family had authored, swinging
  the gap between two rewrite models by 11.6 points. The paraphraser is now a
  third family no arm uses.
