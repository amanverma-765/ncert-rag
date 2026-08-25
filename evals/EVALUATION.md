# NCERT retrieval evaluation

Eight retrieval techniques measured over the same corpus of 17 NCERT science and
mathematics textbooks, to answer one question: **what should serve student
questions, and what does it cost?**

Run 2026-08-26. Reproduce with `ncert-rag build`, `python -m evals.questions`,
`python -m evals.run`, `python -m evals.cost`. Those commands regenerate
`REPORT.md`; this document is the written analysis of it.

---

## 1. Headline

| | technique | R@5 | median latency | tokens/query |
|---|---|---|---|---|
| **Best accuracy** | `expansion_hybrid` | **92.2%** | 4567 ms | 4532 |
| **Best value** | `hybrid` | 86.9% | **11.3 ms** | 2000 |
| Baseline | `bm25` | 75.2% | 3.7 ms | 2037 |

Retrieval without a model at query time tops out around 87%. The LLM rewrite
adds roughly 5 points on top of that, and costs about 400× the latency to get
them.

---

## 2. What was measured

**Corpus.** 17 books, 169 chapters, classes 10-12, English editions. Biology,
biotechnology, chemistry, computer science, informatics practices and
mathematics; physics is excluded pending formula reconstruction. NCERT publishes
one PDF per chapter, so chapter boundaries come from the file layout rather than
inference.

Two chunk sets are built from identical cleaned text, so any difference between
them is chunking and nothing else:

| chunk set | how it is cut | count |
|---|---|---|
| `parsed` | windows stop at printed section boundaries | 4,237 |
| `raw` | fixed ~350-word windows, 60-word overlap, structure ignored | 3,447 |

7,684 embeddings (`BAAI/bge-small-en-v1.5`, 384-dim, local CPU) in one 71 MB
SQLite file holding chunks, FTS5 index and vectors.

**Queries.** 1,231 exercise questions were extracted from chapter ends; 283 were
sampled at up to 2 per chapter, then reworded into how a student would actually
ask. One was dropped because the model could not produce a clean rewording,
leaving **282**. Gold label is the `(book, chapter)` the question was printed in.

Questions are used in reworded form only. Measuring on the printed wording
flatters lexical search badly: BM25 scored 89.8% on printed questions and 75.2%
once the textbook vocabulary was stripped out, and nobody types them as printed.

239 questions come from `clean`-tier books, 43 from the four `fragmented`
mathematics books whose equations shred on extraction.

**Metric.** Chapter-level recall@k: did any of the top k passages come from the
chapter the question was printed in. R@5 is the number that matters, the system
passes its top 5 passages to the answering model, so the answer needs to be *in*
the context, not ranked first.

---

## 3. The eight techniques

| arm | what it does |
|---|---|
| `bm25` | SQLite FTS5 over section-aware chunks |
| `vector` | cosine over embeddings of the same chunks |
| `vector_raw` | cosine over embeddings of unparsed fixed windows |
| `hybrid` | `bm25` + `vector` fused by reciprocal rank (k=60, depth 30) |
| `expansion_bm25` | LLM rewrites query into textbook vocabulary → `bm25` |
| `expansion_vector` | same rewrite → `vector` |
| `expansion_raw` | same rewrite → `vector_raw` |
| `expansion_hybrid` | same rewrite → `hybrid` |

The rewrite model is held constant (`ag/gemini-3.7-flash-medium`) across all four
expansion arms, this compares techniques, not models. The rewritten query keeps
the student's original words and appends textbook terms, so expansion can only
add recall, never trade it away.

---

## 4. Results (n=282)

| technique | R@1 | R@5 | R@10 | MRR | ±R@5 |
|---|---|---|---|---|---|
| bm25 | 53.9% | 75.2% | 81.2% | 0.633 | ±5.0 |
| vector | 57.8% | 84.0% | 89.4% | 0.689 | ±4.3 |
| vector_raw | 59.9% | 83.7% | 90.4% | 0.701 | ±4.3 |
| hybrid | 58.5% | 86.9% | 92.6% | 0.706 | ±3.9 |
| expansion_bm25 | 72.3% | 90.8% | 93.6% | 0.801 | ±3.4 |
| expansion_vector | 69.1% | 90.1% | **95.4%** | 0.788 | ±3.5 |
| expansion_raw | 72.3% | 90.4% | 93.6% | 0.801 | ±3.4 |
| **expansion_hybrid** | **74.1%** | **92.2%** | 94.7% | **0.821** | ±3.1 |

The ± column is the 95% interval on R@5. **Read this table in two groups, not as
a ranking.** The four model-free arms (75.2-86.9%) overlap each other; so do the
four expansion arms (90.1-92.2%). Only the gap *between* those groups clears the
intervals.

### Fragmented tier, the four mathematics books (n=43)

| technique | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| bm25 | 69.8% | 79.1% | 79.1% | 0.726 |
| vector | 65.1% | 83.7% | 88.4% | 0.729 |
| vector_raw | 55.8% | 83.7% | 86.0% | 0.662 |
| hybrid | 69.8% | 88.4% | 88.4% | 0.775 |
| **expansion_bm25** | 83.7% | **100.0%** | **100.0%** | 0.888 |
| expansion_vector | 79.1% | 97.7% | 97.7% | 0.865 |
| expansion_raw | 79.1% | 97.7% | 97.7% | 0.863 |
| expansion_hybrid | 83.7% | 97.7% | 100.0% | **0.894** |

Maths is where the rewrite matters most: it turns a shredded-equation corpus from
the worst tier into a perfectly retrievable one, because the rewrite supplies
clean terminology the extracted text lost.

## 5. Serving cost (n=80 queries, timed serially)

Every arm pays for the context it sends the answering model; only expansion pays
extra for the rewrite. Tokens counted with `cl100k_base` as a stable offline
yardstick, ratios are the point, absolute figures are approximate.

| technique | R@5 | median | p95 | context tok | rewrite tok | total tok |
|---|---|---|---|---|---|---|
| bm25 | 77.5% | 3.7 ms | 6.5 ms | 2037 | 0 | 2037 |
| vector | 85.0% | 5.3 ms | 6.1 ms | **1972** | 0 | **1972** |
| vector_raw | 83.8% | 5.3 ms | 6.5 ms | 2387 | 0 | 2387 |
| hybrid | 88.8% | 11.3 ms | 17.2 ms | 2000 | 0 | 2000 |
| expansion_bm25 | 93.8% | 4636 ms | 8205 ms | 2114 | 2490 | 4604 |
| expansion_vector | 91.2% | 4735 ms | 5960 ms | 1963 | 2494 | 4457 |
| expansion_raw | 91.2% | 4462 ms | 6927 ms | 2397 | 2493 | 4890 |
| expansion_hybrid | 92.5% | 4567 ms | 6433 ms | 2032 | 2500 | 4532 |

One-time index cost: 7,684 vectors, 71 MB total, ~87 chunks/second to encode on
CPU. The lexical arms need none of it.

> The R@5 column here comes from an 80-query sample and differs by a few points
> from section 4, which uses all 282. Section 4 holds the authoritative accuracy
> figures; this table exists for pricing.

---

## 6. Findings

**Embeddings earn their place.** `vector` beats `bm25` by 8.8 points (84.0% vs
75.2%) for 1.6 ms and no extra tokens. BM25 fails on real student questions
because it can only match words that literally appear in the text. Asked *"why do
we feel tired after running fast"* it returns **Computer Science** chapters, 
"running" and "fast" are the only terms it can match. The book says "anaerobic
respiration" and "lactic acid".

**Fusion is the best model-free option.** `hybrid` reaches 86.9%, ahead of either
of its halves (84.0% and 75.2%). BM25's exact-term matching and dense semantic
matching fail on different questions, so fusing their ranks recovers more than
either alone, for 11.3 ms and no tokens.

**The rewrite is worth ~5 points and costs ~400×.** The best expansion arm
reaches 92.2% against `hybrid`'s 86.9% -- the one gap in this eval that clears
the confidence intervals on both sides. All four expansion arms cluster at
4.5-4.7 s median because one model call sits on the critical path of every query.
Which base the rewrite feeds barely matters (90.1-92.2%, inside noise), the
rewrite itself is doing the work, not what searches afterwards.

**The parser does not improve recall, it improves cost.** `vector` 84.0% vs
`vector_raw` 83.7% is inside noise. But section-aware chunks send **1972 tokens
per query against 2387** for fixed windows: 17% less, on every question,
forever, because a window that stops at a section boundary stops padding once the
topic ends. The parser also yields real citations (`class_11_biology ch5 §5.3 p7`
rather than "somewhere in chapter 5"). On recall alone the parser looks
worthless; on cost and citations it pays.

---

## 7. Recommendation

**Ship `hybrid`.** 86.9% R@5 at 11.3 ms, 2000 tokens, no model dependency at
query time. It is the strongest thing available without putting an LLM call in
front of every question.

**Add `expansion_hybrid` behind a cache** if the extra 5 points matter. Student
questions repeat heavily and a cached rewrite costs nothing on reuse, which would
put 92.2% within reach at near-`hybrid` latency for most traffic. Untested, see
limitation 1.

**Consider routing maths queries through expansion regardless.** On the
fragmented tier it lifts R@5 from 88.4% to 100%, which is a far larger gain than
it delivers anywhere else.

---

## 8. Limitations

The things that would change these conclusions, stated plainly.

1. **The cached-expansion path is unmeasured.** The recommendation above rests on
   an assumption about repeat-question rates this eval did not test.
2. **Chapter-level grading is coarse.** An arm retrieving the right chapter but
   the wrong section scores a hit. Section-level ground truth does not exist for
   exercise questions, and raw windows have no section to compare against.
3. **Answer quality is not measured.** An LLM judge was built and abandoned: a
   `no_context` control answering with *no passages at all* scored 1.50/2, level
   with every real arm (1.45-1.60). These are national-syllabus questions the
   model already knows, so the judge measured its prior knowledge rather than
   retrieval. Any LLM-judge result on this corpus needs that control to be
   interpretable.
4. **Sample size.** Every table now carries a 95% interval on R@5 (±3-5 points
   at n=282, ±10 or more at n=43 on the fragmented tier). Two arms differing by
   less than the sum of their intervals are not distinguishable, however tidy a
   mechanism story the gap suggests. This was learned the hard way: `hybrid`
   was written up as "dominated, worse than its own halves" on one run and "the
   best model-free arm" on the next, from a 4.3-point swing that was 1.4
   standard errors of nothing.
5. **Only the model-free arms are reproducible run to run.** Question wordings
   are cached in `questions.json` and reused unless `--regenerate` is passed, so
   BM25 and the vector arms now return byte-identical numbers on a rerun. The
   expansion arms still drift by 1-2 points because the query rewrite is an
   unseeded LLM call made fresh each run and cached only in memory. Persisting
   that cache to disk would close the gap.
6. **Mathematics is measured, not fixed.** The four maths books extract with
   shredded equations and are tagged `fragmented`. Formula reconstruction was out
   of scope.
7. **Token counts use the wrong tokenizer.** `cl100k_base` is not the serving
   model's tokenizer. Ratios between arms hold; absolute figures are approximate.
8. **One rewrite model, one embedding model.** A separate experiment found
   rewriter choice made no reliable difference at n=60, but that was underpowered.

---
## 9. Three measurement bugs worth recording

All three produced confident, wrong results before being caught.

**Test questions leaked into the corpus.** NCERT prints exercise questions inside
the chapter, so every eval question existed verbatim in the index. BM25 was
scoring by matching each question against its own printed copy. The top hit was
the exercise list itself, ranked above the passage explaining the answer. This
inflated BM25 to a fake 93.6% R@1 and inverted the ranking of every arm. Fixed by
excluding exercise pages from the index; both chunk sets drop identical pages so
neither is advantaged.

**The eval graded arms on questions its own model wrote.** The reworded questions
were originally generated by the same model family the expansion arms use. Each
rewriter then scored best on questions its own family had authored. The gap
between two rewrite models swung 11.6 points purely on authorship. The
paraphraser is now `ag/claude-sonnet-4-6`, a family no arm uses.

**A shared cache made three of four expansion arms look free.** Rewrites are
cached across arms so a recall run pays for them once. In the cost benchmark that
meant every expansion arm after the first measured a cache hit: 5 ms and zero
rewrite tokens, against its true 4.5 s. `evals/cost.py` now clears the cache
between arms so each is measured cold.

A fourth, caught late: the reworder answered "write a program" questions with
code instead of rewording them, putting worked solutions full of textbook
vocabulary into 8% of the query set. `evals/questions.py` now rejects any output
that is not a single question under 300 characters.

## 10. Rejected: rewriting the query as an answer passage

The expansion prompt asks the model for search terms. Having it instead draft
two or three sentences of the textbook passage that would answer the question
retrieves better on every base, measured over 70 questions:

| rewrite style | bm25 | vector | raw |
|---|---|---|---|
| no rewrite | 67.1% | 78.6% | 81.4% |
| search terms (shipped) | 84.3% | 87.1% | 91.4% |
| answer passage | 87.1% | 91.4% | 90.0% |

It is not shipped. Drafting an answer invents textbook prose that was never in
the books, and a study tool should not put fabricated passages in front of
retrieval, whatever they do for recall. Naming the vocabulary translates the
question; writing the answer does not.

A third variant asking for terms and a sentence together scored worst (82.9 /
87.1 / 88.6), because the model kept prefixing its output with commentary
instead of following the format.
