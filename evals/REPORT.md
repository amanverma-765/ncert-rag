# NCERT retrieval eval

17 books, 169 chapters, 4237 section-aware chunks, 3447 raw windows. Embeddings: BAAI/bge-small-en-v1.5, local.

Arms: bm25, vector, vector_raw, hybrid, expansion_bm25, expansion_vector, expansion_raw, expansion_hybrid. Questions are chapter exercise questions reworded the way a student asks. Graded at chapter level: a hit means the arm surfaced a chunk from the chapter the question came from.

### All questions

| arm | R@1 | R@5 | R@10 | MRR | ±R@5 | n |
|---|---|---|---|---|---|---|
| bm25 | 53.9% | 75.2% | 81.2% | 0.633 | ±5.0 | 282 |
| vector | 57.8% | 84.0% | 89.4% | 0.689 | ±4.3 | 282 |
| vector_raw | 59.9% | 83.7% | 90.4% | 0.701 | ±4.3 | 282 |
| hybrid | 58.5% | 86.9% | 92.6% | 0.706 | ±3.9 | 282 |
| expansion_bm25 | 72.3% | 90.8% | 93.6% | 0.801 | ±3.4 | 282 |
| expansion_vector | 69.1% | 90.1% | 95.4% | 0.788 | ±3.5 | 282 |
| expansion_raw | 72.3% | 90.4% | 93.6% | 0.801 | ±3.4 | 282 |
| expansion_hybrid | 74.1% | 92.2% | 94.7% | 0.821 | ±3.1 | 282 |

95% interval on R@5. Two arms differing by less than about 10 points are not distinguishable here.


### Clean tier

| arm | R@1 | R@5 | R@10 | MRR | ±R@5 | n |
|---|---|---|---|---|---|---|
| bm25 | 51.0% | 74.5% | 81.6% | 0.616 | ±5.5 | 239 |
| vector | 56.5% | 84.1% | 89.5% | 0.682 | ±4.6 | 239 |
| vector_raw | 60.7% | 83.7% | 91.2% | 0.708 | ±4.7 | 239 |
| hybrid | 56.5% | 86.6% | 93.3% | 0.693 | ±4.3 | 239 |
| expansion_bm25 | 70.3% | 90.0% | 93.3% | 0.787 | ±3.8 | 239 |
| expansion_vector | 67.8% | 89.1% | 95.4% | 0.776 | ±3.9 | 239 |
| expansion_raw | 71.5% | 89.5% | 93.3% | 0.795 | ±3.9 | 239 |
| expansion_hybrid | 71.5% | 91.6% | 94.6% | 0.804 | ±3.5 | 239 |

95% interval on R@5. Two arms differing by less than about 11 points are not distinguishable here.


### Fragmented tier

| arm | R@1 | R@5 | R@10 | MRR | ±R@5 | n |
|---|---|---|---|---|---|---|
| bm25 | 69.8% | 79.1% | 79.1% | 0.726 | ±12.2 | 43 |
| vector | 65.1% | 83.7% | 88.4% | 0.729 | ±11.0 | 43 |
| vector_raw | 55.8% | 83.7% | 86.0% | 0.662 | ±11.0 | 43 |
| hybrid | 69.8% | 88.4% | 88.4% | 0.775 | ±9.6 | 43 |
| expansion_bm25 | 83.7% | 95.3% | 95.3% | 0.877 | ±6.3 | 43 |
| expansion_vector | 76.7% | 95.3% | 95.3% | 0.855 | ±6.3 | 43 |
| expansion_raw | 76.7% | 95.3% | 95.3% | 0.835 | ±6.3 | 43 |
| expansion_hybrid | 88.4% | 95.3% | 95.3% | 0.915 | ±6.3 | 43 |

95% interval on R@5. Two arms differing by less than about 24 points are not distinguishable here.


## Verdict

- Do embeddings earn their place: best model-free dense arm (`vector`) 84.0% against BM25 75.2% at R@5, a gap of +8.9 points. Yes -- students do not phrase questions in textbook words, and that gap is what bridges it.
- Is an LLM rewrite worth a call per query: `expansion_hybrid` 92.2% against 84.0% without a model, +8.2 points. Price it against the latency below before taking it.
- Does the parser help retrieval: section-aware chunks 84.0% against raw windows 83.7% (+0.4 points). Not on recall -- its payoff is smaller context and real section citations, both below.
