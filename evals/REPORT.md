# NCERT retrieval eval

17 books, 169 chapters, 4429 section-aware chunks, 3555 raw windows. Embeddings: BAAI/bge-small-en-v1.5, local.

Arms: bm25, vector, vector_raw, hybrid, expansion_bm25, expansion_vector, expansion_raw, expansion_hybrid. Questions are chapter exercise questions reworded the way a student asks. Graded at chapter level: a hit means the arm surfaced a chunk from the chapter the question came from.

### All questions

| arm | R@1 | R@5 | R@10 | MRR | n |
|---|---|---|---|---|---|
| bm25 | 53.9% | 75.5% | 81.9% | 0.635 | 282 |
| vector | 59.6% | 83.3% | 90.8% | 0.698 | 282 |
| vector_raw | 58.9% | 84.4% | 89.7% | 0.693 | 282 |
| hybrid | 58.5% | 86.9% | 92.9% | 0.704 | 282 |
| expansion_bm25 | 71.6% | 90.8% | 94.3% | 0.802 | 282 |
| expansion_vector | 71.3% | 90.8% | 94.3% | 0.800 | 282 |
| expansion_raw | 69.9% | 92.2% | 95.0% | 0.789 | 282 |
| expansion_hybrid | 72.7% | 92.9% | 95.0% | 0.815 | 282 |

Differences between rows are not tested here. Every arm answers the same questions, so comparing two of them calls for a paired test; those are under Paired comparisons, over the whole question set only. Nothing in this table is tested per tier.


### Clean tier

| arm | R@1 | R@5 | R@10 | MRR | n |
|---|---|---|---|---|---|
| bm25 | 51.0% | 74.9% | 82.4% | 0.617 | 239 |
| vector | 59.8% | 84.5% | 91.2% | 0.705 | 239 |
| vector_raw | 59.0% | 84.5% | 90.4% | 0.698 | 239 |
| hybrid | 56.5% | 87.0% | 94.1% | 0.692 | 239 |
| expansion_bm25 | 70.7% | 90.8% | 93.3% | 0.796 | 239 |
| expansion_vector | 69.9% | 90.4% | 94.6% | 0.791 | 239 |
| expansion_raw | 68.2% | 91.6% | 95.0% | 0.778 | 239 |
| expansion_hybrid | 71.1% | 92.9% | 95.4% | 0.806 | 239 |

Differences between rows are not tested here. Every arm answers the same questions, so comparing two of them calls for a paired test; those are under Paired comparisons, over the whole question set only. Nothing in this table is tested per tier.


### Fragmented tier

| arm | R@1 | R@5 | R@10 | MRR | n |
|---|---|---|---|---|---|
| bm25 | 69.8% | 79.1% | 79.1% | 0.738 | 43 |
| vector | 58.1% | 76.7% | 88.4% | 0.659 | 43 |
| vector_raw | 58.1% | 83.7% | 86.0% | 0.667 | 43 |
| hybrid | 69.8% | 86.0% | 86.0% | 0.771 | 43 |
| expansion_bm25 | 76.7% | 90.7% | 100.0% | 0.834 | 43 |
| expansion_vector | 79.1% | 93.0% | 93.0% | 0.853 | 43 |
| expansion_raw | 79.1% | 95.3% | 95.3% | 0.853 | 43 |
| expansion_hybrid | 81.4% | 93.0% | 93.0% | 0.862 | 43 |

Differences between rows are not tested here. Every arm answers the same questions, so comparing two of them calls for a paired test; those are under Paired comparisons, over the whole question set only. Nothing in this table is tested per tier.


## Paired comparisons

Every arm answers the same questions, so each pair is tested with McNemar over the questions the two disagree about -- the only ones carrying evidence. `disagree` is those counts, b's wins first.

| comparison | R@5 | gap | disagree | p | 95% CI |
|---|---|---|---|---|---|
| `vector` over `bm25` | 75.5% -> 83.3% | +7.8 | 42/20 | 0.0077 | [+2.4, +13.2] |
| `hybrid` over `bm25` | 75.5% -> 86.9% | +11.3 | 35/3 | <0.0001 | [+7.3, +15.4] |
| `vector` over `vector_raw` | 84.4% -> 83.3% | -1.1 | 12/15 | 0.7003 | [-4.7, +2.5] |
| `hybrid` over `vector` | 83.3% -> 86.9% | +3.5 | 22/12 | 0.1227 | [-0.5, +7.6] |
| `expansion_hybrid` over `hybrid` | 86.9% -> 92.9% | +6.0 | 22/5 | 0.0021 | [+2.5, +9.6] |

Five pre-specified tests sharing no multiple-comparison correction, so a Bonferroni floor would be p<0.01 here. A high p is not evidence that two arms are equal: read the interval, which is what it can still hide.


## Serving cost

Accuracy and latency from the same 80 queries, run one at a time on CPU so the timings are what one student waits through rather than pool throughput.

| arm | R@5 | median |
|---|---|---|
| bm25 | 80.0% | 3.5 ms |
| vector | 85.0% | 4.9 ms |
| vector_raw | 83.8% | 4.8 ms |
| hybrid | 88.8% | 8.9 ms |
| expansion_bm25 | 90.0% | 4563.4 ms |
| expansion_vector | 93.8% | 4625.3 ms |
| expansion_raw | 93.8% | 4585.2 ms |
| expansion_hybrid | 96.2% | 4708.8 ms |

Latency is all this measures. What an arm costs in tokens -- the context it hands the answering model, and the rewrite on top for the expansion arms -- is no longer measured at all, so the price of a query cannot be read off this table. See EVALUATION.md §6.
