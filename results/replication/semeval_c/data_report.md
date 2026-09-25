# English replication: data

Counts and word-length statistics (words = `text.split(' ')`, the official Subtask C convention) for every set built by `src/replication/semeval_c/corpus.py`.

## Subtask C (official)

Train 3,649, dev 505, test 11,123. Fully machine documents (boundary 0): train 232, dev 23, test 371. Boundary at or past the last word (no machine word): train 0, dev 0, test 33.

| test domain / generator | docs |
|---|---|
| outfox/gpt4 | 1,000 |
| outfox/llama13b_chat | 1,000 |
| outfox/llama70b_chat | 1,000 |
| outfox/llama7b_chat | 1,000 |
| peerread/chatgpt | 1,522 |
| peerread/llama | 1,035 |
| peerread/llama2_13B_chat | 1,522 |
| peerread/llama2_70B_chat | 1,522 |
| peerread/llama2_7B_chat | 1,522 |

## Words with no token

Words that are empty or whitespace only get no subword token, so their label is inherited from a neighbour. The rule was chosen on train by the error it forces with every other word right:

| rule | train MAE | train docs with error |
|---|---|---|
| prev (chosen) | 0.027 | 98 |
| next | 0.184 | 26 |

Forced error of the chosen rule:

| split | MAE | docs with error | docs | docs with tokenless words |
|---|---|---|---|---|
| train | 0.027 | 98 | 3,649 | 628 |
| dev | 0.036 | 18 | 505 | 70 |
| test | 0.010 | 107 | 11,123 | 1,775 |

## Human-only sources and the 8-gram filter

Any document sharing a lower-cased word 8-gram with any Subtask C text (every file, split and generator; mixed, prefix, continuation and full human original) is removed.

- **PeerRead** (ICLR 2017, ACL 2017, CoNLL 2016): 5,798 reviews and comments, **36 survive** (16 distinct, median 6.5 words). Not usable.
- **OUTFOX** human essays: 15,400, 6,667 survive, all from OUTFOX's train split (none of its valid or test essays: the Subtask C test uses them).

| ASAP-Review venue | reviews | after filter | kept |
|---|---|---|---|
| ICLR_2017 | 1,498 | 133 | excluded (PeerRead's ICLR year) |
| ICLR_2018 | 2,748 | 1,946 | 71% |
| ICLR_2019 | 4,764 | 3,316 | 70% |
| ICLR_2020 | 6,721 | 4,519 | 67% |
| NIPS_2016 | 3,183 | 2,380 | 75% |
| NIPS_2017 | 1,941 | 1,412 | 73% |
| NIPS_2018 | 3,014 | 2,037 | 68% |
| NIPS_2019 | 4,253 | 3,014 | 71% |

Outside ICLR 2017 the filter removes 30% of reviews (8,000 of 26,624), from venues and years Subtask C never drew on. These are generic reviewing phrases, not leakage. The filter is kept strict anyway.

## Human documents added in training

Condition C adds each distinct full human original behind the 3,649 training documents once: 1,835 distinct, minus 4 that also appear in test = **1,831**. Condition B adds the same number of ASAP reviews from train-side papers, matched one to one to the twins' word counts (1616 whole, 215 cut at a sentence boundary). Every human document has weight 1 in both. Human-to-mixed ratio: 0.50 by documents, 0.52 by words.

| set | docs | mean | p5 | p25 | p50 | p75 | p95 | total words |
|---|---|---|---|---|---|---|---|---|
| Subtask C train (mixed) | 3,649 | 237 | 64 | 125 | 201 | 314 | 523 | 864,153 |
| C: twins | 1,831 | 246 | 26 | 96 | 190 | 327 | 647 | 449,892 |
| B: ASAP | 1,831 | 245 | 29 | 96 | 191 | 329 | 638 | 448,594 |

## Human evaluation sets

ASAP evaluation reviews come from papers with no review in training. ASAP and OUTFOX sets are length-matched to the Subtask C test documents of their domain. Test twins are the full human originals of the test documents, minus any that appear in train or dev.

| set | docs | mean | p5 | p25 | p50 | p75 | p95 | total words |
|---|---|---|---|---|---|---|---|---|
| Subtask C test, PeerRead (target) | 7,123 | 183 | 49 | 105 | 155 | 232 | 407 | 1,306,289 |
| ASAP eval | 1,000 | 186 | 52 | 107 | 158 | 237 | 422 | 185,632 |
| Subtask C test, OUTFOX (target) | 4,000 | 383 | 113 | 270 | 379 | 476 | 631 | 1,532,276 |
| OUTFOX eval | 1,000 | 382 | 122 | 271 | 382 | 475 | 668 | 382,001 |
| test twins, PeerRead | 747 | 236 | 25 | 88 | 191 | 324 | 572 | 176,332 |
| test twins, OUTFOX | 997 | 399 | 171 | 256 | 363 | 493 | 774 | 398,158 |

Matching (whole / cut at a sentence boundary / shorter than the target because no unused document was long enough): asap 1000 / 0 / 0; outfox 947 / 45 / 8.
