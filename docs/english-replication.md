# English replication: SemEval-2024 Task 8, Subtask C

[README](../README.md) · [Dataset](dataset.md) · [Detectors](detectors.md) · [Likelihood](likelihood.md) · [Counterfactual twins](counterfactual-twins.md) · [English replication](english-replication.md) · [Limitations](limitations.md) · [Reproducing](reproducing.md)

The Sinhala finding is that a boundary detector trained only on documents
that contain a boundary flags machine text in nearly every purely human
document, and that adding human-only documents to training removes much of
that ([Counterfactual twins](counterfactual-twins.md)). This page tests
whether that is specific to Sinhala, on an established English benchmark:
SemEval-2024 Task 8, Subtask C (human-then-machine boundary detection).

Hypotheses, stated before running:

- **H1.** A Subtask C detector trained on the official data flags machine text
  in most human-only documents from the same domains.
- **H2.** Adding human-only documents to training reduces those false alarms
  substantially, at a small cost in the official metric.

**Result in one line:** H1 holds (the DeBERTa-v3 control flags 70–95% of
human documents, and no threshold on the standard grid brings it below 49%). The first half of H2
holds far more strongly than in Sinhala (false alarms fall by 69–76 points, on
every seed and every human test set). The second half does not: the cost is
2.6–3.0 words of MAE (+15–18%), concentrated on two LLaMA generators.

## Data and provenance

### Subtask C

- **The official repository is blocked.** github.com/mbzuai-nlp/SemEval2024-task8
  has returned HTTP 451 since **2026-09-14**, under a DMCA notice from
  wikiHow ([github/dmca 2026-09-14-wikihow](https://github.com/github/dmca/blob/master/2026/09/2026-09-14-wikihow.md))
  that lists the repository's `#data_source` among datasets holding scraped
  wikiHow articles. wikiHow text is in Subtasks A and B; Subtask C is built
  from PeerRead reviews and OUTFOX essays, which the notice does not claim.
- **Used instead:** the M4GT-Bench release by the same group
  (github.com/mbzuai-nlp/M4GT-Bench, commit `e33e38b`, 2026-06-12), whose
  Google Drive folder `1xC-n5hHiXmaCLmA0q3p8Cbyd7Dk0dpDE` holds five files.
  **Only the two Subtask C files were downloaded**; `SubtaskA.jsonl`,
  `SubtaskA_multilingual.jsonl` and `SubtaskB.jsonl` were not.

  | file | Drive id | sha256 |
  |---|---|---|
  | `subtaskC_train_dev.jsonl` | `1n2fgT071bgnHJVJG4syP_ZSI6GaqmlBt` | `857d91a9…eade79` |
  | `subtaskC_test.jsonl` | `1e6a5GgyFEiVMkTUx_K3lN1zGsMhF58uC` | `86e11f1e…a9e0ea` |

- **Checked against the data, this is the official Subtask C.** The
  train/dev file also holds M4GT-Bench's extra generators (four LLaMA
  variants); its `peerread_chatgpt` records are exactly the official 3,649
  train and 505 dev documents, and only those are used. The test file has the
  official 11,123 documents: PeerRead continued by ChatGPT (1,522), LLaMA-2
  7B/13B/70B chat (1,522 each) and base LLaMA (1,035), and OUTFOX essays
  continued by GPT-4 and LLaMA-2 7B/13B/70B chat (1,000 each).
- **The label** is the index of the first machine word, with words defined as
  `text.split(' ')`: `human_end_boundary == len(prefix.split(' '))` holds for
  every official record. Single-space splitting makes a run of spaces yield
  empty "words", and a newline stays inside its word. 232 train and 371 test
  documents are fully machine (boundary 0); 33 test documents have no machine
  word at all. The human prefix is **not** capped at 50% of the document:
  7% of training documents and 29% of test documents are more than half human
  (the cap presumably applies to the human original, and continuations are
  often shorter than the text they replace).
- Licence: the M4GT-Bench repository carries no licence file, and the task's
  own terms were in the now-blocked repository. The data is used for research
  and is not redistributed (`data/` is gitignored).

### Human-only documents

Any human document that shares a lower-cased word 8-gram with **any** Subtask
C text (both files, every split and generator; mixed text, prefix,
continuation and full human original) is removed.

- **PeerRead is exhausted.** The PeerRead repository (commit `9bb3775`) has
  reviews only for ICLR 2017, ACL 2017 and CoNLL 2016, the venues Subtask C
  was built from; its NIPS reviews must be crawled. **36 of its 5,798 reviews
  and comments survive the filter** (16 distinct texts, median 6.5 words).
- **ASAP-Review replaces it** (github.com/neulab/ReviewAdvisor, Apache-2.0;
  `dataset.zip` sha256 `13c5e5ca…33216c`): ICLR 2017–2020 reviews from
  OpenReview and NeurIPS 2016–2019 reviews from the proceedings site, all
  written before ChatGPT. ICLR 2017 is excluded as PeerRead's ICLR year.

  | venue | reviews | after filter | kept |
  |---|---|---|---|
  | ICLR 2018 | 2,748 | 1,946 | 71% |
  | ICLR 2019 | 4,764 | 3,316 | 70% |
  | ICLR 2020 | 6,721 | 4,519 | 67% |
  | NeurIPS 2016 | 3,183 | 2,380 | 75% |
  | NeurIPS 2017 | 1,941 | 1,412 | 73% |
  | NeurIPS 2018 | 3,014 | 2,037 | 68% |
  | NeurIPS 2019 | 4,253 | 3,014 | 71% |
  | **total** | 26,624 | **18,624** | 70% |

  (ICLR 2017: 1,498 reviews, excluded.)
- **The filter removes 30% of ASAP reviews from venues and years Subtask C
  never drew on.** That is boilerplate, not leakage: the most frequent shared
  8-grams are *"the paper is well written and easy to follow"*, *"i would like
  to thank the authors for"* and citation strings such as *"proceedings of the
  ieee conference on computer vision"*; a removed review shares a median of 2
  8-grams with Subtask C, and 73% share 2 or fewer. The filter is kept strict
  anyway. **This biases the test against H1.** The removed reviews are the most
  formulaic, and formulaic text is what a detector tends to call machine, so
  the filter should lower false alarms on the ASAP set. This is an argument,
  not a measurement.
- **OUTFOX** (github.com/ryuryukke/OUTFOX, Apache-2.0, commit `8dd6bfd`):
  6,667 of its 15,400 human student essays survive, all from its train split.
  Every valid and test essay is removed because the Subtask C test uses them.
  OUTFOX is used **for evaluation only**: the official training set has no
  essays, so adding essays to training would mix up the effect of human-only
  documents with in-domain data for the test essays.

### Training conditions

| | added to the 3,649 Subtask C training documents |
|---|---|
| **A** control | nothing |
| **B** + unrelated human reviews | 1,831 ASAP reviews, all words human |
| **C** + human originals ("twins") | the 1,831 full human reviews the training documents were cut from, all words human |

C uses each distinct human original behind the training documents once
(1,835 distinct; the 4 that also appear in test are dropped). B gets the same
number of ASAP reviews, from papers with no review in any evaluation set,
matched one to one to the twins' word counts (1,616 taken whole within ±10%,
215 cut at a sentence boundary). Every human document has weight 1 in both.
So B and C add the same amount of human text:

| | documents | words | median words | p5–p95 words |
|---|---|---|---|---|
| Subtask C train (mixed) | 3,649 | 864,153 | 201 | 64–523 |
| C: human originals | 1,831 | 449,892 | 190 | 26–647 |
| B: ASAP reviews | 1,831 | 448,594 | 191 | 29–638 |

Human-to-mixed ratio: 0.50 by documents, 0.52 by words. The twins are only
approximately content-matched: the human prefix is shared, but the human
continuation differs from the machine one in content and length, so there is
no word-by-word alignment after the prefix and no paired loss is used.

### Evaluation sets

| human test set | docs | median words | length-matched to |
|---|---|---|---|
| ASAP reviews (papers disjoint from training) | 1,000 | 158 | Subtask C PeerRead test (155) |
| OUTFOX essays | 1,000 | 382 | Subtask C OUTFOX test (379) |
| test twins, PeerRead: full human originals of test documents | 747 | 191 | — |
| test twins, OUTFOX | 997 | 363 | — |

Test twins that also appear in train or dev are removed (4). Full counts and
distributions: [`data_report.md`](../results/replication/semeval_c/data_report.md).

## Setup

- **Native formulation.** Word-level token classification: each word is
  labelled 0 (human) or 1 (machine) from its first subword; the predicted
  boundary is the first word predicted machine, or none. Documents longer than
  510 tokens (12% of train, 15% of test, 31% of OUTFOX essays;
  [`token_lengths.md`](../results/replication/semeval_c/token_lengths.md)) are
  packed into chunks on word boundaries; no word is dropped, and predictions
  are reassembled by (document, word index).
- **Words with no token** (empty or whitespace only; 17% of training
  documents contain one) take the label of the preceding word. That rule was
  chosen on train over "following word" (forced MAE 0.027 vs 0.184 with every
  other word right). Its forced error on test is **0.010 words of MAE**
  (107 documents off by at least one word).
- **Model and training.** `microsoft/deberta-v3-base`, 5 epochs, AdamW, lr
  2e-5, weight decay 0.01, linear warm-up over 10% of steps, batch 8 chunks,
  bf16 autocast with fp32 weights, class-weighted cross-entropy (weights from
  the official training words, identical in every condition). The epoch is
  chosen on **dev MAE** only (official dev, 505 documents), which cannot see
  false alarms. Seeds 42, 43, 44. Identical hyperparameters in every
  condition; B and C see more steps per epoch because they have more
  documents.
- **Two decoders.** The task's *first-word* rule at threshold 0.5, which is
  the headline; and a *change-point* decoder, the single boundary (or none)
  maximising the likelihood of a human-then-machine labelling. It has no
  threshold and needs a run of evidence before placing a boundary. It is
  reported so that the false alarms cannot be blamed on a harsh decoding rule.

## Sanity check

Control test MAE is **17.06 ± 0.64** (first-word rule) and **14.61 ± 0.61**
(change point). Published Subtask C systems range from about 21.5 (the
official Longformer baseline) down to about 15.7 for the best submission, so
the control is a reasonable detector. On the training generator
(PeerRead/ChatGPT) its MAE is 3.4; on OUTFOX with LLaMA-2 7B/13B it is 32–33.

## Results

DeBERTa-v3, test set, mean ± standard deviation over three seeds
([`seeds.md`](../results/replication/semeval_c/seeds.md)). First-word rule
at threshold 0.5 unless labelled. False alarm: share of human-only documents
in which any word is predicted machine.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig7-english-dark.png">
  <img alt="English SemEval-2024 Subtask C, three seeds. Human peer reviews with a false alarm: official training data 74%, plus unrelated human reviews 3%, plus human originals 5%. Human student essays: 89%, 14%, 14%. Test MAE: 17.1, 20.1, 19.6." src="assets/fig7-english-light.png">
</picture>

| | A: control | B: + ASAP reviews | C: + human originals |
|---|---|---|---|
| **false alarm, ASAP reviews** | 73.8% ± 4.9 | **2.8% ± 0.5** | 4.6% ± 1.5 |
| **false alarm, OUTFOX essays** | 89.2% ± 4.3 | **13.6% ± 2.1** | 13.9% ± 5.8 |
| false alarm, test twins PeerRead | 77.9% ± 7.8 | 5.7% ± 1.2 | **3.4% ± 1.4** |
| false alarm, test twins OUTFOX | 90.8% ± 3.8 | 16.0% ± 1.3 | **15.4% ± 6.3** |
| words flagged, ASAP / OUTFOX | 21.5% / 23.6% | 1.0% / 2.7% | 1.7% / 2.9% |
| false alarm, change point, ASAP / OUTFOX | 64.7% / 82.6% | 2.3% / 10.8% | 3.5% / 11.1% |
| **test MAE** | **17.06 ± 0.64** | 20.09 ± 2.28 | 19.63 ± 0.39 |
| test MAE, PeerRead / OUTFOX | 12.2 / 25.7 | 19.6 / 21.0 | 18.1 / 22.3 |
| test MAE, change point | **14.61 ± 0.61** | 22.50 ± 3.23 | 22.45 ± 0.28 |
| exact-boundary accuracy | **0.213** | 0.193 | 0.204 |
| machine text detected in mixed test documents | **99%** | 81% | 82% |
| presence accuracy, mixed + human test, balanced | 0.588 | 0.864 | **0.865** |

Seed-paired differences (seeds on which the first condition is better):

| | B − A | C − A | C − B |
|---|---|---|---|
| false alarm, ASAP reviews | **−71.0 ± 4.4 pts** (3/3) | **−69.2 ± 6.4** (3/3) | +1.8 ± 2.0 (1/3) |
| false alarm, OUTFOX essays | **−75.6 ± 3.3** (3/3) | **−75.3 ± 9.8** (3/3) | +0.2 ± 7.8 (1/3) |
| false alarm, test twins PeerRead | −72.2 ± 6.6 (3/3) | −74.4 ± 9.2 (3/3) | −2.3 ± 2.5 (2/3) |
| false alarm, test twins OUTFOX | −74.8 ± 3.3 (3/3) | −75.4 ± 10.0 (3/3) | −0.6 ± 7.2 (1/3) |
| test MAE | +3.02 ± 1.90 (0/3) | +2.56 ± 0.97 (0/3) | −0.46 ± 2.66 (2/3) |
| test MAE, PeerRead | +7.35 ± 3.08 (0/3) | +5.92 ± 1.30 (0/3) | −1.43 ± 3.98 (2/3) |
| test MAE, OUTFOX | −4.69 ± 1.28 (3/3) | −3.42 ± 0.45 (3/3) | +1.28 ± 1.70 (1/3) |
| balanced presence accuracy | +0.276 ± 0.029 (3/3) | +0.277 ± 0.030 (3/3) | +0.001 ± 0.006 (2/3) |

**Where the MAE cost comes from** (mean over seeds; seeds on which B or C is
better than A):

| test slice | docs | A | B | C |
|---|---|---|---|---|
| PeerRead, ChatGPT (the training generator) | 1,522 | 3.4 | 2.7 (3/3) | 2.5 (3/3) |
| OUTFOX, GPT-4 | 1,000 | 10.3 | 8.4 (3/3) | 8.0 (3/3) |
| OUTFOX, LLaMA-2 7B / 13B | 2,000 | 31.5 / 33.4 | 22.2 / 22.4 (3/3) | 23.4 / 23.1 (3/3) |
| OUTFOX, LLaMA-2 70B | 1,000 | 27.6 | 31.0 (0/3) | 34.6 (0/3) |
| PeerRead, LLaMA-2 7B / 13B / 70B | 4,566 | 15.4 / 13.6 / 16.2 | 19.2 / 16.3 / 24.0 | 17.5 / 14.4 / 21.7 |
| **PeerRead, base LLaMA** | 1,035 | **12.6** | **43.4** (0/3) | **42.5** (0/3) |

Human-only training data makes the tagger better on four of the nine test
slices on every seed, including the training generator. It also makes the
tagger miss LLaMA text altogether: B and C predict no machine word at all in
21–37% of LLaMA and LLaMA-2 review continuations and 34–36% of LLaMA-2 13B
essays, against under 2% for the control. ChatGPT and GPT-4 text is still
almost always found. A missed continuation puts the predicted boundary at the
end of the document; on base-LLaMA reviews that costs the most.

### Across thresholds

[`threshold_sweep.md`](../results/replication/semeval_c/threshold_sweep.md)
varies the first-word threshold from 0.05 to 0.95, and further to 0.999.

| | lowest false alarm, thresholds ≤ 0.95 (ASAP / OUTFOX) | at 0.999 (ASAP / OUTFOX; test MAE) | lowest false alarm with test MAE ≤ 20 |
|---|---|---|---|
| A seed 42 | 49% / 66% | 25% / 34%; 22.9 | 35% / 47% |
| A seed 43 | 55% / 73% | 35% / 45%; 19.7 | 35% / 45% |
| A seed 44 | 65% / 82% | 44% / 62%; 17.7 | 44% / 62% |
| B seeds 42/43/44 | 0.7–2.4% / 5.5–11% | ≤ 1% / 1–7% | unreachable / 3% / 3% (ASAP) |
| C seeds 42/43/44 | 1.9–3.9% / 4.0–14% | ≤ 2% / 2–8% | 3–5% / 7–17% |

- **Across the standard grid the control cannot be made quiet**: its lowest
  false-alarm rate is 49–65% on reviews and 66–82% on essays, always at the
  grid's top threshold.
- **Unlike Sinhala, it keeps falling past the grid.** At 0.999 the control
  flags 25–44% of reviews and 34–62% of essays. So "no threshold makes it
  quiet" is true of the standard grid, not absolutely.
- **At equal accuracy the gap remains large.** Among all thresholds with test
  MAE ≤ 20, the control's best is 35–44% of reviews and 45–62% of essays
  flagged; the human-data models reach 3–5% and 7–17%.
- **The MAE cost does not go away by moving the threshold.** The best test MAE
  at any threshold is 16.1 on average for A, 19.0 for B and 18.4 for C.

### XLM-R

One seed (42) per condition with `xlm-roberta-base`, the encoder of the
Sinhala study, same settings
([`xlmr_*_s42.md`](../results/replication/semeval_c/)):

| | A: control | B: + ASAP reviews | C: + human originals |
|---|---|---|---|
| false alarm, ASAP reviews / OUTFOX essays | 76.2% / 54.9% | 6.2% / 10.9% | 10.8% / 12.6% |
| false alarm, test twins PeerRead / OUTFOX | 76.2% / 62.2% | 9.8% / 13.6% | 9.2% / 16.3% |
| false alarm, change point, ASAP / OUTFOX | 68.6% / 51.2% | 5.2% / 8.8% | 8.6% / 9.2% |
| lowest false alarm, thresholds ≤ 0.95 | 57% / 40% | 4% / 8% | 8% / 9% |
| test MAE (first word / change point) | **19.83** / **17.73** | 25.09 / 27.86 | 21.71 / 22.80 |

The same pattern with a weaker encoder: most human documents flagged by the
control (fewer essays than with DeBERTa), a large drop with human-only
training data, and a larger MAE cost (+1.9 to +5.3 words). A single seed.

## Reading

- **H1 holds.** A DeBERTa-v3 tagger trained on the official Subtask C data,
  with a competitive test MAE, flags machine text in 74% of unrelated human
  peer reviews, 89% of human student essays and 78–91% of the human originals
  of the test documents. With the change-point decoder it is still 65% and
  83%. The conclusion of the Sinhala study, that detectors trained only on
  documents with a boundary invent boundaries in human text, is not specific
  to Sinhala, to sentence-level tagging or to one decoder.
- **H2 holds for false alarms, much more strongly than in Sinhala.** Adding
  about half as many human-only documents as mixed ones cuts false alarms by
  69–76 points on every seed and every human test set, including essays, a
  genre absent from training. In Sinhala the same step removed about a third
  of false alarms (93% → 60%).
- **H2 fails on "small cost".** Test MAE rises by 2.6–3.0 words (15–18%), on
  every seed, and moving the threshold does not recover it. The cost is not
  uniform: human-only data improves four of nine test slices on every seed
  (the training generator among them) and loses heavily on base-LLaMA review
  continuations. The human-data models leave a fifth to a third of LLaMA
  continuations entirely unflagged. That is the same trade-off
  as in Sinhala (a small loss on mixed documents for a large gain on human
  ones), but larger.
- **Content matching makes no measurable difference here.** B (unrelated
  reviews) and C (the documents' own human originals) are level on every
  false-alarm set and on MAE (C − B: −0.5 ± 2.7 MAE, 2/3 seeds). The one
  consistent edge of C, fewer false alarms on the PeerRead test twins (3.4%
  vs 5.7%, 2/3 seeds), is small. This matches the Sinhala result with
  threshold decoding, where unrelated human text gave most of the gain.

## Limitations

- **Formulation.** English uses the native word-level task with one boundary
  per document, human then machine; the Sinhala benchmark tags sentences and
  has one or more machine spans anywhere in the document. The two numbers are
  comparable in direction, not in size.
- **Data substitutions.** Subtask C came from M4GT-Bench because the task
  repository is DMCA-blocked. The review-side human documents are ASAP-Review
  (ICLR 2018–2020, NeurIPS 2016–2019), not PeerRead, because PeerRead is
  entirely used by Subtask C. They are the same genre but later venues and
  years than the Subtask C reviews.
- **The 8-gram filter** removes 30% of ASAP reviews for boilerplate overlap,
  which should make false alarms on ASAP lower than on unfiltered reviews.
- **Twins are approximate.** Condition C's human originals share only the
  prefix with their training documents; there is no word-level alignment and
  no paired loss, unlike the Sinhala twins.
- **Selection sees no human text.** Epochs are chosen on mixed dev documents
  and the threshold is fixed at 0.5, as in the task; a selection rule that
  saw human documents would pick different operating points.
- **Domains.** Two domains (peer reviews, student essays); OUTFOX is
  evaluation only.

## Files

`results/replication/semeval_c/`: one `.json` and `.md` per run
(`deberta_{control,human,twins}_s{42,43,44}`, `xlmr_*_s42`), `seeds.*`
(means and paired differences), `threshold_sweep.*`, `data_report.*`,
`token_lengths.*`. Code: `src/replication/semeval_c/`. Commands and runtimes:
[Reproducing](reproducing.md#6-english-replication).
