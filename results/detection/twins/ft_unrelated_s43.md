# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 1, 12.0 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.225 (sentence F1) and boundary bias +2.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.838 | 0.781 | 0.789 | 0.772 | 0.616 | 0.751 | 0.313 |
| seen generators | 0.862 | 0.818 | 0.810 | 0.826 | 0.654 | 0.781 | 0.385 |
| held-out (Gemini) | 0.792 | 0.700 | 0.740 | 0.664 | 0.546 | 0.697 | 0.167 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.847 | 0.790 | 0.805 | 0.776 | 0.564 | 0.746 | 0.341 |
| **viterbi (pair head)** | 0.838 | 0.781 | 0.789 | 0.772 | 0.616 | 0.751 | 0.313 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with unrelated same-length human windows: margin weight 0.0, consistency weight 0.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.685 | 1.79 | 0.142 | 0.906 | 0.629 | 0.936 | 0.453 |
| threshold | seen | 377 | 0.698 | 1.82 | 0.145 | 0.934 | 0.674 | 0.933 | 0.473 |
| threshold | held_out | 289 | 0.651 | 1.66 | 0.135 | 0.850 | 0.536 | 0.943 | 0.385 |
| viterbi | overall | 578 | 0.908 | 3.56 | 0.207 | 0.906 | 0.570 | 0.900 | 0.441 |
| viterbi | seen | 377 | 0.899 | 3.55 | 0.207 | 0.934 | 0.620 | 0.895 | 0.467 |
| viterbi | held_out | 289 | 0.927 | 3.58 | 0.209 | 0.850 | 0.469 | 0.911 | 0.349 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.639 | 1.57 | 0.128 |
| viterbi | 618 | 0.879 | 3.12 | 0.191 |

## Versus the linear baselines

| model | sent F1(AI) | bound F1 exact | held-out F1(AI) |
|---|---|---|---|
| baseline: all-human | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.542 | 0.000 | 0.536 |
| baseline: random (train prior) | 0.367 | 0.357 | 0.341 |
| baseline: position only (no text) | 0.567 | 0.284 | 0.550 |
| text only | 0.675 | 0.482 | 0.575 |
| text + context | 0.678 | 0.438 | 0.571 |
| text + context + smoothing | 0.651 | 0.314 | 0.538 |
| text + position (diagnostic) | 0.696 | 0.474 | 0.606 |
| likelihood only (no text) | 0.606 | 0.323 | 0.588 |
| text + context + likelihood | 0.704 | 0.450 | 0.615 |
| **xlm-roberta-base** | **0.781** | **0.616** | **0.700** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.807 | 0.588 |
| gemini_2_5_pro | held_out | 305 | 0.722 | 0.514 |
| gpt_4o | seen | 311 | 0.834 | 0.591 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.917 | 0.625 |
| type2_single_internal_segment | 299 | 0.677 | 0.511 |
| type3_multiple_internal_segments | 251 | 0.619 | 0.574 |
