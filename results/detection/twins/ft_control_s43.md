# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 3, 6.5 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.650 (sentence F1) and boundary bias +4.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.828 | 0.782 | 0.739 | 0.830 | 0.622 | 0.740 | 0.307 |
| seen generators | 0.849 | 0.813 | 0.762 | 0.871 | 0.653 | 0.765 | 0.367 |
| held-out (Gemini) | 0.785 | 0.718 | 0.691 | 0.747 | 0.564 | 0.694 | 0.187 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.846 | 0.802 | 0.768 | 0.840 | 0.603 | 0.777 | 0.362 |
| **viterbi (pair head)** | 0.828 | 0.782 | 0.739 | 0.830 | 0.622 | 0.740 | 0.307 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.960 | 3.09 | 0.271 | 0.868 | 0.554 | 0.880 | 0.432 |
| threshold | seen | 377 | 0.958 | 3.05 | 0.268 | 0.892 | 0.591 | 0.877 | 0.449 |
| threshold | held_out | 289 | 0.965 | 3.17 | 0.276 | 0.819 | 0.480 | 0.887 | 0.356 |
| viterbi | overall | 578 | 0.998 | 4.98 | 0.335 | 0.868 | 0.492 | 0.831 | 0.408 |
| viterbi | seen | 377 | 0.997 | 5.01 | 0.333 | 0.892 | 0.523 | 0.822 | 0.427 |
| viterbi | held_out | 289 | 1.000 | 4.90 | 0.339 | 0.819 | 0.427 | 0.849 | 0.327 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.942 | 2.88 | 0.254 |
| viterbi | 618 | 0.994 | 4.69 | 0.324 |

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
| **xlm-roberta-base** | **0.782** | **0.622** | **0.718** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.819 | 0.625 |
| gemini_2_5_pro | held_out | 305 | 0.744 | 0.563 |
| gpt_4o | seen | 311 | 0.841 | 0.623 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.924 | 0.630 |
| type2_single_internal_segment | 299 | 0.690 | 0.542 |
| type3_multiple_internal_segments | 251 | 0.666 | 0.631 |
