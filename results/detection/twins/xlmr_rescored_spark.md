# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 12 epochs, lr 3e-05, effective batch 4, best epoch -1, 0.0 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.800 (sentence F1) and boundary bias +3.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.846 | 0.802 | 0.770 | 0.837 | 0.603 | 0.772 | 0.358 |
| seen generators | 0.865 | 0.830 | 0.787 | 0.878 | 0.629 | 0.793 | 0.418 |
| held-out (Gemini) | 0.809 | 0.743 | 0.733 | 0.753 | 0.555 | 0.732 | 0.239 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.846 | 0.802 | 0.770 | 0.837 | 0.603 | 0.772 | 0.358 |
| **viterbi (pair head)** | 0.833 | 0.792 | 0.739 | 0.853 | 0.604 | 0.751 | 0.305 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.953 | 3.08 | 0.262 | 0.878 | 0.558 | 0.884 | 0.433 |
| threshold | seen | 377 | 0.944 | 3.01 | 0.260 | 0.907 | 0.598 | 0.883 | 0.454 |
| threshold | held_out | 289 | 0.958 | 3.15 | 0.265 | 0.819 | 0.476 | 0.886 | 0.352 |
| viterbi | overall | 578 | 0.995 | 4.54 | 0.333 | 0.878 | 0.505 | 0.844 | 0.401 |
| viterbi | seen | 377 | 0.992 | 4.62 | 0.336 | 0.907 | 0.538 | 0.835 | 0.417 |
| viterbi | held_out | 289 | 1.000 | 4.45 | 0.329 | 0.819 | 0.440 | 0.863 | 0.324 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.935 | 2.82 | 0.241 |
| viterbi | 618 | 0.992 | 4.31 | 0.317 |

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
| **xlm-roberta-base** | **0.802** | **0.603** | **0.743** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.816 | 0.625 |
| gemini_2_5_pro | held_out | 305 | 0.743 | 0.555 |
| gpt_4o | seen | 311 | 0.844 | 0.632 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.925 | 0.635 |
| type2_single_internal_segment | 299 | 0.698 | 0.546 |
| type3_multiple_internal_segments | 251 | 0.654 | 0.628 |
