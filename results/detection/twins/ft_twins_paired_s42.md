# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 2, 11.9 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.200 (sentence F1) and boundary bias +2.75 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.843 | 0.789 | 0.790 | 0.789 | 0.603 | 0.755 | 0.294 |
| seen generators | 0.862 | 0.820 | 0.803 | 0.838 | 0.635 | 0.777 | 0.365 |
| held-out (Gemini) | 0.806 | 0.723 | 0.759 | 0.689 | 0.540 | 0.713 | 0.151 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.849 | 0.794 | 0.808 | 0.780 | 0.574 | 0.742 | 0.329 |
| **viterbi (pair head)** | 0.843 | 0.789 | 0.790 | 0.789 | 0.603 | 0.755 | 0.294 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 1.0, consistency weight 1.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.555 | 1.55 | 0.126 | 0.921 | 0.650 | 0.944 | 0.474 |
| threshold | seen | 377 | 0.565 | 1.56 | 0.127 | 0.943 | 0.697 | 0.939 | 0.504 |
| threshold | held_out | 289 | 0.533 | 1.45 | 0.119 | 0.878 | 0.554 | 0.953 | 0.391 |
| viterbi | overall | 578 | 0.758 | 2.60 | 0.171 | 0.921 | 0.612 | 0.936 | 0.461 |
| viterbi | seen | 377 | 0.782 | 2.69 | 0.175 | 0.943 | 0.653 | 0.930 | 0.482 |
| viterbi | held_out | 289 | 0.720 | 2.40 | 0.161 | 0.878 | 0.527 | 0.948 | 0.383 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.555 | 1.41 | 0.113 |
| viterbi | 618 | 0.744 | 2.31 | 0.155 |

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
| **xlm-roberta-base** | **0.789** | **0.603** | **0.723** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.811 | 0.595 |
| gemini_2_5_pro | held_out | 305 | 0.720 | 0.506 |
| gpt_4o | seen | 311 | 0.844 | 0.623 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.913 | 0.607 |
| type2_single_internal_segment | 299 | 0.685 | 0.512 |
| type3_multiple_internal_segments | 251 | 0.635 | 0.598 |
