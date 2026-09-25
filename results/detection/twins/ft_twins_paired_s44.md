# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 1, 11.9 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.200 (sentence F1) and boundary bias +1.75 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.841 | 0.785 | 0.792 | 0.777 | 0.605 | 0.743 | 0.306 |
| seen generators | 0.858 | 0.814 | 0.801 | 0.827 | 0.635 | 0.761 | 0.370 |
| held-out (Gemini) | 0.807 | 0.720 | 0.771 | 0.675 | 0.548 | 0.707 | 0.177 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.849 | 0.795 | 0.805 | 0.786 | 0.576 | 0.740 | 0.322 |
| **viterbi (pair head)** | 0.841 | 0.785 | 0.792 | 0.777 | 0.605 | 0.743 | 0.306 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 1.0, consistency weight 1.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.616 | 1.63 | 0.131 | 0.928 | 0.644 | 0.951 | 0.472 |
| threshold | seen | 377 | 0.610 | 1.61 | 0.131 | 0.947 | 0.685 | 0.948 | 0.495 |
| threshold | held_out | 289 | 0.612 | 1.55 | 0.127 | 0.891 | 0.559 | 0.956 | 0.403 |
| viterbi | overall | 578 | 0.791 | 2.68 | 0.166 | 0.928 | 0.606 | 0.941 | 0.462 |
| viterbi | seen | 377 | 0.796 | 2.72 | 0.167 | 0.947 | 0.647 | 0.937 | 0.484 |
| viterbi | held_out | 289 | 0.779 | 2.55 | 0.160 | 0.891 | 0.522 | 0.948 | 0.383 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.602 | 1.50 | 0.121 |
| viterbi | 618 | 0.735 | 2.26 | 0.146 |

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
| **xlm-roberta-base** | **0.785** | **0.605** | **0.720** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.805 | 0.584 |
| gemini_2_5_pro | held_out | 305 | 0.734 | 0.527 |
| gpt_4o | seen | 311 | 0.841 | 0.618 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.913 | 0.592 |
| type2_single_internal_segment | 299 | 0.681 | 0.534 |
| type3_multiple_internal_segments | 251 | 0.643 | 0.596 |
