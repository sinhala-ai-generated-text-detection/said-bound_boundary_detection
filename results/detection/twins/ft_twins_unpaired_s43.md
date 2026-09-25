# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 1, 11.8 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.475 (sentence F1) and boundary bias +2.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.844 | 0.796 | 0.774 | 0.819 | 0.612 | 0.759 | 0.322 |
| seen generators | 0.864 | 0.826 | 0.794 | 0.861 | 0.643 | 0.787 | 0.377 |
| held-out (Gemini) | 0.802 | 0.731 | 0.729 | 0.733 | 0.554 | 0.707 | 0.213 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.850 | 0.800 | 0.797 | 0.803 | 0.582 | 0.750 | 0.346 |
| **viterbi (pair head)** | 0.844 | 0.796 | 0.774 | 0.819 | 0.612 | 0.759 | 0.322 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 0.0, consistency weight 0.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.659 | 1.77 | 0.144 | 0.921 | 0.653 | 0.939 | 0.470 |
| threshold | seen | 377 | 0.679 | 1.80 | 0.143 | 0.943 | 0.697 | 0.936 | 0.489 |
| threshold | held_out | 289 | 0.640 | 1.66 | 0.143 | 0.876 | 0.562 | 0.945 | 0.400 |
| viterbi | overall | 578 | 0.862 | 3.13 | 0.204 | 0.921 | 0.614 | 0.918 | 0.449 |
| viterbi | seen | 377 | 0.862 | 3.17 | 0.204 | 0.943 | 0.651 | 0.914 | 0.470 |
| viterbi | held_out | 289 | 0.862 | 3.03 | 0.203 | 0.876 | 0.538 | 0.926 | 0.369 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.625 | 1.61 | 0.131 |
| viterbi | 618 | 0.816 | 2.75 | 0.186 |

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
| **xlm-roberta-base** | **0.796** | **0.612** | **0.731** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.818 | 0.604 |
| gemini_2_5_pro | held_out | 305 | 0.737 | 0.532 |
| gpt_4o | seen | 311 | 0.838 | 0.611 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.924 | 0.639 |
| type2_single_internal_segment | 299 | 0.682 | 0.523 |
| type3_multiple_internal_segments | 251 | 0.644 | 0.596 |
