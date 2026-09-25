# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 1, 29.7 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.475 (sentence F1) and boundary bias +2.75 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.837 | 0.789 | 0.760 | 0.821 | 0.611 | 0.759 | 0.321 |
| seen generators | 0.860 | 0.822 | 0.787 | 0.861 | 0.648 | 0.794 | 0.393 |
| held-out (Gemini) | 0.790 | 0.721 | 0.703 | 0.740 | 0.545 | 0.697 | 0.177 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.846 | 0.799 | 0.777 | 0.821 | 0.601 | 0.772 | 0.350 |
| **viterbi (pair head)** | 0.837 | 0.789 | 0.760 | 0.821 | 0.611 | 0.759 | 0.321 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.955 | 3.18 | 0.272 | 0.860 | 0.532 | 0.877 | 0.429 |
| threshold | seen | 377 | 0.955 | 3.13 | 0.269 | 0.888 | 0.569 | 0.875 | 0.451 |
| threshold | held_out | 289 | 0.955 | 3.23 | 0.276 | 0.805 | 0.458 | 0.880 | 0.348 |
| viterbi | overall | 578 | 0.991 | 4.36 | 0.317 | 0.860 | 0.493 | 0.840 | 0.409 |
| viterbi | seen | 377 | 0.989 | 4.39 | 0.316 | 0.888 | 0.531 | 0.834 | 0.430 |
| viterbi | held_out | 289 | 0.997 | 4.32 | 0.320 | 0.805 | 0.416 | 0.853 | 0.323 |

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
| **xlm-roberta-base** | **0.789** | **0.611** | **0.721** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.815 | 0.639 |
| gemini_2_5_pro | held_out | 305 | 0.739 | 0.550 |
| gpt_4o | seen | 311 | 0.838 | 0.618 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.921 | 0.632 |
| type2_single_internal_segment | 299 | 0.688 | 0.533 |
| type3_multiple_internal_segments | 251 | 0.658 | 0.632 |
