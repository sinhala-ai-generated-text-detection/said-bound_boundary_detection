# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 1, 17.5 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.800 (sentence F1) and boundary bias +2.25 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.829 | 0.789 | 0.730 | 0.858 | 0.593 | 0.748 | 0.292 |
| seen generators | 0.849 | 0.817 | 0.751 | 0.895 | 0.621 | 0.767 | 0.356 |
| held-out (Gemini) | 0.789 | 0.731 | 0.687 | 0.782 | 0.543 | 0.713 | 0.164 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.841 | 0.798 | 0.758 | 0.843 | 0.588 | 0.767 | 0.340 |
| **viterbi (pair head)** | 0.829 | 0.789 | 0.730 | 0.858 | 0.593 | 0.748 | 0.292 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.967 | 3.16 | 0.275 | 0.875 | 0.554 | 0.885 | 0.421 |
| threshold | seen | 377 | 0.966 | 3.08 | 0.275 | 0.903 | 0.590 | 0.877 | 0.444 |
| threshold | held_out | 289 | 0.972 | 3.25 | 0.276 | 0.819 | 0.481 | 0.899 | 0.337 |
| viterbi | overall | 578 | 0.993 | 4.30 | 0.331 | 0.875 | 0.513 | 0.859 | 0.400 |
| viterbi | seen | 377 | 0.989 | 4.31 | 0.334 | 0.903 | 0.541 | 0.851 | 0.417 |
| viterbi | held_out | 289 | 1.000 | 4.31 | 0.331 | 0.819 | 0.454 | 0.875 | 0.323 |

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
| **xlm-roberta-base** | **0.789** | **0.593** | **0.731** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.814 | 0.620 |
| gemini_2_5_pro | held_out | 305 | 0.736 | 0.533 |
| gpt_4o | seen | 311 | 0.840 | 0.616 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.918 | 0.602 |
| type2_single_internal_segment | 299 | 0.687 | 0.524 |
| type3_multiple_internal_segments | 251 | 0.664 | 0.623 |
