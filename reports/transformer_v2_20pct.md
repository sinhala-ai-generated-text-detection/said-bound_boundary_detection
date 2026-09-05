# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 12 epochs, lr 3e-05, effective batch 4, best epoch 9, 36.2 min on GPU. Trained on 502 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.800 (sentence F1) and boundary bias +1.75 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.759 | 0.707 | 0.624 | 0.816 | 0.497 | 0.699 | 0.139 |
| seen generators | 0.778 | 0.737 | 0.638 | 0.870 | 0.510 | 0.735 | 0.190 |
| held-out (Gemini) | 0.733 | 0.664 | 0.602 | 0.741 | 0.482 | 0.654 | 0.067 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.770 | 0.708 | 0.646 | 0.784 | 0.487 | 0.703 | 0.161 |
| **viterbi (pair head)** | 0.759 | 0.707 | 0.624 | 0.816 | 0.497 | 0.699 | 0.139 |

## Versus the linear baselines

| model | sent F1(AI) | bound F1 exact | held-out F1(AI) |
|---|---|---|---|
| baseline: all-human | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.526 | 0.000 | 0.525 |
| baseline: random (train prior) | 0.371 | 0.360 | 0.349 |
| baseline: position only (no text) | 0.518 | 0.421 | 0.487 |
| text only | 0.574 | 0.377 | 0.490 |
| text + context | 0.586 | 0.295 | 0.508 |
| text + context + smoothing | 0.561 | 0.201 | 0.464 |
| text + position (diagnostic) | 0.580 | 0.366 | 0.495 |
| **xlm-roberta-base** | **0.707** | **0.497** | **0.664** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 52 | 0.755 | 0.531 |
| gemini_2_5_pro | held_out | 75 | 0.668 | 0.450 |
| gpt_4o | seen | 53 | 0.715 | 0.500 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 59 | 0.894 | 0.462 |
| type2_single_internal_segment | 73 | 0.553 | 0.420 |
| type3_multiple_internal_segments | 48 | 0.594 | 0.554 |
