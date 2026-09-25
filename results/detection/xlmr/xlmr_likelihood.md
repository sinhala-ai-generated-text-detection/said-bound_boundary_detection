# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 8 epochs, lr 3e-05, effective batch 4, best epoch 4, 44.0 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.675 (sentence F1) and boundary bias +1.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.832 | 0.785 | 0.747 | 0.827 | 0.607 | 0.751 | 0.302 |
| seen generators | 0.847 | 0.808 | 0.761 | 0.862 | 0.630 | 0.764 | 0.362 |
| held-out (Gemini) | 0.802 | 0.737 | 0.718 | 0.757 | 0.564 | 0.728 | 0.180 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.840 | 0.785 | 0.786 | 0.784 | 0.587 | 0.749 | 0.350 |
| **viterbi (pair head)** | 0.832 | 0.785 | 0.747 | 0.827 | 0.607 | 0.751 | 0.302 |

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
| **xlm-roberta-base** | **0.785** | **0.607** | **0.737** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.798 | 0.605 |
| gemini_2_5_pro | held_out | 305 | 0.725 | 0.535 |
| gpt_4o | seen | 311 | 0.827 | 0.625 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.917 | 0.617 |
| type2_single_internal_segment | 299 | 0.666 | 0.559 |
| type3_multiple_internal_segments | 251 | 0.626 | 0.593 |
