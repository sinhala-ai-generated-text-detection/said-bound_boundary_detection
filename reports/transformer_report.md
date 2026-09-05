# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 8 epochs, lr 3e-05, effective batch 4, best epoch 8, 109.4 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.800 (sentence F1) and boundary bias +3.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.833 | 0.792 | 0.739 | 0.853 | 0.603 | 0.751 | 0.305 |
| seen generators | 0.856 | 0.823 | 0.762 | 0.894 | 0.634 | 0.779 | 0.370 |
| held-out (Gemini) | 0.788 | 0.726 | 0.689 | 0.768 | 0.548 | 0.700 | 0.174 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.846 | 0.802 | 0.770 | 0.837 | 0.603 | 0.772 | 0.358 |
| **viterbi (pair head)** | 0.833 | 0.792 | 0.739 | 0.853 | 0.603 | 0.751 | 0.305 |

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
| **xlm-roberta-base** | **0.792** | **0.603** | **0.726** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.816 | 0.625 |
| gemini_2_5_pro | held_out | 305 | 0.742 | 0.556 |
| gpt_4o | seen | 311 | 0.843 | 0.631 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.925 | 0.635 |
| type2_single_internal_segment | 299 | 0.698 | 0.546 |
| type3_multiple_internal_segments | 251 | 0.653 | 0.628 |
