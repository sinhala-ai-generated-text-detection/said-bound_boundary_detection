# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging, 12 epochs, lr 3e-05, effective batch 4, 24.6 min on GPU.

The whole document is encoded in one pass with a marker token before each sentence; the marker's hidden state is classified. This is the point of the model: unlike the linear baseline, each sentence representation is built with its neighbours in attention range, so the model can represent discontinuity rather than judging sentences in isolation.

Decision threshold 0.800, tuned on dev restricted to seen generators — the same protocol as the linear models, so the held-out number stays an honest generalisation estimate.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.790 | 0.718 | 0.690 | 0.748 | 0.513 | 0.702 | 0.178 |
| seen generators | 0.805 | 0.749 | 0.692 | 0.816 | 0.554 | 0.760 | 0.238 |
| held-out (Gemini) | 0.771 | 0.671 | 0.687 | 0.655 | 0.460 | 0.629 | 0.093 |

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
| **xlm-roberta-base** | **0.718** | **0.513** | **0.671** |

The row that matters is **bound F1 exact**: it is the only column the trivial baselines cannot game. `all-AI` scores 0.000 there by predicting no boundaries at all, and the ±1 column rewards scattering boundaries liberally.

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 52 | 0.761 | 0.562 |
| gemini_2_5_pro | held_out | 75 | 0.671 | 0.460 |
| gpt_4o | seen | 53 | 0.736 | 0.547 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 59 | 0.877 | 0.410 |
| type2_single_internal_segment | 73 | 0.588 | 0.493 |
| type3_multiple_internal_segments | 48 | 0.592 | 0.570 |
