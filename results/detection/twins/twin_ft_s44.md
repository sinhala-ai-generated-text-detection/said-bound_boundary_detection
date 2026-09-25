# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 3, 40.1 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.750 (sentence F1) and boundary bias +3.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.840 | 0.793 | 0.764 | 0.824 | 0.597 | 0.752 | 0.297 |
| seen generators | 0.858 | 0.822 | 0.778 | 0.870 | 0.620 | 0.771 | 0.351 |
| held-out (Gemini) | 0.804 | 0.732 | 0.733 | 0.732 | 0.555 | 0.716 | 0.190 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.847 | 0.796 | 0.788 | 0.804 | 0.578 | 0.744 | 0.326 |
| **viterbi (pair head)** | 0.840 | 0.793 | 0.764 | 0.824 | 0.597 | 0.752 | 0.297 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 1.0, consistency weight 1.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.571 | 1.59 | 0.137 | 0.929 | 0.662 | 0.932 | 0.476 |
| threshold | seen | 377 | 0.592 | 1.62 | 0.138 | 0.948 | 0.707 | 0.931 | 0.497 |
| threshold | held_out | 289 | 0.536 | 1.49 | 0.131 | 0.889 | 0.571 | 0.934 | 0.408 |
| viterbi | overall | 578 | 0.766 | 2.66 | 0.192 | 0.929 | 0.628 | 0.930 | 0.453 |
| viterbi | seen | 377 | 0.780 | 2.69 | 0.193 | 0.948 | 0.669 | 0.926 | 0.469 |
| viterbi | held_out | 289 | 0.747 | 2.56 | 0.188 | 0.889 | 0.545 | 0.937 | 0.384 |

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
| **xlm-roberta-base** | **0.793** | **0.597** | **0.732** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.813 | 0.603 |
| gemini_2_5_pro | held_out | 305 | 0.731 | 0.528 |
| gpt_4o | seen | 311 | 0.838 | 0.605 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.915 | 0.590 |
| type2_single_internal_segment | 299 | 0.693 | 0.538 |
| type3_multiple_internal_segments | 251 | 0.642 | 0.597 |
