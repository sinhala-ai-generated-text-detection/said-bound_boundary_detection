# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 3, 29.2 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.800 (sentence F1) and boundary bias +2.75 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.839 | 0.794 | 0.758 | 0.832 | 0.594 | 0.753 | 0.294 |
| seen generators | 0.856 | 0.820 | 0.771 | 0.877 | 0.618 | 0.773 | 0.346 |
| held-out (Gemini) | 0.805 | 0.736 | 0.730 | 0.742 | 0.548 | 0.717 | 0.190 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.844 | 0.793 | 0.781 | 0.806 | 0.575 | 0.743 | 0.317 |
| **viterbi (pair head)** | 0.839 | 0.794 | 0.758 | 0.832 | 0.594 | 0.753 | 0.294 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 1.0, consistency weight 1.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.588 | 1.61 | 0.144 | 0.932 | 0.652 | 0.935 | 0.472 |
| threshold | seen | 377 | 0.602 | 1.62 | 0.143 | 0.950 | 0.698 | 0.933 | 0.494 |
| threshold | held_out | 289 | 0.554 | 1.54 | 0.140 | 0.896 | 0.559 | 0.939 | 0.401 |
| viterbi | overall | 578 | 0.779 | 2.64 | 0.196 | 0.932 | 0.625 | 0.932 | 0.451 |
| viterbi | seen | 377 | 0.796 | 2.68 | 0.199 | 0.950 | 0.662 | 0.932 | 0.468 |
| viterbi | held_out | 289 | 0.754 | 2.53 | 0.190 | 0.896 | 0.548 | 0.933 | 0.380 |

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
| **xlm-roberta-base** | **0.794** | **0.594** | **0.736** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.808 | 0.594 |
| gemini_2_5_pro | held_out | 305 | 0.730 | 0.524 |
| gpt_4o | seen | 311 | 0.835 | 0.607 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.913 | 0.594 |
| type2_single_internal_segment | 299 | 0.672 | 0.518 |
| type3_multiple_internal_segments | 251 | 0.651 | 0.602 |
